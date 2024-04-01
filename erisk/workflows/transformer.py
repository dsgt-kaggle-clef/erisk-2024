from argparse import ArgumentParser

import luigi
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from erisk.utils import spark_resource
from erisk.workflows.baseline import (
    FormatTrec,
    RunInference,
    TrainLogisticRegressionModel,
)
from erisk.workflows.utils import WrappedSentenceTransformer


class ProcessSubsetBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=200)

    def output(self):
        # save both the model pipeline and the dataset
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)
        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": max(self.num_partitions, 200)}
        ) as spark:
            df = spark.read.parquet(self.input_path)
            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)
            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(f"{self.output_path}/data")

        # now write the success file
        with self.output().open("w") as f:
            f.write("")


class ProcessSentenceTransformer(ProcessSubsetBase):
    model_name = luigi.Parameter(default="all-MiniLM-L6-v2")
    batch_size = luigi.IntParameter(default=8)

    @property
    def feature_columns(self):
        return ["transformer"]

    def pipeline(self):
        return Pipeline(
            stages=[
                WrappedSentenceTransformer(
                    input_col="text",
                    output_col="transformer",
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                )
            ]
        )


class Workflow(luigi.Task):
    train_labels_path = luigi.Parameter()
    input_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()
    version = "v4"

    def run(self):
        # run the transformer task
        yield ProcessSentenceTransformer(
            input_path=self.input_path,
            output_path=self.dataset_path,
        )
        # now run logistic regression on word2vec and bert embeddings
        for feature_column in ["word2vec", "transformer"]:
            yield TrainLogisticRegressionModel(
                train_labels_path=self.train_labels_path,
                dataset_path=f"{self.dataset_path}/data",
                model_path=f"{self.output_path}/{feature_column}_logistic/model",
                eval_path=f"{self.output_path}/{feature_column}_logistic/eval.json",
                feature_column=feature_column,
            )
            yield RunInference(
                dataset_path=f"{self.dataset_path}/data",
                model_path=f"{self.output_path}/{feature_column}_logistic/{self.version}/model",
                output_path=f"{self.output_path}/{feature_column}_logistic/{self.version}/inference",
                feature_column=feature_column,
            )
            yield FormatTrec(
                input_path=f"{self.output_path}/{feature_column}_logistic/{self.version}/inference/predictions.csv",
                output_path=f"{self.output_path}/{feature_column}_logistic/{self.version}/inference/predictions.trec",
                system_name=f"{feature_column}_logistic_{self.version}",
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--scheduler-host", default="services.us-central1-a.c.dsgt-clef-2024.internal"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_uri = "/mnt/data/erisk/task1"

    luigi.build(
        [
            Workflow(
                train_labels_path=f"{base_uri}/training/t1_training/TRAINING DATA (2023 COLLECTION)/g_rels_consenso.csv",
                input_path=f"{base_uri}/processed/data/word2vec_relevant/v1",
                dataset_path=f"{base_uri}/processed/data/transformer_relevant/v1",
                output_path=f"{base_uri}/processed/eval",
            )
        ],
        scheduler_host=args.scheduler_host,
    )
