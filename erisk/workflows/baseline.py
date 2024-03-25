import luigi
from erisk.utils import spark_resource
from pyspark.sql import functions as F
from pyspark.ml.functions import array_to_vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import tqdm
import luigi.contrib.gcs
import json
import numpy as np
import time

from pyspark.sql import functions as F, Window
from pyspark.ml.functions import vector_to_array
from functools import reduce
from pyspark.ml import PipelineModel
from pyspark.ml.functions import array_to_vector
from pyspark.sql import functions as F


class TrainModel(luigi.Task):
    train_labels_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    eval_path = luigi.Parameter()
    train_percent = luigi.FloatParameter(default=80)
    feature_column = luigi.Parameter(default="tfidf")

    @staticmethod
    def pivot_training(df, feature_columns, target_prefix="target_"):
        scores = (
            df.withColumn(
                "query", F.concat(F.lit(target_prefix), F.col("query").cast("string"))
            )
            .groupBy("docid")
            .pivot("query")
            .agg(F.first("rel"))
        )
        # hash the docid to make it easy to split test train
        return (
            df.groupBy("docid")
            .agg(*[array_to_vector(F.first(c)).alias(c) for c in feature_columns])
            .join(scores, "docid", how="inner")
            .withColumn("sampleid", F.crc32(F.col("docid")) % 100)
        )

    @staticmethod
    def load_dataset(spark, train_labels_path, dataset_path):
        labels = spark.read.csv(train_labels_path, header=True, inferSchema=True)
        dataset = spark.read.parquet(dataset_path).withColumnRenamed("DOCNO", "docid")
        return dataset.join(labels, "docid", how="inner")

    @staticmethod
    def run_evaluation(df, targets, train_percent=80):
        results = {
            "f1": {
                "train": {},
                "test": {},
            },
            "accuracy": {"train": {}, "test": {}},
        }
        for c in tqdm.tqdm(targets):
            f1_evaluator = MulticlassClassificationEvaluator(
                labelCol=c, predictionCol=f"{c}_prediction", metricName="f1"
            )
            acc_evaluator = MulticlassClassificationEvaluator(
                labelCol=c, predictionCol=f"{c}_prediction", metricName="accuracy"
            )
            subset = df.where(f"{c} is not null")
            test_subset = subset.where(f"sampleid >= {train_percent}")
            train_subset = subset.where(f"sampleid < {train_percent}")
            results["f1"]["train"][c] = f1_evaluator.evaluate(train_subset)
            results["f1"]["test"][c] = f1_evaluator.evaluate(test_subset)
            results["accuracy"]["train"][c] = acc_evaluator.evaluate(train_subset)
            results["accuracy"]["test"][c] = acc_evaluator.evaluate(test_subset)

        mean_f1_test = np.mean(list(results["f1"]["test"].values()))
        mean_accuracy_test = np.mean(list(results["accuracy"]["test"].values()))
        mean_f1_train = np.mean(list(results["f1"]["train"].values()))
        mean_accuracy_train = np.mean(list(results["accuracy"]["train"].values()))
        results["mean_f1_test"] = mean_f1_test
        results["mean_accuracy_test"] = mean_accuracy_test
        results["mean_f1_train"] = mean_f1_train
        results["mean_accuracy_train"] = mean_accuracy_train
        return results

    @staticmethod
    def print_train_results(results):
        mean_f1_test = results["mean_f1_test"]
        mean_accuracy_test = results["mean_accuracy_test"]
        mean_f1_train = results["mean_f1_train"]
        mean_accuracy_train = results["mean_accuracy_train"]

        print(results)
        print(f"Mean F1 Test: {mean_f1_test}")
        print(f"Mean Accuracy Test: {mean_accuracy_test}")
        print(f"Mean F1 Train: {mean_f1_train}")
        print(f"Mean Accuracy Train: {mean_accuracy_train}")

    @staticmethod
    def build_nb_model_pipeline(feature, targets):
        pipeline = Pipeline(
            stages=[MinMaxScaler(inputCol=feature, outputCol=f"{feature}_scaled")]
            + [
                NaiveBayes(
                    labelCol=target,
                    featuresCol=f"{feature}_scaled",
                    predictionCol=f"{target}_prediction",
                    probabilityCol=f"{target}_probability",
                    rawPredictionCol=f"{target}_raw",
                    smoothing=1.0,
                )
                for target in targets
            ]
        )
        return pipeline

    def output(self):
        return [
            luigi.contrib.gcs.GCSTarget(self.model_path),
            luigi.contrib.gcs.GCSTarget(self.eval_path),
        ]

    def run(self):
        with spark_resource() as spark:
            start_time = time.time()
            df = self.pivot_training(
                self.load_dataset(spark, self.train_labels_path, self.dataset_path),
                [self.feature_column],
            ).cache()
            df.printSchema()
            pipeline = self.build_nb_model_pipeline(
                self.feature_column, [c for c in df.columns if c.startswith("target_")]
            )
            model = pipeline.fit(df.where(f"sampleid < {self.train_percent}").fillna(0))
            predictions = model.transform(df)

            # run the evaluation
            results = self.run_evaluation(
                predictions, [c for c in df.columns if c.startswith("target_")]
            )
            total_seconds = time.time() - start_time
            results["time"] = total_seconds
            self.print_train_results(results)
            client = luigi.contrib.gcs.GCSClient()
            client.put_string(
                json.dumps(results, indent=2),
                self.eval_path,
            )

            # retrain on the full dataset
            model = pipeline.fit(df.fillna(0))
            model.write().overwrite().save(self.model_path)


class RunInference(luigi.Task):
    model_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()
    feature_column = luigi.Parameter(default="tfidf")
    system_name = luigi.Parameter(default="baseline")
    k = luigi.IntParameter(default=1000)
    memory = luigi.Parameter(default="10g")

    @staticmethod
    def score_predictions(df, primary_key="DOCNO", k=1000, system_name="placeholder"):
        target_probs = [c for c in df.columns if "_probability" in c]
        target_probs_relevant = [vector_to_array(c)[1].alias(c) for c in target_probs]
        # try to increase partitions to ease memory constraints
        subset = df.select(primary_key, *target_probs_relevant).repartition(200).cache()
        # now for each target, we can compute the most relevant documents
        top_docs = []
        for c in target_probs:
            ordered = subset.select(
                F.lit(int(c.split("_")[1])).alias("symptom_number"),
                primary_key,
                F.col(c).alias("score"),
            )
            top_docs.append(ordered)

        # union all the documents together
        return (
            reduce(lambda a, b: a.union(b), top_docs)
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("symptom_number").orderBy(F.col("score").desc())
                ),
            )
            .where(F.col("rank") <= k)
            .select(
                "symptom_number",
                F.lit("Q0").alias("Qo"),
                F.col(primary_key).alias("sentence_id"),
                F.col("rank").alias("position_in_ranking"),
                "score",
                F.lit(system_name).alias("system_name"),
            )
        )

    def output(self):
        return [
            luigi.contrib.gcs.GCSTarget(f"{self.output_path}/predictions.csv"),
            luigi.contrib.gcs.GCSTarget(
                f"{self.output_path}/predictions_with_text.csv"
            ),
        ]

    def run(self):
        with spark_resource(memory=self.memory) as spark:
            model = PipelineModel.load(self.model_path)
            df = spark.read.parquet(self.dataset_path).withColumn(
                self.feature_column, array_to_vector(self.feature_column)
            )
            predictions = model.transform(df)
            predictions.printSchema()
            scored = self.score_predictions(
                predictions, k=self.k, system_name=self.system_name
            ).cache()
            # write the predictions to a csv file ready for submission
            scored_pd = scored.toPandas()
            scored_pd.to_csv(
                f"{self.output_path}/predictions.csv",
                index=False,
            )

            # also generate a human readable version of the predictions for debugging in csv
            scored_with_text = (
                df.select(
                    F.col("DOCNO").alias("sentence_id"), F.col("TEXT").alias("text")
                )
                .join(scored, "sentence_id", how="inner")
                .toPandas()
            )
            scored_with_text.to_csv(
                f"{self.output_path}/predictions_with_text.csv",
                index=False,
            )


class Workflow(luigi.Task):
    train_labels_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()
    memory = luigi.IntParameter(default="10g")

    def run(self):
        yield TrainModel(
            train_labels_path=self.train_labels_path,
            dataset_path=self.dataset_path,
            model_path=f"{self.output_path}/model",
            eval_path=f"{self.output_path}/eval.json",
        )
        yield RunInference(
            model_path=f"{self.output_path}/model",
            dataset_path=self.dataset_path,
            output_path=f"{self.output_path}/inference",
            memory=self.memory,
        )


if __name__ == "__main__":
    luigi.build(
        [
            Workflow(
                train_labels_path="gs://dsgt-clef-erisk-2024/task1/training/t1_training/TRAINING DATA (2023 COLLECTION)/g_rels_consenso.csv",
                dataset_path="gs://dsgt-clef-erisk-2024/task1/parquet/combined_tfidf",
                output_path="gs://dsgt-clef-erisk-2024/task1/processed/baseline_nb_tfidf",
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
