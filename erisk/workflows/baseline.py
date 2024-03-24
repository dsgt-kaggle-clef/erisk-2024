import luigi
from erisk.utils import spark_resource
from pyspark.sql import functions as F
from pyspark.ml.functions import array_to_vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import tqdm
import luigi.contrib.gcs
import json
import numpy as np
import time


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


def load_dataset(spark, train_labels_path, dataset_path):
    labels = spark.read.csv(train_labels_path, header=True, inferSchema=True)
    dataset = spark.read.parquet(dataset_path).withColumnRenamed("DOCNO", "docid")
    return dataset.join(labels, "docid", how="inner")


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


class TrainModel(luigi.Task):
    train_labels_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    model_path = luigi.Parameter()
    eval_path = luigi.Parameter()
    train_percent = luigi.FloatParameter(default=80)

    def output(self):
        return [
            luigi.contrib.gcs.GCSTarget(self.model_path),
            luigi.contrib.gcs.GCSTarget(self.eval_path),
        ]

    def run(self):
        with spark_resource() as spark:
            start_time = time.time()
            df = pivot_training(
                load_dataset(spark, self.train_labels_path, self.dataset_path),
                ["hashingtf", "tfidf"],
            ).cache()
            df.printSchema()
            pipeline = build_nb_model_pipeline(
                "tfidf", [c for c in df.columns if c.startswith("target_")]
            )
            model = pipeline.fit(df.where(f"sampleid < {self.train_percent}").fillna(0))
            predictions = model.transform(df)

            # run the evaluation
            results = run_evaluation(
                predictions, [c for c in df.columns if c.startswith("target_")]
            )
            total_seconds = time.time() - start_time
            results["time"] = total_seconds
            print_train_results(results)
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

    def run(self):
        with spark_resource() as spark:
            model = spark.read.load(self.model_path)
            df = spark.read.parquet(self.dataset_path)
            predictions = model.transform(df)


class Workflow(luigi.Task):
    train_labels_path = luigi.Parameter()
    dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        yield TrainModel(
            train_labels_path=self.train_labels_path,
            dataset_path=self.dataset_path,
            model_path=f"{self.output_path}/model",
            eval_path=f"{self.output_path}/eval.json",
        )
        # yield RunInference(
        #     model_path=f"{self.output_path}/model",
        #     dataset_path=self.dataset_path,
        #     output_path=f"{self.output_path}/inference",
        # )


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
