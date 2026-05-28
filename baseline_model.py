"""
Phase 1 - Step 2: Baseline Model Training & Evaluation
Model: Gradient Boosted Trees (GBTClassifier) via Spark MLlib
This is the baseline BEFORE any HPO — single fixed config.
"""

import time
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql import functions as F


# ──────────────────────────────────────────────
# Default hyperparameter config (our baseline)
# ──────────────────────────────────────────────
BASELINE_CONFIG = {
    "maxIter": 50,           # n_estimators (budget)
    "maxDepth": 5,           # tree depth
    "stepSize": 0.1,         # learning rate
    "subsamplingRate": 0.8,  # row subsampling
    "featureSubsetStrategy": "sqrt",  # feature sampling
    "minInstancesPerNode": 1,         # min samples per leaf
}


def build_gbt(config=None):
    """
    Build a GBTClassifier with the given config.
    Falls back to BASELINE_CONFIG if none provided.
    """
    cfg = config or BASELINE_CONFIG
    return GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=cfg["maxIter"],
        maxDepth=cfg["maxDepth"],
        stepSize=cfg["stepSize"],
        subsamplingRate=cfg["subsamplingRate"],
        featureSubsetStrategy=cfg["featureSubsetStrategy"],
        minInstancesPerNode=cfg["minInstancesPerNode"],
        seed=42
    )


def train_model(train_df, config=None):
    """Train a GBT model and return the fitted model + time taken."""
    gbt = build_gbt(config)
    start = time.time()
    model = gbt.fit(train_df)
    elapsed = time.time() - start
    return model, elapsed


def evaluate_model(model, test_df):
    """
    Evaluate a trained GBT model.
    Returns dict of: accuracy, f1, auc, precision, recall
    """
    predictions = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    multi_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )

    auc = binary_eval.evaluate(predictions)
    accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
    f1 = multi_eval.setMetricName("f1").evaluate(predictions)
    precision = multi_eval.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multi_eval.setMetricName("weightedRecall").evaluate(predictions)

    return {
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def print_metrics(metrics, config, elapsed):
    """Pretty-print model results."""
    print("\n" + "=" * 50)
    print("BASELINE MODEL RESULTS")
    print("=" * 50)
    print(f"  Config used:")
    for k, v in config.items():
        print(f"    {k:30s}: {v}")
    print(f"\n  Training time : {elapsed:.2f}s")
    print(f"  AUC           : {metrics['auc']}")
    print(f"  Accuracy      : {metrics['accuracy']}")
    print(f"  F1 Score      : {metrics['f1']}")
    print(f"  Precision     : {metrics['precision']}")
    print(f"  Recall        : {metrics['recall']}")
    print("=" * 50)


def run_baseline(train_df, test_df):
    """Full baseline run: train + evaluate + print."""
    print("\nTraining baseline GBT model...")
    model, elapsed = train_model(train_df, BASELINE_CONFIG)
    metrics = evaluate_model(model, test_df)
    print_metrics(metrics, BASELINE_CONFIG, elapsed)
    return model, metrics, elapsed


if __name__ == "__main__":
    from data_loader import create_spark_session, load_adult_dataset, preprocess, split_data

    spark = create_spark_session()
    df = load_adult_dataset(spark)
    processed_df, _, _ = preprocess(df)
    train_df, test_df = split_data(processed_df)

    model, metrics, elapsed = run_baseline(train_df, test_df)

    spark.stop()