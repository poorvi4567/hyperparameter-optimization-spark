"""
Phase 1 - Step 1: Data Loading & Preprocessing
Dataset: UCI Adult Income (predict if income > $50K)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline


def create_spark_session(app_name="HPO_Phase1"):
    """Create and return a SparkSession with event logging enabled."""
    import os
    log_dir = "/tmp/spark-events"
    os.makedirs(log_dir, exist_ok=True)

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", f"file://{log_dir}") \
        .config("spark.history.fs.logDirectory", f"file://{log_dir}") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_adult_dataset(spark):
    """
    Load the UCI Adult Income dataset.
    Downloads directly from UCI ML Repository.
    Columns: age, workclass, fnlwgt, education, education-num,
             marital-status, occupation, relationship, race, sex,
             capital-gain, capital-loss, hours-per-week, native-country, income
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    import pandas as pd
    pdf = pd.read_csv(url, header=None, names=column_names,
                      skipinitialspace=True, na_values="?")

    df = spark.createDataFrame(pdf)
    return df


def preprocess(df):
    """
    Clean and prepare the dataset for ML:
    - Drop rows with nulls
    - Encode label: income => 0 or 1
    - StringIndex + OneHotEncode categorical columns
    - Assemble feature vector
    Returns: processed DataFrame, fitted pipeline, feature column names
    """
    # Drop nulls
    df = df.dropna()

    # Encode target label: ">50K" -> 1.0, "<=50K" -> 0.0
    df = df.withColumn(
        "label",
        F.when(F.col("income").contains(">50K"), 1.0).otherwise(0.0)
    ).drop("income")

    # Cast numeric columns explicitly
    numeric_cols = ["age", "fnlwgt", "education_num",
                    "capital_gain", "capital_loss", "hours_per_week"]
    for col in numeric_cols:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))

    # Categorical columns to encode
    categorical_cols = [
        "workclass", "education", "marital_status",
        "occupation", "relationship", "race", "sex", "native_country"
    ]

    # Step 1: StringIndexer for each categorical col
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in categorical_cols
    ]

    # Step 2: OneHotEncoder for each indexed col
    encoders = [
        OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")
        for c in categorical_cols
    ]

    # Final feature columns
    ohe_cols = [c + "_ohe" for c in categorical_cols]
    all_feature_cols = numeric_cols + ohe_cols

    # Step 3: VectorAssembler
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features"
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df).select("features", "label")

    return processed_df, pipeline_model, all_feature_cols


def split_data(processed_df, train_ratio=0.8, seed=42):
    """Split into train and test sets."""
    train_df, test_df = processed_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    train_df.cache()
    test_df.cache()
    return train_df, test_df


if __name__ == "__main__":
    spark = create_spark_session()

    print("=" * 50)
    print("Loading Adult Income dataset...")
    df = load_adult_dataset(spark)
    print(f"Raw rows: {df.count()}")
    df.printSchema()
    df.show(5)

    print("\nPreprocessing...")
    processed_df, pipeline_model, feature_cols = preprocess(df)
    print(f"Processed rows: {processed_df.count()}")
    processed_df.show(3)

    print("\nSplitting into train/test...")
    train_df, test_df = split_data(processed_df)
    print(f"Train rows: {train_df.count()} | Test rows: {test_df.count()}")

    # Class balance check
    print("\nLabel distribution in train set:")
    train_df.groupBy("label").count().show()

    spark.stop()
    print("Phase 1 - Data loading complete.")