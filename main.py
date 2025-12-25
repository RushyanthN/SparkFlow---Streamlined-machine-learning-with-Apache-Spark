import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Set environment variables BEFORE importing pyspark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import (
    RegressionEvaluator, 
    MulticlassClassificationEvaluator, 
    BinaryClassificationEvaluator
)

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_FILE = RESULTS_DIR / "metrics.json"


def setup_spark_session(app_name="SparkMLCLI"):
    """Initialize Spark session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_metrics():
    """Load existing metrics from file."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {"runs": []}


def save_metrics(metrics):
    """Save metrics to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Metrics saved to {METRICS_FILE}")


def run_regression(spark):
    """
    Run regression task using Linear Regression (DataFrame API).
    Uses sample_libsvm_data.txt for regression (treating labels as continuous).
    """
    print("\n" + "=" * 60)
    print("REGRESSION TASK: Linear Regression")
    print("=" * 60)
    
    # Use LIBSVM data for regression demo (native Spark loading, no Python workers)
    data_path = str(DATA_DIR / "sample_libsvm_data.txt")
    
    print(f"\n[1/5] Loading data from {data_path}")
    start_time = time.time()
    
    # Native Spark loading - no Python worker issues
    df = spark.read.format("libsvm").load(data_path)
    num_samples = df.count()
    num_features = df.select("features").first()[0].size
    load_time = time.time() - start_time
    print(f"      Samples: {num_samples}, Features: {num_features}")
    
    # Split data
    print("\n[2/5] Splitting data (80/20)...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"      Train: {train_count}, Test: {test_count}")
    
    # Train model
    print("\n[3/5] Training Linear Regression model...")
    start_time = time.time()
    lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100)
    model = lr.fit(train_df)
    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.2f}s")
    
    # Make predictions
    print("\n[4/5] Generating predictions...")
    predictions = model.transform(test_df)
    
    # Evaluate
    print("\n[5/5] Calculating metrics...")
    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mae"
    )
    
    rmse = evaluator_rmse.evaluate(predictions)
    mse = evaluator_mse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    # Get coefficients
    coefficients = model.coefficients.toArray().tolist()[:3]  # First 3 coeffs
    intercept = model.intercept
    
    metrics = {
        "task": "regression",
        "model": "LinearRegression",
        "num_samples": num_samples,
        "train_samples": train_count,
        "test_samples": test_count,
        "num_features": num_features,
        "max_iter": 100,
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "intercept": round(intercept, 4),
        "coefficients_sample": [round(c, 4) for c in coefficients],
        "load_time_sec": round(load_time, 3),
        "train_time_sec": round(train_time, 3)
    }
    
    print("\n" + "-" * 40)
    print("REGRESSION RESULTS:")
    print("-" * 40)
    print(f"  MSE:        {metrics['mse']}")
    print(f"  RMSE:       {metrics['rmse']}")
    print(f"  MAE:        {metrics['mae']}")
    print(f"  R2:         {metrics['r2']}")
    print(f"  Intercept:  {metrics['intercept']}")
    print(f"  Train Time: {metrics['train_time_sec']}s")
    
    return metrics


def run_classification(spark):
    """
    Run classification task using Logistic Regression (DataFrame API).
    Returns metrics dict.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION TASK: Logistic Regression")
    print("=" * 60)
    
    data_path = str(DATA_DIR / "sample_libsvm_data.txt")
    
    # Load data in LIBSVM format
    print(f"\n[1/5] Loading data from {data_path}")
    start_time = time.time()
    df = spark.read.format("libsvm").load(data_path)
    num_samples = df.count()
    load_time = time.time() - start_time
    print(f"      Samples: {num_samples}")
    
    # Split data
    print("\n[2/5] Splitting data (80/20)...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"      Train: {train_count}, Test: {test_count}")
    
    # Train model
    print("\n[3/5] Training Logistic Regression model...")
    start_time = time.time()
    lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100)
    model = lr.fit(train_df)
    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.2f}s")
    
    # Make predictions
    print("\n[4/5] Generating predictions on test set...")
    predictions = model.transform(test_df)
    
    # Evaluate
    print("\n[5/5] Calculating metrics...")
    
    # Accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)
    
    # Precision, Recall, F1
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = evaluator_prec.evaluate(predictions)
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    
    # AUC (for binary classification)
    try:
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        auc = evaluator_auc.evaluate(predictions)
    except:
        auc = None
    
    metrics = {
        "task": "classification",
        "model": "LogisticRegression",
        "num_samples": num_samples,
        "train_samples": train_count,
        "test_samples": test_count,
        "max_iter": 100,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if auc else None,
        "load_time_sec": round(load_time, 3),
        "train_time_sec": round(train_time, 3)
    }
    
    print("\n" + "-" * 40)
    print("CLASSIFICATION RESULTS:")
    print("-" * 40)
    print(f"  Accuracy:   {metrics['accuracy']}")
    print(f"  Precision:  {metrics['precision']}")
    print(f"  Recall:     {metrics['recall']}")
    print(f"  F1-Score:   {metrics['f1_score']}")
    if auc:
        print(f"  AUC-ROC:    {metrics['auc_roc']}")
    print(f"  Train Time: {metrics['train_time_sec']}s")
    
    return metrics


def run_clustering(spark):
    """
    Run clustering task using K-Means (DataFrame API).
    Returns metrics dict.
    """
    print("\n" + "=" * 60)
    print("CLUSTERING TASK: K-Means Clustering")
    print("=" * 60)
    
    data_path = str(DATA_DIR / "sample_kmeans_data.txt")
    
    # Load LIBSVM format data
    print(f"\n[1/4] Loading data from {data_path}")
    start_time = time.time()
    df = spark.read.format("libsvm").load(data_path)
    num_samples = df.count()
    load_time = time.time() - start_time
    print(f"      Samples: {num_samples}")
    
    # Train K-Means with different k values
    k_values = [2, 3]
    best_k = None
    best_wssse = float('inf')
    best_model = None
    results_by_k = {}
    
    print("\n[2/4] Training K-Means models...")
    for k in k_values:
        print(f"\n      Training with k={k}...")
        start_time = time.time()
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=20)
        model = kmeans.fit(df)
        train_time = time.time() - start_time
        
        # Calculate WSSSE (Within Set Sum of Squared Errors)
        wssse = model.summary.trainingCost
        
        results_by_k[k] = {
            "wssse": round(wssse, 4),
            "train_time_sec": round(train_time, 3)
        }
        
        print(f"      k={k}: WSSSE={wssse:.4f}, Time={train_time:.2f}s")
        
        if wssse < best_wssse:
            best_wssse = wssse
            best_k = k
            best_model = model
    
    # Get cluster centers for best model
    print(f"\n[3/4] Best model: k={best_k}")
    centers = best_model.clusterCenters()
    
    # Predict clusters
    print("\n[4/4] Generating cluster assignments...")
    predictions = best_model.transform(df)
    cluster_counts = predictions.groupBy("prediction").count().collect()
    cluster_distribution = {int(row["prediction"]): int(row["count"]) for row in cluster_counts}
    
    metrics = {
        "task": "clustering",
        "model": "KMeans",
        "num_samples": num_samples,
        "k_values_tested": k_values,
        "best_k": best_k,
        "best_wssse": round(best_wssse, 4),
        "results_by_k": results_by_k,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": [[round(float(c), 4) for c in center] for center in centers],
        "load_time_sec": round(load_time, 3)
    }
    
    print("\n" + "-" * 40)
    print("CLUSTERING RESULTS:")
    print("-" * 40)
    print(f"  Best k:     {metrics['best_k']}")
    print(f"  WSSSE:      {metrics['best_wssse']}")
    print(f"  Clusters:   {cluster_distribution}")
    
    return metrics


def run_topic_modeling(spark):
    """
    Run topic modeling task using LDA (DataFrame API).
    Returns metrics dict.
    """
    print("\n" + "=" * 60)
    print("TOPIC MODELING TASK: Latent Dirichlet Allocation (LDA)")
    print("=" * 60)
    
    data_path = str(DATA_DIR / "sample_lda_libsvm_data.txt")
    
    # Load data
    print(f"\n[1/4] Loading data from {data_path}")
    start_time = time.time()
    df = spark.read.format("libsvm").load(data_path)
    num_docs = df.count()
    vocab_size = df.select("features").first()[0].size
    load_time = time.time() - start_time
    print(f"      Documents: {num_docs}, Vocabulary Size: {vocab_size}")
    
    # Train LDA with different k values
    k_values = [5, 10]
    results_by_k = {}
    
    print("\n[2/4] Training LDA models...")
    for k in k_values:
        print(f"\n      Training with k={k} topics...")
        start_time = time.time()
        lda = LDA(k=k, maxIter=10, seed=42)
        model = lda.fit(df)
        train_time = time.time() - start_time
        
        # Calculate perplexity (lower is better)
        perplexity = model.logPerplexity(df)
        log_likelihood = model.logLikelihood(df)
        
        results_by_k[k] = {
            "perplexity": round(perplexity, 4),
            "log_likelihood": round(log_likelihood, 4),
            "train_time_sec": round(train_time, 3)
        }
        
        print(f"      k={k}: Perplexity={perplexity:.4f}, Time={train_time:.2f}s")
    
    # Use k=10 as final model
    print("\n[3/4] Getting topic descriptions (k=10)...")
    final_lda = LDA(k=10, maxIter=10, seed=42)
    final_model = final_lda.fit(df)
    
    # Get topics
    topics_df = final_model.describeTopics(3)
    topics_data = topics_df.collect()
    topics_summary = []
    for row in topics_data:
        topics_summary.append({
            "topic_id": int(row["topic"]),
            "term_indices": [int(i) for i in row["termIndices"]],
            "term_weights": [round(float(w), 4) for w in row["termWeights"]]
        })
    
    # Transform documents
    print("\n[4/4] Generating document-topic distributions...")
    transformed = final_model.transform(df)
    
    final_perplexity = final_model.logPerplexity(df)
    
    metrics = {
        "task": "topic_modeling",
        "model": "LDA",
        "num_documents": num_docs,
        "vocabulary_size": vocab_size,
        "k_values_tested": k_values,
        "results_by_k": results_by_k,
        "final_model_k": 10,
        "final_perplexity": round(final_perplexity, 4),
        "topics_sample": topics_summary[:3],
        "load_time_sec": round(load_time, 3)
    }
    
    print("\n" + "-" * 40)
    print("TOPIC MODELING RESULTS:")
    print("-" * 40)
    print(f"  Documents:    {metrics['num_documents']}")
    print(f"  Vocab Size:   {metrics['vocabulary_size']}")
    print(f"  Final k:      {metrics['final_model_k']}")
    print(f"  Perplexity:   {metrics['final_perplexity']}")
    
    return metrics


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Spark ML Project - Command Line Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task regression
  python main.py --task classification  
  python main.py --task clustering
  python main.py --task topic_modeling
  python main.py --task all
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        choices=["regression", "classification", "clustering", "topic_modeling", "all"],
        help="ML task to run"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save metrics to file"
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "=" * 60)
    print("         SPARK ML PROJECT - CLI RUNNER")
    print("=" * 60)
    print(f"  Task:      {args.task}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize Spark
    print("\n[*] Initializing Spark session...")
    spark = setup_spark_session(f"SparkML_{args.task}")
    
    # Task mapping
    task_runners = {
        "regression": run_regression,
        "classification": run_classification,
        "clustering": run_clustering,
        "topic_modeling": run_topic_modeling
    }
    
    all_metrics = load_metrics()
    run_metrics = {
        "timestamp": datetime.now().isoformat(),
        "tasks": {}
    }
    
    try:
        if args.task == "all":
            print("\n[*] Running ALL tasks...")
            for task_name, runner in task_runners.items():
                metrics = runner(spark)
                run_metrics["tasks"][task_name] = metrics
        else:
            metrics = task_runners[args.task](spark)
            run_metrics["tasks"][args.task] = metrics
        
        # Save metrics
        if not args.no_save:
            all_metrics["runs"].append(run_metrics)
            all_metrics["last_run"] = run_metrics
            save_metrics(all_metrics)
        
        # Summary
        print("\n" + "=" * 60)
        print("                    SUMMARY")
        print("=" * 60)
        for task_name, task_metrics in run_metrics["tasks"].items():
            print(f"\n  [{task_name.upper()}]")
            if task_name == "regression":
                print(f"    MSE: {task_metrics['mse']}, R2: {task_metrics['r2']}")
            elif task_name == "classification":
                print(f"    Accuracy: {task_metrics['accuracy']}, F1: {task_metrics['f1_score']}")
            elif task_name == "clustering":
                print(f"    Best k: {task_metrics['best_k']}, WSSSE: {task_metrics['best_wssse']}")
            elif task_name == "topic_modeling":
                print(f"    Perplexity: {task_metrics['final_perplexity']}")
        
        print("\n" + "=" * 60)
        print("[OK] All tasks completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        spark.stop()
        print("\n[*] Spark session stopped")


if __name__ == "__main__":
    main()
