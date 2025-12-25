# Spark ML Project

A comprehensive Apache Spark Machine Learning project demonstrating various ML algorithms and techniques using PySpark with a unified CLI runner and automated metrics logging.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Algorithms & Results](#algorithms--results)
- [Metrics Output](#metrics-output)
- [Data](#data)
- [Documentation](#documentation)

## Overview

This project showcases machine learning implementations using Apache Spark ML, featuring:

- **Regression Analysis**: Linear Regression with R² evaluation
- **Classification**: Logistic Regression with full metrics suite
- **Clustering**: K-Means with WSSSE optimization
- **Topic Modeling**: Latent Dirichlet Allocation (LDA)
- **MLOps**: CLI runner with automated JSON metrics logging

## Features

- **One-Command Runner**: Execute any ML task with `python main.py --task <task>`
- **Automated Metrics Logging**: All results saved to `results/metrics.json`
- **Multiple ML Algorithms**: Regression, classification, clustering, and topic modeling
- **Hyperparameter Comparison**: Automatic comparison across different parameter values
- **Experiment Tracking**: Timestamped run history with full metrics

## Project Structure

```
spark_ml/
├── main.py                 # CLI runner (main entry point)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── sample_libsvm_data.txt
│   │   ├── sample_kmeans_data.txt
│   │   ├── sample_lda_libsvm_data.txt
│   │   ├── linearReg_data.txt
│   │   └── ...
│   └── processed/
├── notebooks/
│   └── Spark_ML.ipynb      # Interactive notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   └── topic_modeling.py
├── results/
│   └── metrics.json        # Auto-generated metrics
└── docs/
    ├── setup_guide.md
    └── algorithm_explanations.md
```

## Prerequisites

- Python 3.7+
- Java 8, 11, or 17
- Apache Spark 3.x
- PySpark

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spark_ml
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Java installation**
   ```bash
   java -version
   ```

## Quick Start

Run all ML tasks with a single command:

```bash
python main.py --task all
```

This will execute regression, classification, clustering, and topic modeling, then save all metrics to `results/metrics.json`.

## CLI Usage

### Available Commands

```bash
# Run individual tasks
python main.py --task regression
python main.py --task classification
python main.py --task clustering
python main.py --task topic_modeling

# Run all tasks
python main.py --task all

# Run without saving metrics
python main.py --task classification --no-save
```

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--task` | `-t` | ML task to run: `regression`, `classification`, `clustering`, `topic_modeling`, `all` |
| `--no-save` | | Skip saving metrics to JSON file |

### Example Output

```
============================================================
         SPARK ML PROJECT - CLI RUNNER
============================================================
  Task:      all
  Timestamp: 2025-12-25 02:49:09
============================================================

[*] Running ALL tasks...

============================================================
REGRESSION TASK: Linear Regression
============================================================
[1/5] Loading data...
[2/5] Splitting data (80/20)...
[3/5] Training Linear Regression model...
[4/5] Generating predictions...
[5/5] Calculating metrics...

----------------------------------------
REGRESSION RESULTS:
----------------------------------------
  MSE:        0.0679
  RMSE:       0.2606
  R²:         0.7274

... (continues for other tasks)

[OK] Metrics saved to results/metrics.json
```

## Algorithms & Results

### 1. Linear Regression

| Metric | Value |
|--------|-------|
| MSE | 0.0679 |
| RMSE | 0.2606 |
| MAE | 0.1733 |
| R² | 0.7274 |
| Features | 692 |
| Train/Test Split | 80/20 |

### 2. Logistic Regression (Classification)

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.0 |
| Recall | 1.0 |
| F1-Score | 1.0 |
| AUC-ROC | 1.0 |
| Samples | 100 |

### 3. K-Means Clustering

| Metric | Value |
|--------|-------|
| Best k | 3 |
| WSSSE (k=2) | 0.12 |
| WSSSE (k=3) | 0.075 |
| Improvement | 37.5% |

### 4. LDA Topic Modeling

| Metric | Value |
|--------|-------|
| Documents | 12 |
| Vocabulary Size | 11 |
| Topics (k) | 10 |
| Perplexity (k=5) | 2.78 |
| Perplexity (k=10) | 3.10 |

## Metrics Output

All metrics are automatically saved to `results/metrics.json`:

```json
{
  "runs": [...],
  "last_run": {
    "timestamp": "2025-12-25T02:49:16",
    "tasks": {
      "regression": {
        "model": "LinearRegression",
        "mse": 0.0679,
        "rmse": 0.2606,
        "r2": 0.7274,
        "train_time_sec": 5.014
      },
      "classification": {
        "model": "LogisticRegression",
        "accuracy": 1.0,
        "f1_score": 1.0,
        "auc_roc": 1.0
      },
      "clustering": {
        "model": "KMeans",
        "best_k": 3,
        "best_wssse": 0.075
      },
      "topic_modeling": {
        "model": "LDA",
        "final_perplexity": 3.0992
      }
    }
  }
}
```

## Data

The project uses LIBSVM format datasets located in `data/raw/`:

| File | Description | Samples |
|------|-------------|---------|
| `sample_libsvm_data.txt` | Classification/Regression | 100 |
| `sample_kmeans_data.txt` | Clustering | 6 |
| `sample_lda_libsvm_data.txt` | Topic Modeling | 12 |

## Interactive Notebook

For exploratory analysis, use the Jupyter notebook:

```bash
jupyter notebook notebooks/Spark_ML.ipynb
```

## Configuration

Key Spark configurations used:

```python
spark = SparkSession.builder \
    .appName("SparkML") \
    .master("local[1]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()
```

## Documentation

- **Setup Guide**: `docs/setup_guide.md`
- **Algorithm Explanations**: `docs/algorithm_explanations.md`

## Technologies Used

- **Apache Spark 3.x** - Distributed computing framework
- **PySpark ML** - Machine learning library
- **Python 3.7+** - Programming language
- **LIBSVM** - Data format for ML

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

**Note**: This project demonstrates distributed machine learning techniques using Apache Spark for educational purposes.
