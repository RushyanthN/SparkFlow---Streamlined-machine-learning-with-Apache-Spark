# Spark ML Project

A comprehensive Apache Spark Machine Learning project demonstrating various ML algorithms and techniques using PySpark.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project showcases machine learning implementations using Apache Spark and PySpark, including:

- **Regression Analysis**: Linear regression for fare prediction
- **Classification**: Logistic regression and SVM
- **Clustering**: K-means clustering
- **Topic Modeling**: Latent Dirichlet Allocation (LDA)
- **Performance Analysis**: Spark performance benchmarking

## ✨ Features

- **Multiple ML Algorithms**: Regression, classification, clustering, and topic modeling
- **Real-world Dataset**: NYC taxi fare prediction with comprehensive data analysis
- **Performance Benchmarking**: Comparative analysis of different models
- **Data Preprocessing**: Outlier detection, feature engineering, and data cleaning
- **Visualization**: Performance plots and data analysis charts

## 📁 Project Structure

```
spark_ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── Indy_rainfall.csv
│   │   ├── kmeans_data.txt
│   │   ├── linearReg_data.txt
│   │   ├── logistic_data.txt
│   │   ├── sample_kmeans_data.txt
│   │   ├── sample_lda_libsvm_data.txt
│   │   ├── sample_libsvm_data.txt
│   │   └── svm_data.txt
│   └── processed/
├── notebooks/
│   └── Spark_ML.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   └── topic_modeling.py
├── docs/
│   ├── setup_guide.md
│   └── algorithm_explanations.md
└── results/
    └── spark_performance.png
```

## 🛠 Prerequisites

- Python 3.7+
- Java 8 or 11
- Apache Spark 3.2.0
- Jupyter Notebook (for interactive analysis)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spark_ml
   ```

2. **Install Java (if not already installed)**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openjdk-8-jdk-headless
   
   # macOS
   brew install openjdk@8
   
   # Windows
   # Download and install from Oracle or OpenJDK
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Spark environment**
   ```bash
   # Download and extract Spark
   wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
   tar xf spark-3.2.0-bin-hadoop3.2.tgz
   
   # Set environment variables
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   export SPARK_HOME=spark-3.2.0-bin-hadoop3.2
   export PATH=$PATH:$SPARK_HOME/bin
   ```

## 🚀 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook notebooks/Spark_ML.ipynb
```

### Running Python Scripts

```bash
# Run regression analysis
python src/regression.py

# Run classification
python src/classification.py

# Run clustering
python src/clustering.py
```

### Interactive Spark Session

```python
from pyspark.sql import SparkSession
import findspark

findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()
```

## 📊 Data

The project uses several datasets:

- **NYC Taxi Data**: For fare prediction and analysis
- **Sample Datasets**: Various ML algorithm demonstrations
- **Rainfall Data**: Environmental data analysis

All data files are located in the `data/raw/` directory.

## 🤖 Algorithms Implemented

### 1. Linear Regression
- **Purpose**: Fare prediction based on distance and duration
- **Features**: Distance, duration, time of day
- **Performance**: R² score evaluation

### 2. Logistic Regression
- **Purpose**: Binary classification
- **Features**: Multi-dimensional feature vectors
- **Evaluation**: Training error analysis

### 3. K-means Clustering
- **Purpose**: Data clustering and pattern recognition
- **Parameters**: Configurable number of clusters
- **Visualization**: Cluster center analysis

### 4. Latent Dirichlet Allocation (LDA)
- **Purpose**: Topic modeling and text analysis
- **Parameters**: Number of topics, iterations
- **Output**: Topic distributions and term weights

## 📈 Results

The project includes performance analysis comparing different models:

- **Model M1**: Distance-based fare prediction
- **Model M2**: Distance + duration-based prediction
- **Performance Metrics**: R² score, execution time analysis
- **Visualization**: Performance comparison charts

## 🔧 Configuration

Key configuration parameters:

```python
# Spark Configuration
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Model Parameters
NUM_SAMPLES = 1000000  # For Monte Carlo simulation
K_CLUSTERS = 2         # For K-means
LDA_TOPICS = 10        # For topic modeling
```

## 📚 Documentation

- **Setup Guide**: `docs/setup_guide.md`
- **Algorithm Explanations**: `docs/algorithm_explanations.md`
- **API Reference**: Inline code documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project is for educational purposes and demonstrates various machine learning techniques using Apache Spark.
