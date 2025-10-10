# Setup Guide

This guide will help you set up the Spark ML project on your local machine.

## Prerequisites

### 1. Java Installation

**Windows:**
1. Download OpenJDK 8 or 11 from [AdoptOpenJDK](https://adoptopenjdk.net/)
2. Install and set JAVA_HOME environment variable
3. Add Java bin directory to PATH

**macOS:**
```bash
brew install openjdk@8
export JAVA_HOME=/usr/local/opt/openjdk@8
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

### 2. Python Installation

Install Python 3.7 or higher:
```bash
# Using conda
conda create -n spark_ml python=3.8
conda activate spark_ml

# Using pyenv
pyenv install 3.8.10
pyenv local 3.8.10
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd spark_ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and Setup Spark

**Option A: Automatic Setup (Recommended)**
```bash
# The project will automatically download Spark when you run the code
```

**Option B: Manual Setup**
```bash
# Download Spark
wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar xf spark-3.2.0-bin-hadoop3.2.tgz

# Set environment variables
export SPARK_HOME=$(pwd)/spark-3.2.0-bin-hadoop3.2
export PATH=$PATH:$SPARK_HOME/bin
```

### 5. Verify Installation

```python
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
print("Spark version:", spark.version)
spark.stop()
```

## Configuration

### Environment Variables

Add these to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
export JAVA_HOME=/path/to/java
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin
```

### Spark Configuration

For better performance, you can configure Spark in your code:

```python
spark = SparkSession.builder \
    .appName("SparkML") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
```

## Troubleshooting

### Common Issues

1. **Java not found:**
   - Ensure JAVA_HOME is set correctly
   - Verify Java version (8 or 11 recommended)

2. **Spark not found:**
   - Check SPARK_HOME environment variable
   - Ensure Spark is properly extracted

3. **Memory issues:**
   - Reduce the number of cores: `master("local[2]")`
   - Increase driver memory: `config("spark.driver.memory", "4g")`

4. **Permission issues:**
   - Ensure you have write permissions in the project directory
   - Check file permissions for data files

### Performance Tips

1. **Use appropriate number of cores:**
   ```python
   # For development
   spark = SparkSession.builder.master("local[2]").getOrCreate()
   
   # For production
   spark = SparkSession.builder.master("local[*]").getOrCreate()
   ```

2. **Optimize memory usage:**
   ```python
   spark.conf.set("spark.sql.adaptive.enabled", "true")
   spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
   ```

3. **Use caching for repeated operations:**
   ```python
   df.cache()
   ```

## Next Steps

1. Run the Jupyter notebook: `jupyter notebook notebooks/Spark_ML.ipynb`
2. Execute the Python scripts: `python src/regression.py`
3. Explore the data in the `data/` directory
4. Check the results in the `results/` directory

## Getting Help

- Check the [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- Review the [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- Open an issue in the repository for project-specific questions
