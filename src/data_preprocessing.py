"""
Data preprocessing utilities for Spark ML project.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat, lit, unix_timestamp, hour
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


def setup_spark_session(app_name="SparkML"):
    """
    Initialize and configure Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    
    # Set legacy time parser policy for older date formats
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    
    return spark


def load_taxi_data(spark, file_path):
    """
    Load and preprocess NYC taxi data.
    
    Args:
        spark (SparkSession): Spark session
        file_path (str): Path to the CSV file
        
    Returns:
        DataFrame: Preprocessed taxi data
    """
    # Load data
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Filter out negative values
    df = df.filter((col("fare") >= 0) & (col("tip") >= 0))
    
    # Remove outliers using 99th percentile
    fare_threshold = df.approxQuantile("fare", [0.99], 0.01)[0]
    df = df.filter(col("fare") <= fare_threshold)
    
    # IQR-based outlier removal
    Q1_distance, Q3_distance = df.approxQuantile("distance", [0.25, 0.75], 0.01)
    Q1_tip, Q3_tip = df.approxQuantile("tip", [0.25, 0.75], 0.01)
    Q1_fare, Q3_fare = df.approxQuantile("fare", [0.25, 0.75], 0.01)
    
    IQR_distance = Q3_distance - Q1_distance
    IQR_tip = Q3_tip - Q1_tip
    IQR_fare = Q3_fare - Q1_fare
    
    df = df.filter(
        (col("distance") >= Q1_distance - 1.5 * IQR_distance) & 
        (col("distance") <= Q3_distance + 1.5 * IQR_distance) &
        (col("tip") >= Q1_tip - 1.5 * IQR_tip) & 
        (col("tip") <= Q3_tip + 1.5 * IQR_tip) &
        (col("fare") >= Q1_fare - 1.5 * IQR_fare) & 
        (col("fare") <= Q3_fare + 1.5 * IQR_fare)
    )
    
    return df


def create_datetime_features(df):
    """
    Create datetime features from pickup and dropoff times.
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: DataFrame with datetime features
    """
    # Combine date and time columns
    df = df.withColumn('pickup_datetime',
        concat(col('pickup_date'), lit(' '), col('pickup_time')))
    df = df.withColumn('dropoff_datetime',
        concat(col('dropoff_date'), lit(' '), col('dropoff_time')))
    
    # Calculate trip duration in minutes
    df = df.withColumn('duration',
        (unix_timestamp(col('dropoff_datetime'), 'M/d/yyyy HH:mm') -
         unix_timestamp(col('pickup_datetime'), 'M/d/yyyy HH:mm')) / 60)
    
    return df


def create_time_features(df):
    """
    Create time-based features.
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: DataFrame with time features
    """
    # Extract hour from pickup time
    df = df.withColumn("pickup_hour", hour(col("pickup_time")))
    
    # Create time of day categories
    df = df.withColumn(
        "time_of_day",
        when(col("pickup_time").substr(1, 2).cast("int").between(4, 5), "Early Morning")
        .when(col("pickup_time").substr(1, 2).cast("int").between(6, 7), "Morning")
        .when(col("pickup_time").substr(1, 2).cast("int").between(8, 11), "Mid Morning")
        .when(col("pickup_time").substr(1, 2).cast("int").between(12, 13), "Noon")
        .when(col("pickup_time").substr(1, 2).cast("int").between(14, 16), "Afternoon")
        .when(col("pickup_time").substr(1, 2).cast("int").between(17, 19), "Evening")
        .when(col("pickup_time").substr(1, 2).cast("int").between(20, 23), "Night")
        .otherwise("Late Night")
    )
    
    # Create period labels
    df = df.withColumn("pickup_period",
        when(col("pickup_hour") < 12, concat(col("pickup_hour"), lit(" AM")))
        .when(col("pickup_hour") == 12, lit("12 PM"))
        .when(col("pickup_hour") > 12, concat((col("pickup_hour") - 12), lit(" PM"))))
    
    return df


def get_data_summary(df):
    """
    Get summary statistics for the dataset.
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_records': df.count(),
        'average_fare': df.agg({"fare": "avg"}).collect()[0][0],
        'average_tip': df.agg({"tip": "avg"}).collect()[0][0],
        'average_distance': df.agg({"distance": "avg"}).collect()[0][0]
    }
    
    return summary
