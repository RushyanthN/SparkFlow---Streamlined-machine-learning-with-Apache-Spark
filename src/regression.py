"""
Regression analysis module for Spark ML project.
"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import time
import matplotlib.pyplot as plt


class TaxiFarePredictor:
    """
    Taxi fare prediction using linear regression.
    """
    
    def __init__(self, spark_session):
        """
        Initialize the predictor.
        
        Args:
            spark_session (SparkSession): Spark session
        """
        self.spark = spark_session
        self.model_m1 = None
        self.model_m2 = None
        self.assembler_m1 = None
        self.assembler_m2 = None
        
    def prepare_model_m1(self, df):
        """
        Prepare Model M1: Distance-based fare prediction.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (model, assembler, predictions)
        """
        # Feature assembly for M1
        self.assembler_m1 = VectorAssembler(
            inputCols=["distance"], 
            outputCol="features"
        )
        df_assembled = self.assembler_m1.transform(df)
        
        # Train-test split
        train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
        
        # Train model
        lr = LinearRegression(featuresCol="features", labelCol="fare")
        self.model_m1 = lr.fit(train_data)
        
        # Make predictions
        predictions = self.model_m1.transform(test_data)
        
        return self.model_m1, self.assembler_m1, predictions
    
    def prepare_model_m2(self, df):
        """
        Prepare Model M2: Distance + duration-based fare prediction.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (model, assembler, predictions)
        """
        # Feature assembly for M2
        self.assembler_m2 = VectorAssembler(
            inputCols=["distance", "duration"], 
            outputCol="features"
        )
        df_assembled = self.assembler_m2.transform(df)
        
        # Train-test split
        train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
        
        # Train model
        lr = LinearRegression(featuresCol="features", labelCol="fare")
        self.model_m2 = lr.fit(train_data)
        
        # Make predictions
        predictions = self.model_m2.transform(test_data)
        
        return self.model_m2, self.assembler_m2, predictions
    
    def evaluate_model(self, predictions, model_name="Model"):
        """
        Evaluate model performance.
        
        Args:
            predictions (DataFrame): Predictions DataFrame
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        evaluator_r2 = RegressionEvaluator(
            labelCol="fare", 
            predictionCol="prediction", 
            metricName="r2"
        )
        evaluator_rmse = RegressionEvaluator(
            labelCol="fare", 
            predictionCol="prediction", 
            metricName="rmse"
        )
        
        r2 = evaluator_r2.evaluate(predictions)
        rmse = evaluator_rmse.evaluate(predictions)
        
        print(f"R-squared ({model_name}): {r2:.4f}")
        print(f"RMSE ({model_name}): {rmse:.4f}")
        
        return {"r2": r2, "rmse": rmse}
    
    def predict_fare_m1(self, distance):
        """
        Predict fare using Model M1.
        
        Args:
            distance (float): Trip distance
            
        Returns:
            float: Predicted fare
        """
        if self.model_m1 is None or self.assembler_m1 is None:
            raise ValueError("Model M1 not trained yet. Call prepare_model_m1 first.")
        
        temp_df = self.spark.createDataFrame([(distance,)], ["distance"])
        temp_df = self.assembler_m1.transform(temp_df)
        prediction = self.model_m1.transform(temp_df)
        
        return prediction.select('prediction').first()[0]
    
    def predict_fare_m2(self, distance, duration):
        """
        Predict fare using Model M2.
        
        Args:
            distance (float): Trip distance
            duration (float): Trip duration in minutes
            
        Returns:
            float: Predicted fare
        """
        if self.model_m2 is None or self.assembler_m2 is None:
            raise ValueError("Model M2 not trained yet. Call prepare_model_m2 first.")
        
        temp_df = self.spark.createDataFrame([(distance, duration)], ["distance", "duration"])
        temp_df = self.assembler_m2.transform(temp_df)
        prediction = self.model_m2.transform(temp_df)
        
        return prediction.select('prediction').first()[0]
    
    def compare_trips(self, trip1_distance, trip1_duration, trip2_distance, trip2_duration):
        """
        Compare two trips using Model M2.
        
        Args:
            trip1_distance (float): First trip distance
            trip1_duration (float): First trip duration
            trip2_distance (float): Second trip distance
            trip2_duration (float): Second trip duration
            
        Returns:
            dict: Comparison results
        """
        fare1 = self.predict_fare_m2(trip1_distance, trip1_duration)
        fare2 = self.predict_fare_m2(trip2_distance, trip2_duration)
        
        result = {
            "trip1": {"distance": trip1_distance, "duration": trip1_duration, "fare": fare1},
            "trip2": {"distance": trip2_distance, "duration": trip2_duration, "fare": fare2}
        }
        
        if fare1 > fare2:
            result["higher_fare"] = "trip1"
        elif fare2 > fare1:
            result["higher_fare"] = "trip2"
        else:
            result["higher_fare"] = "equal"
        
        return result
    
    def performance_benchmark(self, df, fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        """
        Benchmark model performance across different data sizes.
        
        Args:
            df (DataFrame): Input DataFrame
            fractions (list): Data fractions to test
            
        Returns:
            dict: Performance results
        """
        times_m1 = []
        times_m2 = []
        
        for fraction in fractions:
            # Sample data
            sampled_df = df.sample(fraction=fraction, seed=42)
            
            # Time M1
            def run_m1(data):
                assembler = VectorAssembler(inputCols=["distance"], outputCol="features")
                df_assembled = assembler.transform(data)
                train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
                lr = LinearRegression(featuresCol="features", labelCol="fare")
                model = lr.fit(train_data)
                predictions = model.transform(test_data)
                return predictions
            
            # Time M2
            def run_m2(data):
                assembler = VectorAssembler(inputCols=["distance", "duration"], outputCol="features")
                df_assembled = assembler.transform(data)
                train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
                lr = LinearRegression(featuresCol="features", labelCol="fare")
                model = lr.fit(train_data)
                predictions = model.transform(test_data)
                return predictions
            
            # Measure execution time
            start_time = time.time()
            run_m1(sampled_df)
            times_m1.append(time.time() - start_time)
            
            start_time = time.time()
            run_m2(sampled_df)
            times_m2.append(time.time() - start_time)
        
        # Create performance plot
        plt.figure(figsize=(10, 6))
        plt.plot(fractions, times_m1, label="M1 (distance only)", marker='o')
        plt.plot(fractions, times_m2, label="M2 (distance and duration)", marker='s')
        plt.xlabel("Fraction of Data")
        plt.ylabel("Time (seconds)")
        plt.title("Spark Performance Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/spark_performance.png")
        plt.show()
        
        return {
            "fractions": fractions,
            "times_m1": times_m1,
            "times_m2": times_m2
        }
