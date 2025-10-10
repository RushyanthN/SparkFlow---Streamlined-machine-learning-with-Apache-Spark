"""
Clustering module for Spark ML project.
"""

from pyspark.ml.clustering import KMeans as MLKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.clustering import KMeans as MLlibKMeans
from pyspark.sql import SparkSession
import numpy as np


class ClusteringModels:
    """
    Clustering models using both MLlib and ML APIs.
    """
    
    def __init__(self, spark_session):
        """
        Initialize the clustering models.
        
        Args:
            spark_session (SparkSession): Spark session
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        
    def parse_point_kmeans(self, line):
        """
        Parse line for K-means clustering.
        
        Args:
            line (str): Input line in format "label feature1:value1 feature2:value2 ..."
            
        Returns:
            list: Feature values
        """
        values = [float(x.split(':')[1]) for x in line.split(' ')[1:]]
        return values
    
    def create_sample_data(self, data_array):
        """
        Create sample data for clustering.
        
        Args:
            data_array (numpy.array): Input data array
            
        Returns:
            RDD: Spark RDD
        """
        return self.sc.parallelize(data_array)
    
    def train_kmeans_mllib(self, data, k=2, max_iterations=10, seed=50):
        """
        Train K-means using MLlib.
        
        Args:
            data (RDD): Input data RDD
            k (int): Number of clusters
            max_iterations (int): Maximum iterations
            seed (int): Random seed
            
        Returns:
            KMeansModel: Trained model
        """
        model = MLlibKMeans.train(
            data, 
            k, 
            maxIterations=max_iterations,
            initializationMode="random",
            seed=seed,
            initializationSteps=5,
            epsilon=1e-4
        )
        
        return model
    
    def train_kmeans_ml(self, df, k=2, max_iter=20, seed=42):
        """
        Train K-means using ML API.
        
        Args:
            df (DataFrame): Input DataFrame
            k (int): Number of clusters
            max_iter (int): Maximum iterations
            seed (int): Random seed
            
        Returns:
            tuple: (model, predictions)
        """
        # Create K-means
        kmeans = MLKMeans(featuresCol="features", k=k, maxIter=max_iter, seed=seed)
        
        # Train model
        model = kmeans.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        return model, predictions
    
    def predict_cluster(self, model, data_point):
        """
        Predict cluster for a data point.
        
        Args:
            model: Trained K-means model
            data_point (list): Data point to predict
            
        Returns:
            int: Predicted cluster
        """
        if hasattr(model, 'predict'):
            # ML API model
            return model.predict(data_point)
        else:
            # MLlib model
            return model.predict(data_point)
    
    def get_cluster_centers(self, model):
        """
        Get cluster centers from the model.
        
        Args:
            model: Trained K-means model
            
        Returns:
            list: Cluster centers
        """
        if hasattr(model, 'clusterCenters'):
            # MLlib model
            return model.clusterCenters()
        elif hasattr(model, 'clusterCenters'):
            # ML API model
            return model.clusterCenters()
        else:
            return None
    
    def evaluate_clustering(self, model, data):
        """
        Evaluate clustering model using within-set sum of squared errors.
        
        Args:
            model: Trained K-means model
            data (RDD): Input data
            
        Returns:
            float: WSSSE (Within Set Sum of Squared Errors)
        """
        if hasattr(model, 'computeCost'):
            # MLlib model
            wssse = model.computeCost(data)
            print(f"Within Set Sum of Squared Errors = {wssse}")
            return wssse
        else:
            print("WSSSE not available for this model type")
            return None
    
    def load_libsvm_data(self, file_path):
        """
        Load data in LIBSVM format.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            DataFrame: Loaded DataFrame
        """
        return self.spark.read.format("libsvm").load(file_path)
    
    def prepare_clustering_data(self, df, feature_cols):
        """
        Prepare data for clustering by assembling features.
        
        Args:
            df (DataFrame): Input DataFrame
            feature_cols (list): List of feature column names
            
        Returns:
            DataFrame: DataFrame with assembled features
        """
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        return assembler.transform(df)
    
    def analyze_clusters(self, predictions):
        """
        Analyze cluster distribution.
        
        Args:
            predictions (DataFrame): Predictions DataFrame
            
        Returns:
            DataFrame: Cluster distribution
        """
        cluster_dist = predictions.groupBy("prediction").count().orderBy("prediction")
        cluster_dist.show()
        
        return cluster_dist
    
    def monte_carlo_pi_estimation(self, num_samples=1000000):
        """
        Estimate Pi using Monte Carlo method with Spark.
        
        Args:
            num_samples (int): Number of samples
            
        Returns:
            float: Estimated value of Pi
        """
        import random
        
        def is_in_circle(p):
            x = random.random()
            y = random.random()
            return x*x + y*y < 1
        
        samples = self.sc.parallelize(range(0, num_samples))
        in_circle = samples.filter(is_in_circle)
        count = in_circle.count()
        
        pi_estimate = 4.0 * count / num_samples
        print(f"Pi is roughly {pi_estimate}")
        
        return pi_estimate
