"""
Topic modeling module for Spark ML project.
"""

from pyspark.ml.clustering import LDA
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


class TopicModeling:
    """
    Topic modeling using Latent Dirichlet Allocation (LDA).
    """
    
    def __init__(self, spark_session):
        """
        Initialize the topic modeling.
        
        Args:
            spark_session (SparkSession): Spark session
        """
        self.spark = spark_session
        
    def load_libsvm_data(self, file_path):
        """
        Load data in LIBSVM format for topic modeling.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            DataFrame: Loaded DataFrame
        """
        return self.spark.read.format("libsvm").load(file_path)
    
    def train_lda(self, df, k=10, max_iter=10, seed=42):
        """
        Train LDA model.
        
        Args:
            df (DataFrame): Input DataFrame
            k (int): Number of topics
            max_iter (int): Maximum iterations
            seed (int): Random seed
            
        Returns:
            tuple: (model, transformed_data)
        """
        # Create LDA
        lda = LDA(k=k, maxIter=max_iter, seed=seed)
        
        # Train model
        model = lda.fit(df)
        
        # Transform data
        transformed = model.transform(df)
        
        return model, transformed
    
    def describe_topics(self, model, max_terms_per_topic=3):
        """
        Describe topics discovered by the model.
        
        Args:
            model: Trained LDA model
            max_terms_per_topic (int): Maximum terms per topic
            
        Returns:
            DataFrame: Topics description
        """
        topics = model.describeTopics(max_terms_per_topic)
        topics.show()
        
        return topics
    
    def get_topic_distributions(self, transformed_df):
        """
        Get topic distributions for documents.
        
        Args:
            transformed_df (DataFrame): Transformed DataFrame
            
        Returns:
            DataFrame: Topic distributions
        """
        # Show topic distributions
        transformed_df.select("topicDistribution").show(truncate=False)
        
        return transformed_df.select("topicDistribution")
    
    def analyze_topic_weights(self, topics_df):
        """
        Analyze topic weights.
        
        Args:
            topics_df (DataFrame): Topics DataFrame
            
        Returns:
            list: Topic weights
        """
        topic_weights = topics_df.select("termWeights").collect()
        
        print("Topic weights analysis:")
        for i, row in enumerate(topic_weights):
            print(f"Topic {i}: {row[0]}")
        
        return topic_weights
    
    def get_document_topic_distribution(self, transformed_df, doc_id=0):
        """
        Get topic distribution for a specific document.
        
        Args:
            transformed_df (DataFrame): Transformed DataFrame
            doc_id (int): Document ID
            
        Returns:
            list: Topic distribution for the document
        """
        doc_topics = transformed_df.select("topicDistribution").collect()[doc_id][0]
        
        print(f"Document {doc_id} topic distribution: {doc_topics}")
        
        return doc_topics
    
    def find_dominant_topics(self, transformed_df, threshold=0.1):
        """
        Find documents with dominant topics above threshold.
        
        Args:
            transformed_df (DataFrame): Transformed DataFrame
            threshold (float): Topic probability threshold
            
        Returns:
            DataFrame: Documents with dominant topics
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import BooleanType
        
        def has_dominant_topic(topic_dist):
            return max(topic_dist) > threshold
        
        has_dominant_udf = udf(has_dominant_topic, BooleanType())
        
        dominant_docs = transformed_df.filter(has_dominant_udf("topicDistribution"))
        
        print(f"Documents with dominant topics (>{threshold}): {dominant_docs.count()}")
        
        return dominant_docs
    
    def evaluate_model_perplexity(self, model, df):
        """
        Evaluate model using perplexity.
        
        Args:
            model: Trained LDA model
            df (DataFrame): Input DataFrame
            
        Returns:
            float: Model perplexity
        """
        perplexity = model.logPerplexity(df)
        
        print(f"Model perplexity: {perplexity}")
        
        return perplexity
    
    def get_vocabulary_size(self, df):
        """
        Get vocabulary size from the dataset.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            int: Vocabulary size
        """
        # Get the size of the feature vector
        sample_features = df.select("features").first()[0]
        vocab_size = sample_features.size
        
        print(f"Vocabulary size: {vocab_size}")
        
        return vocab_size
    
    def compare_models(self, df, k_values=[5, 10, 15, 20], max_iter=10):
        """
        Compare LDA models with different numbers of topics.
        
        Args:
            df (DataFrame): Input DataFrame
            k_values (list): List of k values to test
            max_iter (int): Maximum iterations
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for k in k_values:
            print(f"\nTraining LDA with k={k}")
            
            # Train model
            lda = LDA(k=k, maxIter=max_iter)
            model = lda.fit(df)
            
            # Evaluate
            perplexity = model.logPerplexity(df)
            results[k] = perplexity
            
            print(f"Perplexity for k={k}: {perplexity}")
        
        return results
