"""
Classification module for Spark ML project.
"""

from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession


class ClassificationModels:
    """
    Classification models using both MLlib and ML APIs.
    """
    
    def __init__(self, spark_session):
        """
        Initialize the classification models.
        
        Args:
            spark_session (SparkSession): Spark session
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        
    def parse_point_logreg(self, line):
        """
        Parse line for logistic regression.
        
        Args:
            line (str): Input line
            
        Returns:
            LabeledPoint: Parsed labeled point
        """
        values = [float(x) for x in line.split(' ')]
        return LabeledPoint(values[0], values[1:])
    
    def parse_point_svm(self, line):
        """
        Parse line for SVM.
        
        Args:
            line (str): Input line
            
        Returns:
            LabeledPoint: Parsed labeled point
        """
        values = [float(x) for x in line.split(' ')]
        return LabeledPoint(values[0], values[1:])
    
    def train_logistic_regression_mllib(self, data_path):
        """
        Train logistic regression using MLlib.
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            LogisticRegressionModel: Trained model
        """
        # Load and parse data
        data = self.sc.textFile(data_path)
        parsed_data = data.map(self.parse_point_logreg)
        
        # Train model
        model = LogisticRegressionWithLBFGS.train(parsed_data)
        
        return model
    
    def train_svm_mllib(self, data_path):
        """
        Train SVM using MLlib.
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            SVMModel: Trained model
        """
        # Load and parse data
        data = self.sc.textFile(data_path)
        parsed_data = data.map(self.parse_point_svm)
        
        # Train model
        model = SVMWithSGD.train(parsed_data)
        
        return model
    
    def evaluate_logistic_regression(self, model, data_path):
        """
        Evaluate logistic regression model.
        
        Args:
            model: Trained model
            data_path (str): Path to the test data
            
        Returns:
            dict: Evaluation metrics
        """
        # Load and parse test data
        data = self.sc.textFile(data_path)
        parsed_data = data.map(self.parse_point_logreg)
        
        # Make predictions
        labels_and_preds = parsed_data.map(lambda x: (x.label, model.predict(x.features)))
        
        # Calculate training error
        train_err = labels_and_preds.filter(lambda k: k[0] != k[1]).count() / float(parsed_data.count())
        
        print(f"Training Error: {train_err:.4f}")
        
        return {"training_error": train_err}
    
    def train_logistic_regression_ml(self, df):
        """
        Train logistic regression using ML API.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (model, predictions)
        """
        # Create logistic regression
        lr = LogisticRegression(featuresCol='features', labelCol='label')
        
        # Train model
        model = lr.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        return model, predictions
    
    def train_svm_ml(self, df):
        """
        Train SVM using ML API.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (model, predictions)
        """
        # Create SVM
        svm = LinearSVC(featuresCol='features', labelCol='label')
        
        # Train model
        model = svm.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        return model, predictions
    
    def evaluate_binary_classification(self, predictions):
        """
        Evaluate binary classification model.
        
        Args:
            predictions (DataFrame): Predictions DataFrame
            
        Returns:
            dict: Evaluation metrics
        """
        # Binary classification evaluator
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        auc = evaluator.evaluate(predictions)
        
        # Multiclass classification evaluator
        multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
        
        metrics = {
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def predict_sample(self, model, sample_data):
        """
        Make prediction on sample data.
        
        Args:
            model: Trained model
            sample_data (list): Sample feature vector
            
        Returns:
            float: Prediction
        """
        return model.predict(sample_data)
    
    def get_feature_importance(self, model):
        """
        Get feature importance from logistic regression model.
        
        Args:
            model: Trained logistic regression model
            
        Returns:
            list: Feature coefficients
        """
        if hasattr(model, 'coefficients'):
            return model.coefficients.toArray().tolist()
        else:
            return None
