#!/usr/bin/env python3
"""
Main script for Spark ML project demonstration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import setup_spark_session, load_taxi_data, create_datetime_features, create_time_features
from regression import TaxiFarePredictor
from classification import ClassificationModels
from clustering import ClusteringModels
from topic_modeling import TopicModeling


def main():
    """
    Main function to demonstrate the Spark ML project.
    """
    print("Starting Spark ML Project Demonstration")
    print("=" * 50)
    
    # Initialize Spark session
    print("Initializing Spark session...")
    spark = setup_spark_session("SparkMLDemo")
    
    try:
        # Demonstrate data preprocessing
        print("\nData Preprocessing Demo")
        print("-" * 30)
        
        # Note: You would need to provide actual data file paths
        print("To run with actual data, update the file paths in the code.")
        print("Example data files should be in data/raw/ directory")
        
        # Demonstrate regression
        print("\nRegression Analysis Demo")
        print("-" * 30)
        
        predictor = TaxiFarePredictor(spark)
        print("TaxiFarePredictor initialized")
        print("   - Model M1: Distance-based fare prediction")
        print("   - Model M2: Distance + duration-based prediction")
        
        # Demonstrate classification
        print("\nClassification Demo")
        print("-" * 30)
        
        classifier = ClassificationModels(spark)
        print("ClassificationModels initialized")
        print("   - Logistic Regression (MLlib & ML)")
        print("   - Support Vector Machine")
        
        # Demonstrate clustering
        print("\nClustering Demo")
        print("-" * 30)
        
        clusterer = ClusteringModels(spark)
        print("ClusteringModels initialized")
        print("   - K-means clustering")
        print("   - Monte Carlo Pi estimation")
        
        # Demonstrate topic modeling
        print("\nTopic Modeling Demo")
        print("-" * 30)
        
        topic_modeler = TopicModeling(spark)
        print("TopicModeling initialized")
        print("   - Latent Dirichlet Allocation (LDA)")
        print("   - Topic analysis and visualization")
        
        print("\nProject structure is ready!")
        print("\nProject Structure:")
        print("   ├── data/raw/          # Raw data files")
        print("   ├── data/processed/      # Processed data")
        print("   ├── notebooks/           # Jupyter notebooks")
        print("   ├── src/                 # Python modules")
        print("   ├── docs/               # Documentation")
        print("   ├── results/            # Output files")
        print("   ├── README.md           # Project documentation")
        print("   ├── requirements.txt    # Dependencies")
        print("   └── .gitignore          # Git ignore rules")
        
        print("\nNext Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run Jupyter notebook: jupyter notebook notebooks/Spark_ML.ipynb")
        print("   3. Execute individual modules: python src/regression.py")
        print("   4. Check documentation: docs/setup_guide.md")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This is expected if data files are not available.")
        print("The project structure is ready for your data!")
    
    finally:
        # Clean up
        spark.stop()
        print("\nSpark session stopped")


if __name__ == "__main__":
    main()
