"""
Train all machine learning models and save them to the models directory.
"""
import os
import sys
import joblib

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor
from src.features.feature_engineer import FeatureEngineer
from src.models.energy_star_predictor import EnergyStarScorePredictor
from src.models.building_clustering import BuildingClusteringModel
from src.models.high_accuracy_predictor import HighAccuracyEnergyStarPredictor
from src.models.recommendation_engine import BuildingRecommendationEngine
from src.models.energy_efficiency_classifier import EnergyEfficiencyClassifier

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# Visualization directory
visualization_dir = os.path.join(os.path.dirname(__file__), '..', 'visualization')
os.makedirs(visualization_dir, exist_ok=True)

# Data paths
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
energy_file = os.path.join(data_dir, 'raw', 'Chicago_Energy_Benchmarking_20250403.csv')
buildings_file = os.path.join(data_dir, 'raw', 'Chicago_Energy_Benchmarking_-_Covered_Buildings_20250403.csv')

def main():
    print("Starting model training pipeline...")
    
    # Load and process data
    print("Loading and processing data...")
    data_processor = DataProcessor()
    energy_df, buildings_df, merged_df, latest_year = data_processor.process_data(
        energy_file, buildings_file
    )
    
    # Apply feature engineering
    print("Applying feature engineering...")
    feature_engineer = FeatureEngineer(merged_df)
    merged_df = feature_engineer.engineer_all_features()
    
    # Save processed data
    processed_data_path = os.path.join(data_dir, 'processed', 'merged_data.csv')
    merged_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # Train ENERGY STAR Score Predictor
    print("\nTraining ENERGY STAR Score Predictor...")
    energy_star_predictor = EnergyStarScorePredictor(merged_df)
    energy_star_predictor.prepare_data()
    energy_star_predictor.create_and_train_model()
    metrics, predictions = energy_star_predictor.evaluate_model()
    feature_importance = energy_star_predictor.get_feature_importance()
    energy_star_predictor.plot_prediction_performance(predictions)
    
    # Save the model
    model_path = os.path.join(models_dir, 'energy_star_predictor.joblib')
    joblib.dump(energy_star_predictor.model, model_path)
    print(f"ENERGY STAR Score Predictor saved to {model_path}")
    
    # Train Building Clustering Model
    print("\nTraining Building Clustering Model...")
    clustering_model = BuildingClusteringModel(merged_df)
    cluster_data, cluster_centers_df, cluster_analysis = clustering_model.run_full_clustering_pipeline()
    
    # Save the model
    model_path = os.path.join(models_dir, 'building_clustering.joblib')
    joblib.dump(clustering_model.kmeans, model_path)
    print(f"Building Clustering Model saved to {model_path}")
    
    # Train High Accuracy Predictor
    print("\nTraining High Accuracy ENERGY STAR Predictor...")
    high_accuracy_predictor = HighAccuracyEnergyStarPredictor()
    model, metrics = high_accuracy_predictor.train_and_evaluate(energy_df, buildings_df)
    
    # Save the model
    model_path = os.path.join(models_dir, 'high_accuracy_predictor.joblib')
    joblib.dump(model, model_path)
    print(f"High Accuracy Predictor saved to {model_path}")
    
    # Train Recommendation Engine
    print("\nTraining Building Recommendation Engine...")
    recommendation_engine = BuildingRecommendationEngine()
    recommendation_engine.load_data(energy_file, buildings_file)
    recommendation_engine.prepare_data()
    recommendation_engine.train_decision_trees()
    
    # Save decision tree models
    for system, model_info in recommendation_engine.dt_models.items():
        model_path = os.path.join(models_dir, f'recommendation_engine_{system}.joblib')
        joblib.dump(model_info['model'], model_path)
        print(f"Recommendation Engine - {system} model saved to {model_path}")
        
        # Visualize and save decision tree
        recommendation_engine.visualize_decision_tree(system)
    
    # Train Energy Efficiency Classifier
    print("\nTraining Energy Efficiency Classifier...")
    classification_model = EnergyEfficiencyClassifier(merged_df)
    model, conf_matrix, class_report, feature_importance = classification_model.run_full_classification_pipeline()
    
    # Save the model
    model_path = os.path.join(models_dir, 'energy_efficiency_classifier.joblib')
    joblib.dump(model, model_path)
    print(f"Energy Efficiency Classifier saved to {model_path}")
    
    print("\nAll models trained and saved successfully.")
    print(f"Model files saved to {models_dir}")
    print(f"Visualization files saved to {visualization_dir}")

if __name__ == "__main__":
    main()