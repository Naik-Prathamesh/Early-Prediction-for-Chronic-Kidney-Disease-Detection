# main.py
# Main Training Script for CKD Prediction Project
# Run this script to execute the complete ML pipeline

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
from src.models.data_processing import CKDDataProcessor
from model_training import (CKDModelTrainer, RandomForestModel, XGBoostModel, 
                           SVMModel, NeuralNetworkModel)
from utils import (create_directories, save_json, create_model_comparison_report,
                  generate_patient_report, setup_plotting_style)
from config import (DATA_CONFIG, MODEL_CONFIG, RESULTS_DIR, MODEL_DIR, 
                   LOGS_DIR, is_development)

# Configure logging
def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(LOGS_DIR / "ckd_training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CKD Prediction Model Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/chronic_kidney_disease_full.csv",
        help="Path to the CKD dataset CSV file"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "xgb", "svm", "nn", "all"],
        default=["all"],
        help="Models to train (rf: Random Forest, xgb: XGBoost, svm: SVM, nn: Neural Network)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for testing"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=True,
        help="Save trained models to disk"
    )
    
    parser.add_argument(
        "--create-plots",
        action="store_true",
        default=True,
        help="Create and save visualization plots"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        default=False,
        help="Skip data preprocessing (use if data is already processed)"
    )
    
    return parser.parse_args()

def initialize_models(args):
    """Initialize models based on command line arguments"""
    models = {}
    
    model_configs = MODEL_CONFIG
    
    if "all" in args.models:
        selected_models = ["rf", "xgb", "svm", "nn"]
    else:
        selected_models = args.models
    
    if "rf" in selected_models:
        models["Random Forest"] = RandomForestModel(**model_configs['random_forest'])
    
    if "xgb" in selected_models:
        models["XGBoost"] = XGBoostModel(**model_configs['xgboost'])
    
    if "svm" in selected_models:
        models["SVM"] = SVMModel(**model_configs['svm'])
    
    if "nn" in selected_models:
        models["Neural Network"] = NeuralNetworkModel(**model_configs['neural_network'])
    
    return models

def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting CKD Prediction Model Training Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create directories
        create_directories()
        setup_plotting_style()
        
        # Check if data file exists
        if not os.path.exists(args.data_path):
            logger.error(f"Data file not found: {args.data_path}")
            logger.info("Please ensure the dataset file exists at the specified path.")
            logger.info("You can download the CKD dataset from: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease")
            sys.exit(1)
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = CKDDataProcessor()
        
        # Load data
        logger.info(f"Loading data from: {args.data_path}")
        df = data_processor.load_data(args.data_path)
        
        if df is None:
            logger.error("Failed to load dataset")
            sys.exit(1)
        
        # Data exploration
        logger.info("Performing data exploration...")
        data_processor.explore_data(df)
        
        # Prepare data for ML
        if not args.skip_preprocessing:
            logger.info("Preprocessing data for machine learning...")
            X_train, X_test, y_train, y_test = data_processor.prepare_data_for_ml(
                df, 
                target_column=DATA_CONFIG['target_column'],
                test_size=args.test_size,
                random_state=args.random_state
            )
        else:
            logger.info("Skipping preprocessing as requested...")
            # Assume data is already processed
            # This branch would need implementation based on your processed data format
            pass
        
        feature_names = data_processor.get_feature_importance_names()
        logger.info(f"Training with {len(feature_names)} features")
        
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = CKDModelTrainer()
        
        # Initialize and add models
        logger.info("Initializing models...")
        models = initialize_models(args)
        
        for model_name, model in models.items():
            trainer.add_model(model_name, model)
            logger.info(f"Added model: {model_name}")
        
        # Train all models
        logger.info("Starting model training...")
        start_time = datetime.now()
        
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Compare models
        logger.info("Comparing model performance...")
        comparison_df = trainer.compare_models()
        
        # Save comparison report
        comparison_report_path = RESULTS_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_report_path, index=False)
        logger.info(f"Model comparison saved to: {comparison_report_path}")
        
        # Create detailed report
        detailed_report = create_model_comparison_report(trainer.results)
        detailed_report_path = RESULTS_DIR / "detailed_model_comparison.csv"
        detailed_report.to_csv(detailed_report_path, index=False)
        logger.info(f"Detailed comparison saved to: {detailed_report_path}")
        
        # Create visualizations
        if args.create_plots:
            logger.info("Creating visualizations...")
            
            try:
                # ROC curves
                trainer.plot_roc_curves(y_test)
                
                # Feature importance (for tree-based models)
                trainer.plot_feature_importance(feature_names)
                
                logger.info("Visualizations created and saved")
            
            except Exception as e:
                logger.warning(f"Error creating visualizations: {e}")
        
        # Save models
        if args.save_models:
            logger.info("Saving trained models...")
            trainer.save_all_models()
            
            # Save data processor as well
            import joblib
            processor_path = MODEL_DIR / "data_processor.joblib"
            processor_data = {
                'label_encoders': data_processor.label_encoders,
                'scaler': data_processor.scaler,
                'feature_names': data_processor.feature_names,
                'imputer': data_processor.imputer
            }
            joblib.dump(processor_data, processor_path)
            logger.info(f"Data processor saved to: {processor_path}")
        
        # Generate summary report
        logger.info("Generating summary report...")
        
        best_model_row = detailed_report.iloc[0]
        best_model_name = best_model_row['Model']
        best_f1_score = best_model_row['F1-Score']
        
        summary_report = {
            'pipeline_completion_time': datetime.now().isoformat(),
            'dataset_path': args.data_path,
            'dataset_shape': df.shape,
            'training_set_size': X_train.shape[0],
            'test_set_size': X_test.shape[0],
            'n_features': len(feature_names),
            'models_trained': list(models.keys()),
            'best_model': best_model_name,
            'best_f1_score': float(best_f1_score),
            'total_training_time_seconds': training_time,
            'arguments_used': vars(args)
        }
        
        summary_path = RESULTS_DIR / "training_summary.json"
        save_json(summary_report, summary_path)
        
        # Print summary
        print("\n" + "="*60)
        print("CKD PREDICTION MODEL TRAINING COMPLETED")
        print("="*60)
        print(f"Dataset: {args.data_path}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {len(feature_names)}")
        print(f"Models trained: {', '.join(models.keys())}")
        print(f"Best model: {best_model_name} (F1-Score: {best_f1_score:.4f})")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Results saved to: {RESULTS_DIR}")
        if args.save_models:
            print(f"Models saved to: {MODEL_DIR}")
        print("="*60)
        
        logger.info("Training pipeline completed successfully!")
        
        # API server startup hint
        if is_development():
            print("\nTo start the API server, run:")
            print("cd src && python api.py")
            print("or")
            print("uvicorn src.api:app --reload")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

def demo_prediction():
    """Demonstrate prediction with sample data"""
    print("\n" + "="*50)
    print("DEMO: CKD PREDICTION EXAMPLE")
    print("="*50)
    
    # Sample patient data (you can modify these values)
    sample_patient = {
        'age': 65,
        'bp': 95,
        'sg': 1.015,
        'al': 2,
        'su': 1,
        'rbc': 'abnormal',
        'pc': 'abnormal',
        'pcc': 'present',
        'ba': 'notpresent',
        'bgr': 180,
        'bu': 45,
        'sc': 2.1,
        'sod': 135,
        'pot': 4.2,
        'hemo': 9.5,
        'pcv': 32,
        'wc': 9500,
        'rc': 4.2,
        'htn': 'yes',
        'dm': 'yes',
        'cad': 'no',
        'appet': 'poor',
        'pe': 'yes',
        'ane': 'yes'
    }
    
    print("Sample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    print("\nNote: To get actual predictions, start the API server and send a POST request to /predict")
    print("Example API usage:")
    print("curl -X POST 'http://localhost:8000/predict' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"age\": 65, \"bp\": 95, \"sc\": 2.1, ...}'")

if __name__ == "__main__":
    # Check if user wants to run demo
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_prediction()
    else:
        main()