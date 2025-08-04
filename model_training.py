import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our data processor
from src.models.data_processing import CKDDataProcessor

class BaseModel:
    """
    Base class for CKD prediction models with comprehensive training and evaluation capabilities.
    """
    
    def __init__(self, model_name="CKD_Predictor"):
        self.model_name = model_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.data_processor = CKDDataProcessor()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        
    def initialize_models(self):
        """
        Initialize multiple ML models for comparison.
        """
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                random_state=42,
                kernel='rbf',
                C=1.0,
                probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Naive Bayes': GaussianNB()
        }
        print(f"Initialized {len(self.models)} models for training")
    
    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess the CKD dataset.
        """
        print("Loading and preprocessing CKD data...")
        
        # Load data
        raw_data = self.data_processor.load_data(data_path)
        print(f"Loaded dataset with shape: {raw_data.shape}")
        
        # Preprocess data
        X, y = self.data_processor.preprocess_data(raw_data)
        self.feature_names = self.data_processor.get_feature_importance_names()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution - Training: {np.bincount(self.y_train)}")
        print(f"Class distribution - Test: {np.bincount(self.y_test)}")
        
        return X, y
    
    def train_models(self):
        """
        Train all initialized models and evaluate their performance.
        """
        if not self.models:
            self.initialize_models()
        
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        results = {}
        print("\nTraining and evaluating models...")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate scores
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # AUC score (if model supports probability prediction)
            try:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
            except:
                auc_score = 0.0
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'auc_score': auc_score,
                'predictions': y_pred_test
            }
            
            print(f"{name} Results:")
            print(f"  Train Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            print(f"  AUC Score: {auc_score:.4f}")
            
            # Update best model
            if test_accuracy > self.best_score:
                self.best_score = test_accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*50)
        print(f"Best Model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for the specified model.
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            base_model = SVC(random_state=42, probability=True)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model if this one is better
        test_score = grid_search.best_estimator_.score(self.X_test, self.y_test)
        if test_score > self.best_score:
            self.best_model = grid_search.best_estimator_
            self.best_score = test_score
            self.best_model_name = f"{model_name} (Tuned)"
            print(f"New best model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model=None, model_name="Best Model"):
        """
        Comprehensive model evaluation with detailed metrics.
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        if model is None:
            raise ValueError("No model to evaluate. Train models first.")
        
        print(f"\nDetailed Evaluation for {model_name}")
        print("="*50)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        except:
            pass
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # AUC Score
        if y_pred_proba is not None:
            auc = roc_auc_score(self.y_test, y_pred_proba)
            print(f"AUC Score: {auc:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        target_names = ['Not CKD', 'CKD']
        if 'target' in self.data_processor.label_encoders:
            target_names = self.data_processor.label_encoders['target'].classes_
        
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc if y_pred_proba is not None else None,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, model=None, top_n=10):
        """
        Get and display feature importance for tree-based models.
        """
        if model is None:
            model = self.best_model
        
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance")
            return None
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = self.feature_names[:len(importance)]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def save_model(self, model=None, filename=None):
        """
        Save the trained model and preprocessor to disk.
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/ckd_model_{timestamp}"
        
        # Save model
        model_file = f"{filename}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessor
        preprocessor_file = f"{filename}_preprocessor.pkl"
        with open(preprocessor_file, 'wb') as f:
            pickle.dump(self.data_processor, f)
        
        print(f"Model saved to: {model_file}")
        print(f"Preprocessor saved to: {preprocessor_file}")
        
        return model_file, preprocessor_file
    
    def load_model(self, model_file, preprocessor_file):
        """
        Load a saved model and preprocessor.
        """
        # Load model
        with open(model_file, 'rb') as f:
            self.best_model = pickle.load(f)
        
        # Load preprocessor
        with open(preprocessor_file, 'rb') as f:
            self.data_processor = pickle.load(f)
        
        print(f"Model loaded from: {model_file}")
        print(f"Preprocessor loaded from: {preprocessor_file}")
    
    def predict_single(self, input_data):
        """
        Make prediction for a single patient.
        input_data: dict with feature names as keys
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        processed_input = self.data_processor.transform_new_data(input_df)
        
        # Predict
        prediction = self.best_model.predict(processed_input)[0]
        
        # Get probability if available
        try:
            probability = self.best_model.predict_proba(processed_input)[0]
            prob_dict = {
                'Not CKD': probability[0],
                'CKD': probability[1]
            }
        except:
            prob_dict = None
        
        # Convert prediction back to original label
        if 'target' in self.data_processor.label_encoders:
            prediction_label = self.data_processor.label_encoders['target'].inverse_transform([prediction])[0]
        else:
            prediction_label = 'CKD' if prediction == 1 else 'Not CKD'
        
        return {
            'prediction': prediction_label,
            'prediction_code': int(prediction),
            'probabilities': prob_dict
        }

# Quick test function
def test_model_training():
    """
    Test the model training pipeline.
    """
    print("Testing CKD Model Training Pipeline...")
    
    # Initialize model
    ckd_model = BaseModel("CKD_Test_Model")
    
    # Load and preprocess data
    ckd_model.load_and_preprocess_data()
    
    # Train models
    results = ckd_model.train_models()
    
    # Evaluate best model
    evaluation = ckd_model.evaluate_model()
    
    # Get feature importance
    importance = ckd_model.get_feature_importance()
    
    # Save model
    model_files = ckd_model.save_model()
    
    print("\n✅ Model training pipeline test completed successfully!")
    
    return ckd_model

if __name__ == "__main__":
    # Run test
    model = test_model_training()