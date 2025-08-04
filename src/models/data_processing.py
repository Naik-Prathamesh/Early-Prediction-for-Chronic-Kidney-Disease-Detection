import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class CKDDataProcessor:
    """
    Comprehensive data preprocessing class for Chronic Kidney Disease prediction.
    Handles data loading, cleaning, encoding, and preparation for machine learning models.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='mean')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.feature_names = []
        
    def load_data(self, file_path=None):
        """
        Load CKD dataset. If no file path provided, creates sample data for demonstration.
        """
        if file_path and os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                return data
            except Exception as e:
                print(f"Error loading data: {e}")
                return self._create_sample_data()
        else:
            print("Creating sample CKD data for demonstration...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create a comprehensive sample CKD dataset with realistic medical parameters.
        """
        np.random.seed(42)
        n_samples = 400
        
        # Create realistic medical data
        data = pd.DataFrame({
            'age': np.random.randint(20, 80, n_samples),
            'bp': np.random.randint(80, 180, n_samples),  # blood pressure
            'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_samples),  # specific gravity
            'al': np.random.randint(0, 6, n_samples),  # albumin
            'su': np.random.randint(0, 6, n_samples),  # sugar
            'rbc': np.random.choice(['normal', 'abnormal'], n_samples),  # red blood cells
            'pc': np.random.choice(['normal', 'abnormal'], n_samples),  # pus cell
            'pcc': np.random.choice(['present', 'notpresent'], n_samples),  # pus cell clumps
            'ba': np.random.choice(['present', 'notpresent'], n_samples),  # bacteria
            'bgr': np.random.randint(70, 300, n_samples),  # blood glucose random
            'bu': np.random.randint(15, 150, n_samples),  # blood urea
            'sc': np.random.uniform(0.5, 10.0, n_samples),  # serum creatinine
            'sod': np.random.randint(120, 160, n_samples),  # sodium
            'pot': np.random.uniform(2.5, 6.0, n_samples),  # potassium
            'hemo': np.random.uniform(8.0, 18.0, n_samples),  # hemoglobin
            'pcv': np.random.randint(25, 55, n_samples),  # packed cell volume
            'wc': np.random.randint(4000, 15000, n_samples),  # white blood cell count
            'rc': np.random.uniform(3.5, 6.0, n_samples),  # red blood cell count
            'htn': np.random.choice(['yes', 'no'], n_samples),  # hypertension
            'dm': np.random.choice(['yes', 'no'], n_samples),  # diabetes mellitus
            'cad': np.random.choice(['yes', 'no'], n_samples),  # coronary artery disease
            'appet': np.random.choice(['good', 'poor'], n_samples),  # appetite
            'pe': np.random.choice(['yes', 'no'], n_samples),  # pedal edema
            'ane': np.random.choice(['yes', 'no'], n_samples),  # anemia
        })
        
        # Create target variable with realistic correlation to features
        # Higher risk factors increase CKD probability
        risk_score = (
            (data['age'] > 60).astype(int) * 0.2 +
            (data['bp'] > 140).astype(int) * 0.2 +
            (data['sc'] > 1.5).astype(int) * 0.3 +
            (data['hemo'] < 10).astype(int) * 0.2 +
            (data['htn'] == 'yes').astype(int) * 0.15 +
            (data['dm'] == 'yes').astype(int) * 0.15 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        data['class'] = (risk_score > 0.5).astype(int)
        data['class'] = data['class'].map({0: 'notckd', 1: 'ckd'})
        
        # Introduce some missing values to make it realistic
        missing_cols = ['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in missing_cols:
            missing_indices = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
            data.loc[missing_indices, col] = np.nan
            
        return data
    
    def preprocess_data(self, data, target_column='class'):
        """
        Comprehensive preprocessing pipeline for CKD data.
        """
        print("Starting data preprocessing...")
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Handle missing values
        if numeric_features:
            X[numeric_features] = self.imputer_numeric.fit_transform(X[numeric_features])
        
        if categorical_features:
            X[categorical_features] = self.imputer_categorical.fit_transform(X[categorical_features])
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable if it exists
        if y is not None:
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
        
        # Scale numeric features
        if numeric_features:
            X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        print("Data preprocessing completed successfully!")
        return X, y
    
    def transform_new_data(self, data):
        """
        Transform new data using fitted preprocessors.
        """
        X = data.copy()
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values
        if numeric_features:
            X[numeric_features] = self.imputer_numeric.transform(X[numeric_features])
        
        if categorical_features:
            X[categorical_features] = self.imputer_categorical.transform(X[categorical_features])
        
        # Encode categorical variables using fitted encoders
        for col in categorical_features:
            if col in self.label_encoders:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                known_classes = self.label_encoders[col].classes_
                X[col] = X[col].apply(lambda x: x if x in known_classes else known_classes[0])
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numeric features
        if numeric_features:
            X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        return X
    
    def get_feature_importance_names(self):
        """
        Return feature names for importance analysis.
        """
        return self.feature_names
    
    def inverse_transform_target(self, y_encoded):
        """
        Convert encoded target back to original labels.
        """
        if 'target' in self.label_encoders:
            return self.label_encoders['target'].inverse_transform(y_encoded)
        return y_encoded

import os