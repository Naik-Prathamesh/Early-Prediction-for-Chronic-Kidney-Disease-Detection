# config.py
# Configuration Module for CKD Prediction Project

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'dataset_filename': 'chronic_kidney_disease_full.csv',
    'target_column': 'classification',
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'stratify': True
}

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'eval_metric': 'logloss'
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    },
    'neural_network': {
        'hidden_layer_sizes': (128, 64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'constant',
        'max_iter': 1000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info',
    'title': 'CKD Prediction API',
    'description': 'Early Prediction for Chronic Kidney Disease Detection API',
    'version': '1.0.0'
}

# HIPAA compliance settings
SECURITY_CONFIG = {
    'enable_authentication': False,  # Set to True in production
    'api_key_header': 'X-API-Key',
    'encryption_enabled': False,  # Set to True in production
    'audit_logging': True,
    'data_retention_days': 90,
    'max_request_size': 10 * 1024 * 1024  # 10MB
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'ckd_prediction.log'),
            'mode': 'a',
            'level': 'DEBUG',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'ckd_prediction': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 
        'hemo', 'pcv', 'wc', 'rc'
    ],
    'categorical_features': [
        'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ],
    'risk_thresholds': {
        'age_high_risk': 60,
        'bp_high_risk': 90,
        'creatinine_high_risk': 1.2,
        'glucose_high_risk': 140,
        'hemoglobin_low_risk': 10
    },
    'feature_scaling': True,
    'create_composite_features': True
}

# Clinical risk assessment
CLINICAL_CONFIG = {
    'risk_levels': {
        'low': {'min': 0.0, 'max': 0.3, 'color': 'green'},
        'moderate': {'min': 0.3, 'max': 0.7, 'color': 'yellow'},
        'high': {'min': 0.7, 'max': 1.0, 'color': 'red'}
    },
    'recommendations': {
        'low_risk': [
            "Continue preventive care",
            "Annual health screening",
            "Maintain healthy lifestyle"
        ],
        'moderate_risk': [
            "Follow-up with primary care physician",
            "Lifestyle modifications",
            "Monitor kidney function every 6 months"
        ],
        'high_risk': [
            "Immediate medical consultation recommended",
            "Consider specialist referral",
            "Monitor kidney function every 3 months"
        ]
    }
}

# Performance monitoring
MONITORING_CONFIG = {
    'model_performance_threshold': 0.85,
    'data_drift_threshold': 0.05,
    'prediction_latency_threshold_ms': 1000,
    'api_rate_limit_per_minute': 100,
    'health_check_interval_seconds': 300
}

# Environment-specific settings
def get_environment():
    """Get current environment (development, staging, production)"""
    return os.getenv('ENVIRONMENT', 'development').lower()

def is_production():
    """Check if running in production environment"""
    return get_environment() == 'production'

def is_development():
    """Check if running in development environment"""
    return get_environment() == 'development'

# Environment-specific configurations
if is_production():
    SECURITY_CONFIG['enable_authentication'] = True
    SECURITY_CONFIG['encryption_enabled'] = True
    API_CONFIG['reload'] = False
    API_CONFIG['log_level'] = 'warning'
    
elif is_development():
    SECURITY_CONFIG['enable_authentication'] = False
    API_CONFIG['reload'] = True
    API_CONFIG['log_level'] = 'debug'

# Database configuration (if needed in the future)
DATABASE_CONFIG = {
    'url': os.getenv('DATABASE_URL', 'sqlite:///./ckd_predictions.db'),
    'echo': is_development(),
    'pool_size': 5,
    'max_overflow': 10
}

# Export all configurations
__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'MODEL_DIR', 'RESULTS_DIR', 'LOGS_DIR',
    'DATA_CONFIG', 'MODEL_CONFIG', 'API_CONFIG', 'SECURITY_CONFIG',
    'LOGGING_CONFIG', 'FEATURE_CONFIG', 'CLINICAL_CONFIG', 'MONITORING_CONFIG',
    'DATABASE_CONFIG', 'get_environment', 'is_production', 'is_development'
]