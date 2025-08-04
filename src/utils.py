# utils.py
# Utility Functions for CKD Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import RESULTS_DIR, MODEL_DIR, LOGS_DIR, CLINICAL_CONFIG

def setup_plotting_style():
    """Set up consistent plotting style for the project"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def create_directories():
    """Create necessary project directories"""
    directories = [RESULTS_DIR, MODEL_DIR, LOGS_DIR, 
                  RESULTS_DIR / 'plots', RESULTS_DIR / 'reports']
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("Project directories created successfully")

def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def load_json(filepath: str) -> Dict:
    """Load JSON file as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_detailed_metrics(y_true, y_pred, y_pred_proba=None) -> Dict:
    """
    Calculate comprehensive evaluation metrics for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, average_precision_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'support': len(y_true)
    }
    
    # Calculate specificity manually
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Add AUC metrics if probabilities are provided
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: Optional[str] = None):
    """
    Plot confusion matrix with detailed statistics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CKD', 'CKD'],
                yticklabels=['No CKD', 'CKD'])
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    
    # Add statistics text
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        stats_text = f'TN: {tn}, FP: {fp}\nFN: {fn}, TP: {tp}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve_single(y_true, y_pred_proba, model_name: str, save_path: Optional[str] = None):
    """
    Plot ROC curve for a single model
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance_comparison(importance_dfs: Dict[str, pd.DataFrame], 
                                     top_n: int = 15, save_path: Optional[str] = None):
    """
    Plot feature importance comparison across multiple models
    
    Args:
        importance_dfs: Dictionary of {model_name: importance_dataframe}
        top_n: Number of top features to show
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    n_models = len(importance_dfs)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, importance_df) in enumerate(importance_dfs.items()):
        top_features = importance_df.head(top_n)
        
        ax = axes[idx]
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name}')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
    
    plt.suptitle(f'Top {top_n} Feature Importance Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_curves(model, X, y, model_name: str, save_path: Optional[str] = None):
    """
    Plot learning curves to analyze model performance vs training size
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """
    Generate classification report as DataFrame
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as DataFrame
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def create_model_comparison_report(results: Dict) -> pd.DataFrame:
    """
    Create comprehensive model comparison report
    
    Args:
        results: Dictionary with model results
        
    Returns:
        Comparison report as DataFrame
    """
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        training_time = result['training_history'].get('training_time', 0)
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall/Sensitivity': round(metrics['recall'], 4),
            'Specificity': round(metrics['specificity'], 4),
            'F1-Score': round(metrics['f1_score'], 4),
            'ROC-AUC': round(metrics.get('roc_auc', 0), 4),
            'Training Time (s)': round(training_time, 2)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add ranking columns
    comparison_df['F1_Rank'] = comparison_df['F1-Score'].rank(ascending=False, method='min')
    comparison_df['AUC_Rank'] = comparison_df['ROC-AUC'].rank(ascending=False, method='min')
    comparison_df['Overall_Rank'] = (comparison_df['F1_Rank'] + comparison_df['AUC_Rank']) / 2
    
    # Sort by overall rank
    comparison_df = comparison_df.sort_values('Overall_Rank').reset_index(drop=True)
    
    return comparison_df

def assess_clinical_risk(prediction_proba: float) -> Tuple[str, str]:
    """
    Assess clinical risk level based on prediction probability
    
    Args:
        prediction_proba: Probability of CKD (0-1)
        
    Returns:
        Tuple of (risk_level, risk_color)
    """
    risk_levels = CLINICAL_CONFIG['risk_levels']
    
    for level, thresholds in risk_levels.items():
        if thresholds['min'] <= prediction_proba < thresholds['max']:
            return level, thresholds['color']
    
    # Default to high risk if probability is 1.0
    if prediction_proba >= 1.0:
        return 'high', risk_levels['high']['color']
    
    return 'low', risk_levels['low']['color']

def generate_patient_report(patient_data: Dict, prediction_result: Dict) -> str:
    """
    Generate a clinical report for a patient prediction
    
    Args:
        patient_data: Patient data dictionary
        prediction_result: Prediction result dictionary
        
    Returns:
        Formatted clinical report string
    """
    report = f"""
    CHRONIC KIDNEY DISEASE RISK ASSESSMENT REPORT
    ============================================
    
    Patient ID: {prediction_result.get('patient_id', 'N/A')}
    Assessment Date: {prediction_result.get('timestamp', 'N/A')}
    Model Used: {prediction_result.get('model_used', 'N/A')}
    
    RISK ASSESSMENT:
    ---------------
    Prediction: {prediction_result.get('prediction', 'N/A')}
    Risk Level: {prediction_result.get('risk_level', 'N/A').upper()}
    Confidence: {prediction_result.get('confidence', 0):.1%}
    CKD Probability: {prediction_result.get('probability_ckd', 0):.1%}
    
    IDENTIFIED RISK FACTORS:
    -----------------------
    """
    
    risk_factors = prediction_result.get('risk_factors', [])
    if risk_factors:
        for i, factor in enumerate(risk_factors, 1):
            report += f"    {i}. {factor}\n"
    else:
        report += "    No significant risk factors identified.\n"
    
    report += f"""
    RECOMMENDATIONS:
    ---------------
    """
    
    recommendations = prediction_result.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            report += f"    {i}. {rec}\n"
    
    report += f"""
    
    CLINICAL PARAMETERS:
    -------------------
    """
    
    # Add relevant clinical parameters
    clinical_params = ['age', 'bp', 'sc', 'bgr', 'hemo', 'htn', 'dm']
    for param in clinical_params:
        if param in patient_data and patient_data[param] is not None:
            report += f"    {param.upper()}: {patient_data[param]}\n"
    
    report += f"""
    
    NOTE: This assessment is for clinical decision support only.
    Please consult with a qualified healthcare professional for 
    proper medical evaluation and treatment decisions.
    
    Report generated by CKD Prediction System v1.0
    """
    
    return report

def save_model_artifacts(model, model_name: str, metrics: Dict, 
                        feature_names: List[str], save_dir: str = None):
    """
    Save model and related artifacts
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Performance metrics
        feature_names: List of feature names
        save_dir: Directory to save artifacts
    """
    if save_dir is None:
        save_dir = MODEL_DIR
    
    save_dir = os.path.join(save_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.json')
    save_json(metrics, metrics_path)
    
    # Save feature names
    features_path = os.path.join(save_dir, 'features.json')
    save_json({'feature_names': feature_names}, features_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'creation_timestamp': datetime.now().isoformat(),
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'performance_metrics': metrics
    }
    metadata_path = os.path.join(save_dir, 'metadata.json')
    save_json(metadata, metadata_path)
    
    print(f"Model artifacts saved to: {save_dir}")

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and return quality report
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Data quality report dictionary
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Check for columns with high missing rate
    high_missing = [col for col, pct in report['missing_percentage'].items() if pct > 50]
    report['high_missing_columns'] = high_missing
    
    # Basic statistics for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return report

# Initialize plotting style when module is imported
setup_plotting_style()

if __name__ == "__main__":
    # Create project directories
    create_directories()
    print("Utility functions module initialized successfully!")