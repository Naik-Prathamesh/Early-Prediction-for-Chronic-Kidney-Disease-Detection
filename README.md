# Chronic Kidney Disease (CKD) Prediction 

## 🏥 Early Prediction for Chronic Kidney Disease Detection: A Progressive Approach to Health Management

A comprehensive machine learning project for early detection and risk assessment of Chronic Kidney Disease using state-of-the-art algorithms and clinical decision support systems.

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)](https://fastapi.tiangolo.com/)


## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)


## 🔍 Overview

Chronic Kidney Disease (CKD) affects approximately 24-28 million people in the U.S., with an additional 20 million unidentified or at risk. This project implements advanced machine learning techniques to enable early detection and risk stratification, facilitating timely medical intervention and improved patient outcomes.

### Key Objectives

- **Early Risk Detection**: Identify CKD risk before clinical symptoms appear
- **Progressive Monitoring**: Track disease progression and risk stratification
- **Clinical Integration**: Provide interpretable results for healthcare professionals
- **Healthcare Compliance**: Ensure HIPAA compliance and regulatory adherence
- **Scalable Deployment**: Support real-time clinical decision support

## ✨ Features

### Machine Learning Models

- **Random Forest**: Ensemble method with feature importance ranking
- **XGBoost**: Gradient boosting with superior handling of imbalanced datasets
- **Support Vector Machine (SVM)**: Robust classification with kernel tricks
- **Neural Networks**: Deep learning for complex pattern recognition

### Clinical Decision Support

- **Risk Stratification**: Multi-level risk categorization (Low/Moderate/High)
- **Feature Importance**: Clinical interpretation of prediction drivers
- **Personalized Recommendations**: Tailored clinical guidance
- **Confidence Scoring**: Uncertainty quantification for decision-making

### Production-Ready API

- **RESTful Endpoints**: Standard HTTP interfaces for clinical systems
- **Real-time Predictions**: Low-latency inference for clinical workflows
- **Batch Processing**: Efficient handling of multiple patient assessments
- **Security Integration**: Authentication and authorization capabilities

### Healthcare Compliance

- **HIPAA Compliance**: End-to-end encryption and audit logging
- **Data Privacy**: Anonymization and de-identification protocols
- **Regulatory Standards**: FDA AI/ML guidelines adherence
- **Clinical Validation**: Healthcare professional review integration

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)
- 4GB+ RAM recommended
- Internet connection for package downloads

### Step 1: Clone or Download

```bash
# Clone from repository
gh repo clone Naik-Prathamesh/Early-Prediction-for-Chronic-Kidney-Disease-Detection
cd ckd-prediction

```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, xgboost, fastapi; print('Installation successful!')"
```



## 🚀 Quick Start

### 1. Train Models

```bash
# Navigate to src directory
cd src

# Run complete training pipeline
python main.py --data-path ../data/chronic_kidney_disease_full.csv

# Train specific models only
python main.py --models rf xgb --data-path ../data/chronic_kidney_disease_full.csv
```

### 2. Start API Server

```bash
# Start development server
python api.py

```
