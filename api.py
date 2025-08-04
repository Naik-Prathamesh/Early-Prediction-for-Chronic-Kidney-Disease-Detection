import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import pickle
import traceback
from datetime import datetime
import logging

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create required directories
os.makedirs('src', exist_ok=True)
os.makedirs('src/models', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Create empty __init__.py files
with open('src/__init__.py', 'w') as f:
    f.write('')
    
with open('src/models/__init__.py', 'w') as f:
    f.write('')

print("src/__init__.py already exists")
print("src/models/__init__.py already exists")

# Import our custom modules
try:
    from model_training import BaseModel
    print("✅ Successfully imported BaseModel")
except ImportError as e:
    print(f"❌ Error importing BaseModel: {e}")
    print("Please ensure model_training.py is in the same directory")

# Initialize Flask app
app = Flask(__name__)
app.config['DEBUG'] = False

# Setup logging
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Global variables for model
model = None
model_loaded = False

# CKD feature definitions for the web interface
CKD_FEATURES = {
    'age': {'type': 'number', 'min': 1, 'max': 120, 'description': 'Age in years'},
    'bp': {'type': 'number', 'min': 50, 'max': 250, 'description': 'Blood Pressure (mm/Hg)'},
    'sg': {'type': 'select', 'options': [1.005, 1.010, 1.015, 1.020, 1.025], 'description': 'Specific Gravity'},
    'al': {'type': 'number', 'min': 0, 'max': 5, 'description': 'Albumin (0-5)'},
    'su': {'type': 'number', 'min': 0, 'max': 5, 'description': 'Sugar (0-5)'},
    'rbc': {'type': 'select', 'options': ['normal', 'abnormal'], 'description': 'Red Blood Cells'},
    'pc': {'type': 'select', 'options': ['normal', 'abnormal'], 'description': 'Pus Cell'},
    'pcc': {'type': 'select', 'options': ['present', 'notpresent'], 'description': 'Pus Cell Clumps'},
    'ba': {'type': 'select', 'options': ['present', 'notpresent'], 'description': 'Bacteria'},
    'bgr': {'type': 'number', 'min': 50, 'max': 500, 'description': 'Blood Glucose Random (mgs/dl)'},
    'bu': {'type': 'number', 'min': 10, 'max': 200, 'description': 'Blood Urea (mgs/dl)'},
    'sc': {'type': 'number', 'min': 0.1, 'max': 15.0, 'step': 0.1, 'description': 'Serum Creatinine (mgs/dl)'},
    'sod': {'type': 'number', 'min': 100, 'max': 180, 'description': 'Sodium (mEq/L)'},
    'pot': {'type': 'number', 'min': 2.0, 'max': 8.0, 'step': 0.1, 'description': 'Potassium (mEq/L)'},
    'hemo': {'type': 'number', 'min': 5.0, 'max': 20.0, 'step': 0.1, 'description': 'Hemoglobin (gms)'},
    'pcv': {'type': 'number', 'min': 20, 'max': 60, 'description': 'Packed Cell Volume'},
    'wc': {'type': 'number', 'min': 2000, 'max': 20000, 'description': 'White Blood Cell Count (cells/cumm)'},
    'rc': {'type': 'number', 'min': 2.0, 'max': 8.0, 'step': 0.1, 'description': 'Red Blood Cell Count (millions/cmm)'},
    'htn': {'type': 'select', 'options': ['yes', 'no'], 'description': 'Hypertension'},
    'dm': {'type': 'select', 'options': ['yes', 'no'], 'description': 'Diabetes Mellitus'},
    'cad': {'type': 'select', 'options': ['yes', 'no'], 'description': 'Coronary Artery Disease'},
    'appet': {'type': 'select', 'options': ['good', 'poor'], 'description': 'Appetite'},
    'pe': {'type': 'select', 'options': ['yes', 'no'], 'description': 'Pedal Edema'},
    'ane': {'type': 'select', 'options': ['yes', 'no'], 'description': 'Anemia'}
}

def load_or_train_model():
    """
    Load existing model or train a new one if none exists.
    """
    global model, model_loaded
    
    try:
        # Try to load existing model
        model_files = [f for f in os.listdir('models') if f.endswith('_preprocessor.pkl')]
        
        if model_files:
            # Load the most recent model
            latest_preprocessor = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
            model_file = latest_preprocessor.replace('_preprocessor.pkl', '.pkl')
            
            model = BaseModel("CKD_API_Model")
            model.load_model(
                os.path.join('models', model_file),
                os.path.join('models', latest_preprocessor)
            )
            model_loaded = True
            print(f"✅ Loaded existing model: {model_file}")
            
        else:
            # Train new model
            print("🔄 No existing model found. Training new model...")
            model = BaseModel("CKD_API_Model")
            model.load_and_preprocess_data()
            model.train_models()
            
            # Perform hyperparameter tuning for best performance
            model.hyperparameter_tuning('Random Forest')
            
            # Save the trained model
            model.save_model()
            model_loaded = True
            print("✅ New model trained and saved successfully!")
            
    except Exception as e:
        print(f"❌ Error loading/training model: {str(e)}")
        traceback.print_exc()
        model_loaded = False

# HTML template for the web interface (fixed CSS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chronic Kidney Disease Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #34495e;
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="number"]:focus, select:focus {
            border-color: #3498db;
            outline: none;
        }
        .submit-btn {
            background-color: #3498db;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
            width: 100%;
            margin-top: 20px;
        }
        .submit-btn:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            font-size: 18px;
            text-align: center;
            display: none;
        }
        .result.positive {
            background-color: #e74c3c;
            color: white;
        }
        .result.negative {
            background-color: #27ae60;
            color: white;
        }
        .loading {
            text-align: center;
            color: #3498db;
            display: none;
        }
        .description {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 3px;
        }
        .api-info {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-top: 30px;
        }
        .api-info h3 {
            color: #2c3e50;
            margin-top: 0;
        }
        .api-endpoint {
            background-color: #34495e;
            color: white;
            padding: 10px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Chronic Kidney Disease Prediction System</h1>
        
        <form id="ckdForm">
            <div class="form-grid">
                {% for feature, config in features.items() %}
                <div class="form-group">
                    <label for="{{ feature }}">{{ config.description }}</label>
                    {% if config.type == 'number' %}
                    <input type="number" 
                           id="{{ feature }}" 
                           name="{{ feature }}" 
                           min="{{ config.get('min', '') }}" 
                           max="{{ config.get('max', '') }}" 
                           step="{{ config.get('step', 'any') }}" 
                           required>
                    {% elif config.type == 'select' %}
                    <select id="{{ feature }}" name="{{ feature }}" required>
                        <option value="">Select...</option>
                        {% for option in config.options %}
                        <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                    {% endif %}
                    <div class="description">{{ config.description }}</div>
                </div>
                {% endfor %}
            </div>
            
            <button type="submit" class="submit-btn">🔬 Predict CKD Risk</button>
        </form>
        
        <div class="loading" id="loading">
            <h3>🔄 Analyzing patient data...</h3>
        </div>
        
        <div class="result" id="result"></div>
        
       
    </div>

    <script>
        document.getElementById('ckdForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Collect form data
            const formData = new FormData(this);
            const data = {};
            
            for (let [key, value] of formData.entries()) {    
                if (!isNaN(value) && value !== '') {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show result
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (result.success) {
                    const prediction = result.prediction;
                    const probabilities = result.probabilities;
                    
                    if (prediction === 'CKD') {
                        resultDiv.className = 'result positive';
                        resultDiv.innerHTML = `
                            <h3>⚠️ CKD Risk Detected</h3>
                            <p><strong>Prediction:</strong> ${prediction}</p>
                            ${probabilities ? `<p><strong>CKD Probability:</strong> ${(probabilities.CKD * 100).toFixed(1)}%</p>` : ''}
                            <p><em>Please consult with a healthcare professional for proper diagnosis and treatment.</em></p>
                        `;
                    } else {
                        resultDiv.className = 'result negative';
                        resultDiv.innerHTML = `
                            <h3>✅ Low CKD Risk</h3>
                            <p><strong>Prediction:</strong> ${prediction}</p>
                            ${probabilities ? `<p><strong>No CKD Probability:</strong> ${(probabilities['Not CKD'] * 100).toFixed(1)}%</p>` : ''}
                            <p><em>Continue maintaining a healthy lifestyle and regular check-ups.</em></p>
                        `;
                    }
                } else {
                    resultDiv.className = 'result positive';
                    resultDiv.innerHTML = `<h3>❌ Prediction Error</h3><p>${result.error}</p>`;
                }
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result positive';
                resultDiv.innerHTML = `<h3>❌ Connection Error</h3><p>Unable to connect to the prediction service.</p>`;
            }
        });
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def home():
    """Serve the main web interface for CKD prediction."""
    return render_template_string(HTML_TEMPLATE, features=CKD_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for CKD prediction."""
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided. Please send JSON data with patient features.'
            }), 400
        
        # Validate required features
        missing_features = []
        for feature in CKD_FEATURES.keys():
            if feature not in data:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing required features: {missing_features}'
            }), 400
        
        # Make prediction
        result = model.predict_single(data)
        
        # Log the prediction
        logging.info(f"Prediction made: {result['prediction']} for data: {data}")
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'prediction_code': result['prediction_code'],
            'probabilities': result['probabilities'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Prediction error: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {error_msg}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        info = {
            'success': True,
            'model_name': model.model_name,
            'best_model': model.best_model_name if hasattr(model, 'best_model_name') else 'Unknown',
            'best_score': model.best_score,
            'features': list(CKD_FEATURES.keys()),
            'feature_count': len(CKD_FEATURES),
            'model_type': type(model.best_model).__name__ if model.best_model else 'Unknown'
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'error': 'Bad request'}), 400

@app.errorhandler(404)  
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting CKD Prediction API...")
    print("Loading/training model...")
    
    # Load or train model
    load_or_train_model()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
        print("🌐 Starting Flask server...")
        print("📊 Web interface available at: http://localhost:5000")
        print("🔌 API endpoint: http://localhost:5000/predict")
        print("📋 Model info: http://localhost:5000/model_info")
        print("❤️ Health check: http://localhost:5000/health")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check the logs.")