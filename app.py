from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import warnings
from flask import send_file
from PIL import Image, ImageDraw

# Suppress specific warnings
warnings.filterwarnings("ignore", message="X has feature names, but GradientBoostingClassifier was fitted without feature names")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Set environment variable for joblib to avoid the physical cores warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

app = Flask(__name__)

# Load model artifacts
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"

# Load models - Fixed to match the actual model name
try:
    # Try to load the stacking ensemble model with the correct name
    ensemble_model = joblib.load(MODEL_DIR / 'stackingensemble.pkl')
    # For backward compatibility, try loading xgboost if it exists
    try:
        xgboost_model = joblib.load(MODEL_DIR / 'xgboost.pkl')
        has_xgboost = True
    except FileNotFoundError:
        has_xgboost = False
except FileNotFoundError as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Model 'stackingensemble.pkl' not found in {MODEL_DIR}")

# Load feature names
try:
    feature_names = joblib.load(MODEL_DIR / 'feature_names.pkl')
except FileNotFoundError as e:
    app.logger.error(f"Error loading feature names: {str(e)}")
    raise RuntimeError(f"Feature names file not found in {MODEL_DIR}")

# Define a dictionary to store training medians for imputation
try:
    training_medians = joblib.load(MODEL_DIR / 'training_medians.pkl')
except FileNotFoundError:
    # Create default medians (should be replaced with actual values)
    app.logger.warning("Training medians not found, using default values")
    training_medians = {name: 0 for name in feature_names}

# Try to load scaler if available
try:
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    has_scaler = True
    app.logger.info("Scaler loaded successfully")
except FileNotFoundError:
    has_scaler = False
    app.logger.warning("No scaler found, using raw features")

# Custom JSON encoder to handle non-serializable numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ---------------------- Helper Functions ----------------------
def calculate_bmi(weight, height):
    """Calculate BMI given weight in kg and height in cm"""
    try:
        if height <= 0 or weight <= 0:
            return None
        return round(weight / ((height/100) ** 2), 1)
    except Exception:
        return None

def calculate_bun_creatinine_ratio(bu, sc):
    """Calculate BUN to Creatinine ratio from blood urea and serum creatinine"""
    try:
        if sc <= 0 or bu <= 0:
            return None
        # Convert BU (blood urea) to BUN (blood urea nitrogen)
        # BUN = BU * 0.467 (standard conversion factor)
        bun = bu * 0.467
        return round(bun / sc, 2)
    except Exception:
        return None

def validate_input(data):
    """Validate the input data based on reasonable ranges"""
    # Define reasonable ranges for each feature (adjust as needed)
    required_fields = {
        'age': (0, 120),
        'bp': (60, 240),
        'sc': (0.5, 15.0),
        'hemo': (3.0, 20.0),
        'bgr': (50, 500),
        'bu': (10, 300),
        'sg': (1.000, 1.040),
        'sod': (100, 160),
        'pot': (2.5, 7.5),
        'wc': (2000, 20000),  # corrected from wbcc
        'rc': (2.0, 8.0),     # corrected from rbcc
        'pcv': (20, 60),
        'bmi': (15.0, 50.0),
        'al': (0, 5),
        'su': (0, 5)
        # Categorical features don't need range validation
    }
    
    # Define binary fields
    binary_fields = ['dm', 'htn', 'pc', 'pcc', 'ba', 'rbc', 'appetite']
    
    errors = []
    
    # Check required fields that are in feature_names
    for field in required_fields:
        if field in feature_names and field not in data:
            errors.append(f"Field {field} must be filled")
    
    # Validate binary fields
    for field in binary_fields:
        if field in feature_names and field in data:
            try:
                val = int(data[field])
                if val not in [0, 1]:
                    errors.append(f"{field} must be 0 or 1")
            except:
                errors.append(f"Invalid format for {field}, must be 0 or 1")
    
    # Validate data types for non-binary fields
    for field, value in data.items():
        if field in feature_names and field not in binary_fields:
            try:
                data[field] = float(value)
            except:
                errors.append(f"Invalid format for {field}")
    
    # Validate value ranges for non-binary fields
    for field, (min_val, max_val) in required_fields.items():
        if field in data and field in feature_names:
            try:
                val = float(data[field])
                if not (min_val <= val <= max_val):
                    errors.append(f"Value of {field} must be between {min_val}-{max_val}")
            except:
                pass  # Already caught in previous validation
    
    return errors

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create DataFrame with all feature names
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = [0.0] * len(feature_names)  # Initialize with floats to avoid dtype issues
    
    # Fill in provided values
    for feature in feature_names:
        if feature in data:
            # Convert to float explicitly to avoid dtype warnings
            try:
                input_df.at[0, feature] = float(data[feature])
            except ValueError:
                # If conversion fails, use median value
                if feature in training_medians:
                    input_df.at[0, feature] = float(training_medians[feature])
    
    # Calculate bun_to_creatinine_ratio if it's in feature names and not provided
    if 'bun_to_creatinine_ratio' in feature_names and 'bun_to_creatinine_ratio' not in data:
        if 'bu' in data and 'sc' in data:
            ratio = calculate_bun_creatinine_ratio(float(data['bu']), float(data['sc']))
            if ratio:
                input_df.at[0, 'bun_to_creatinine_ratio'] = ratio
            elif 'bun_to_creatinine_ratio' in training_medians:
                input_df.at[0, 'bun_to_creatinine_ratio'] = float(training_medians['bun_to_creatinine_ratio'])
    
    # Handle missing values using training medians
    for col in input_df.columns:
        if pd.isna(input_df.loc[0, col]) and col in training_medians:
            input_df.at[0, col] = float(training_medians[col])
    
    # Apply scaling if available
    if has_scaler:
        try:
            # Transform and convert to numpy array (dropping feature names)
            scaled_values = scaler.transform(input_df)
            return scaled_values
        except Exception as e:
            app.logger.error(f"Error scaling input data: {str(e)}")
            return input_df.values
    else:
        # If no scaler is found, return the DataFrame values
        return input_df.values

def get_ensemble_prediction(processed_data):
    """Get prediction from ensemble model"""
    try:
        prediction = ensemble_model.predict(processed_data)[0]
        probability = ensemble_model.predict_proba(processed_data)[0][1]
        # Convert numpy types to Python native types
        return int(prediction), float(probability)
    except Exception as e:
        app.logger.error(f"Ensemble prediction error: {str(e)}")
        return None, None

def get_xgboost_prediction(processed_data):
    """Get prediction from XGBoost model if available"""
    if not has_xgboost:
        return None, None
        
    try:
        prediction = xgboost_model.predict(processed_data)[0]
        probability = xgboost_model.predict_proba(processed_data)[0][1]
        # Convert numpy types to Python native types
        return int(prediction), float(probability)
    except Exception as e:
        app.logger.error(f"XGBoost prediction error: {str(e)}")
        return None, None

def get_combined_prediction(processed_data):
    """Combine predictions from available models"""
    ensemble_pred, ensemble_prob = get_ensemble_prediction(processed_data)
    
    # Only try XGBoost prediction if the model is available
    if has_xgboost:
        xgboost_pred, xgboost_prob = get_xgboost_prediction(processed_data)
    else:
        xgboost_pred, xgboost_prob = None, None
    
    results = {}
    
    # Add ensemble prediction if available
    if ensemble_prob is not None:
        results.update({
            'ensemble_prediction': int(ensemble_pred),
            'ensemble_probability': round(float(ensemble_prob), 4),
        })
    
    # Add XGBoost prediction if available
    if xgboost_prob is not None:
        results.update({
            'xgboost_prediction': int(xgboost_pred),
            'xgboost_probability': round(float(xgboost_prob), 4),
        })
    
    # Determine combined prediction
    if ensemble_prob is not None and xgboost_prob is not None:
        # Average the probabilities if both models are available
        avg_probability = (ensemble_prob + xgboost_prob) / 2
        combined_prediction = 1 if avg_probability >= 0.5 else 0
        
        results.update({
            'combined_prediction': int(combined_prediction),
            'combined_probability': round(float(avg_probability), 4)
        })
    elif ensemble_prob is not None:
        # Use only ensemble if XGBoost is not available
        results.update({
            'combined_prediction': int(ensemble_pred),
            'combined_probability': round(float(ensemble_prob), 4)
        })
    elif xgboost_prob is not None:
        # Use only XGBoost if ensemble is not available (shouldn't happen)
        results.update({
            'combined_prediction': int(xgboost_pred),
            'combined_probability': round(float(xgboost_prob), 4)
        })
    else:
        # If neither model made a prediction
        return {'error': 'Failed to generate predictions from the models'}
    
    return results

# ---------------------- Routes ----------------------
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if request.content_type == 'application/json':
            data = request.json
        else:
            data = request.form.to_dict()
        
        # Calculate BMI if height and weight are provided
        try:
            weight = float(data.get('weight', 0))
            height = float(data.get('height', 0))
            bmi = calculate_bmi(weight, height)
            if bmi:
                data['bmi'] = bmi
            elif 'bmi' in training_medians:
                data['bmi'] = float(training_medians['bmi'])
        except Exception as e:
            app.logger.warning(f"BMI calculation error: {str(e)}")
            if 'bmi' in training_medians:
                data['bmi'] = float(training_medians['bmi'])
        
        # Calculate BUN to creatinine ratio if not provided
        try:
            if 'bun_to_creatinine_ratio' not in data and 'bu' in data and 'sc' in data:
                bu = float(data.get('bu', 0))
                sc = float(data.get('sc', 0))
                ratio = calculate_bun_creatinine_ratio(bu, sc)
                if ratio:
                    data['bun_to_creatinine_ratio'] = ratio
        except Exception as e:
            app.logger.warning(f"BUN:Creatinine ratio calculation error: {str(e)}")
        
        # Handle feature name mappings - correct any mismatched names
        feature_mapping = {
            'wbcc': 'wc',  # White blood cell count
            'rbcc': 'rc',  # Red blood cell count
            'bp': 'bp',    # Blood pressure
        }
        
        # Apply mappings
        for old_name, new_name in feature_mapping.items():
            if old_name in data and new_name in feature_names and old_name not in feature_names:
                data[new_name] = data.pop(old_name)
        
        # Log input data for debugging
        app.logger.debug(f"Input data: {data}")
        
        # Validate input
        errors = validate_input(data)
        if errors:
            app.logger.warning(f"Validation errors: {errors}")
            return jsonify({'errors': errors}), 400
        
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Get predictions from models
        predictions = get_combined_prediction(processed_data)
        
        if 'error' in predictions:
            app.logger.error(predictions['error'])
            return jsonify({'error': predictions['error']}), 500
        
        # Add interpretation and confidence
        combined_pred = predictions['combined_prediction']
        combined_prob = predictions['combined_probability']
        
        predictions['interpretation'] = 'CKD Detected' if combined_pred == 1 else 'No CKD Detected'
        
        # Determine confidence level
        if combined_prob > 0.85 or combined_prob < 0.15:
            confidence = 'high'
        elif combined_prob > 0.7 or combined_prob < 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        predictions['confidence'] = confidence
        
        # Include BMI if calculated
        if 'bmi' in data:
            predictions['bmi'] = float(data['bmi'])
        
        # Include BUN/creatinine ratio if calculated
        if 'bun_to_creatinine_ratio' in data:
            predictions['bun_to_creatinine_ratio'] = float(data['bun_to_creatinine_ratio'])
        
        # Log prediction results
        app.logger.info(f"Prediction: {predictions['interpretation']} with {confidence} confidence")
        
        # Use the custom JSON encoder to handle numpy types
        return app.response_class(
            response=json.dumps(predictions, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'System error: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the loaded models"""
    try:
        available_models = ['stackingensemble']
        if has_xgboost:
            available_models.append('xgboost')
            
        return jsonify({
            'models': available_models,
            'feature_names': list(feature_names),
            'has_scaler': has_scaler
        })
    except Exception as e:
        app.logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({'error': f'Error retrieving model info: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    try:
        # More comprehensive health check
        model_files = os.listdir(MODEL_DIR)
        return jsonify({
            'status': 'ok', 
            'models_loaded': True,
            'available_models': {
                'stackingensemble': True,
                'xgboost': has_xgboost
            },
            'feature_count': len(feature_names),
            'model_files': model_files
        })
    except Exception as e:
        app.logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Return available features and their expected ranges"""
    try:
        # Enhanced feature ranges with descriptions
        feature_ranges = {
            'age': {'min': 0, 'max': 120, 'unit': 'years', 'description': 'Patient age'},
            'bp': {'min': 60, 'max': 240, 'unit': 'mm/Hg', 'description': 'Blood pressure'},
            'sg': {'min': 1.000, 'max': 1.040, 'unit': 'specific gravity', 'description': 'Specific gravity of urine'},
            'hemo': {'min': 3.0, 'max': 20.0, 'unit': 'g/dL', 'description': 'Hemoglobin level'},
            'bgr': {'min': 50, 'max': 500, 'unit': 'mg/dL', 'description': 'Blood glucose random'},
            'bu': {'min': 10, 'max': 300, 'unit': 'mg/dL', 'description': 'Blood urea'},
            'sc': {'min': 0.5, 'max': 15.0, 'unit': 'mg/dL', 'description': 'Serum creatinine'},
            'sod': {'min': 100, 'max': 160, 'unit': 'mEq/L', 'description': 'Sodium level'},
            'pot': {'min': 2.5, 'max': 7.5, 'unit': 'mEq/L', 'description': 'Potassium level'},
            'wc': {'min': 2000, 'max': 20000, 'unit': 'cells/cubic mm', 'description': 'White blood cell count'},
            'rc': {'min': 2.0, 'max': 8.0, 'unit': 'million cells/cubic mm', 'description': 'Red blood cell count'},
            'bmi': {'min': 15.0, 'max': 50.0, 'unit': 'kg/m²', 'description': 'Body Mass Index'},
            'pcv': {'min': 20, 'max': 60, 'unit': '%', 'description': 'Packed cell volume'},
            'al': {'min': 0, 'max': 5, 'unit': 'level', 'description': 'Albumin level (0-5 scale)'},
            'su': {'min': 0, 'max': 5, 'unit': 'level', 'description': 'Sugar level (0-5 scale)'},
            'dm': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Diabetes mellitus (0=No, 1=Yes)'},
            'htn': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Hypertension (0=No, 1=Yes)'},
            'pc': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Pus cells (0=Normal, 1=Abnormal)'},
            'pcc': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Pus cell clumps (0=Absent, 1=Present)'},
            'ba': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Bacteria (0=Absent, 1=Present)'},
            'rbc': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Red blood cells (0=Normal, 1=Abnormal)'},
            'appetite': {'min': 0, 'max': 1, 'unit': 'boolean', 'description': 'Appetite (0=Good, 1=Poor)'},
            'bun_to_creatinine_ratio': {'min': 5, 'max': 50, 'unit': 'ratio', 'description': 'BUN to creatinine ratio (calculated)'}
        }
        
        # Only include features that are actually in the model
        available_features = {
            feature: feature_ranges.get(feature, {'min': None, 'max': None, 'unit': 'unknown', 'description': 'Not described'})
            for feature in feature_names
        }
        
        return jsonify({
            'features': available_features,
            'medians': {k: float(v) for k, v in training_medians.items() if k in feature_names},
            'feature_count': len(feature_names)
        })
    except Exception as e:
        app.logger.error(f"Error retrieving feature info: {str(e)}")
        return jsonify({'error': f'Error retrieving feature info: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback about predictions to potentially improve the model"""
    try:
        if request.content_type == 'application/json':
            data = request.json
        else:
            data = request.form.to_dict()
        
        # Required fields for feedback
        required = ['prediction_id', 'actual_outcome', 'model_prediction', 'user_rating']
        
        # Validate required fields
        missing = [field for field in required if field not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
        
        # Here you could store feedback in a database
        # For now, we just log it
        app.logger.info(f"Feedback received: {data}")
        
        # In a real application, you might store this in a database for later model improvement
        
        return jsonify({'status': 'success', 'message': 'Feedback recorded successfully'})
        
    except Exception as e:
        app.logger.error(f"Error recording feedback: {str(e)}")
        return jsonify({'error': f'Error recording feedback: {str(e)}'}), 500

@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder_image(width, height):
    # Generate simple placeholder image
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Placeholder Image", fill=(180,180,180))
    
    from io import BytesIO
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

# Add additional error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Enable logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    
    # Print startup information
    print(f"CKD Prediction Web App starting...")
    print(f"Models loaded from: {MODEL_DIR}")
    print(f"Number of features: {len(feature_names)}")
    
    # Check if models directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"WARNING: Models directory not found at {MODEL_DIR}. Creating directory.")
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check model files
    required_files = ['stackingensemble.pkl', 'feature_names.pkl']
    missing_files = [f for f in required_files if not os.path.exists(MODEL_DIR / f)]
    if missing_files:
        print(f"ERROR: Required model files missing: {', '.join(missing_files)}")
        print(f"Please ensure these files are in the {MODEL_DIR} directory.")
        exit(1)
    
    print(f"Ready to make predictions.")
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000)