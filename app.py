from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable to store artifacts
artifacts = None

def load_model_artifacts():
    """Load model artifacts with detailed error handling"""
    global artifacts
    try:
        import predictor
        
        print("🔍 Checking artifacts folder...")
        if not os.path.exists("artifacts"):
            print("❌ artifacts folder not found!")
            return False
            
        artifact_files = os.listdir("artifacts")
        print("📁 Artifact files:", artifact_files)
        
        # Check for required files
        required_files = [
            'category_model.pkl', 'category_vectorizer.pkl', 
            'retrieval_vectorizer.pkl', 'label_vectors_med.pkl',
            'label_vectors_proc.pkl', 'label_metadata.json'
        ]
        
        missing_files = [f for f in required_files if f not in artifact_files]
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
            
        print("✅ All required artifact files found!")
        artifacts = predictor.load_artifacts("artifacts")
        print("✅ Artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    return render_template('index.ejs')

@app.route('/api/health')
def health_check():
    status = 'healthy' if artifacts else 'unhealthy'
    return jsonify({'status': status, 'message': 'Medical Treatment Prediction API'})

@app.route('/api/predict-treatment', methods=['POST'])
def predict_treatment():
    try:
        if artifacts is None:
            return jsonify({'error': 'Models not loaded'}), 500
            
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        condition = data.get('condition', '')
        
        if not symptoms or not condition:
            return jsonify({'error': 'Both symptoms and condition are required'}), 400
        
        import predictor
        category, confidence, meds, procs = predictor.predict_treatment(
            artifacts=artifacts,
            symptom=symptoms,
            condition=condition,
            top_k=data.get('topK', 5)
        )
        
        return jsonify({
            'category': category,
            'confidence': confidence,
            'medications': [{'name': m[0], 'score': float(m[1])} for m in meds],
            'procedures': [{'name': p[0], 'score': float(p[1])} for p in procs]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Medical Treatment Prediction API...")
    print("Python executable:", sys.executable)
    
    if load_model_artifacts():
        print("✅ Server starting successfully!")
        print("🌐 Access at: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Server failed to start - check errors above")