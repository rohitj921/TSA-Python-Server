from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import os
import logging

# if os.path.exists('.env'):
#     from dotenv import load_dotenv
#     load_dotenv(override=True)

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%I:%M:%S %p",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)

try:
    MODEL_PATH = os.getenv('MODEL_PATH', './Sentiment_Analysis_Predictor.joblib')
    VECTORIZER_PATH = os.getenv('VECTORIZER_PATH', './vectorizer.joblib')
    # print(MODEL_PATH)
    # MODEL_PATH = './Sentiment_Analysis_Predictor.joblib'
    # VECTORIZER_PATH = './vectorizer.joblib'
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    logging.info("Model and vectorizer loaded successfully.")
    
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    logging.error(f"Failed to load model/vectorizer: {e}")
    model, vectorizer = None, None

@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        if not request.is_json:
            logging.warning("Invalid request: Expected JSON")
            return jsonify({'error': 'Invalid request. Expected JSON'}), 400
        
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logging.warning("Empty or malformed JSON request received")
            return jsonify({"error": "Invalid JSON format or empty request body"}), 400

        if model is None or vectorizer is None:
            logging.critical("Model or vectorizer not loaded properly")
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        text = str(data.get('comment', '')).strip()
        if not text:
            logging.warning("Missing 'comment' field in request")
            return jsonify({'error': "Input 'comment' is required"}), 400
        
        if text.isdigit():
            logging.warning("Comment cannot be just a number")
            return jsonify({'error': "Input 'comment' cannot be just a number"}), 400
        
        logging.info(f"Processing comment: {text}")
        new_comment_vec = vectorizer.transform([text])
        prediction = model.predict(new_comment_vec)
        
        outcomes = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}
        result = outcomes.get(prediction[0], "Unknown")
        logging.info(f"Prediction result: {result} ({int(prediction[0])})")
        
        return jsonify({'Result': result, "Sentiment": int(prediction[0])}), 200
    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/', methods=['GET'])
def main():
    return jsonify({"message": "Server is running!"}), 200

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed on this endpoint. Please check the API documentation."}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Something went wrong on our end'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': "Invalid endpoint. Please check the URL and try again."}), 404

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))  
    app.run(host='0.0.0.0', port=port, debug=False)
