from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import os

# if os.path.exists('.env'):
#     from dotenv import load_dotenv
#     load_dotenv(override=True)


app = Flask(__name__)
CORS(app)  

MODEL_PATH = os.getenv('MODEL_PATH', './Sentiment_Analysis_Predictor.joblib')
VECTORIZER_PATH = os.getenv('VECTORIZER_PATH', './vectorizer.joblib')
# print(MODEL_PATH)
# MODEL_PATH = './Sentiment_Analysis_Predictor.joblib'
# VECTORIZER_PATH = './vectorizer.joblib'

try:
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model, vectorizer = None, None

@app.route('/analyse', methods=['POST'])
def analyse():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Input comment is required'}), 400

    try:
        new_comment_vec = vectorizer.transform([str(text)])
        prediction = model.predict(new_comment_vec)
        print(prediction)
        outcomes = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}
        result = outcomes.get(prediction[0], "Unknown")

        return jsonify({'Result': [result, int(prediction[0])]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def main():
    return jsonify({"message": "Server is running"})

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))  
    app.run(host='0.0.0.0', port=port, debug=True)
