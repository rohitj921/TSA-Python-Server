from pymongo import MongoClient
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from flask import send_file
from google import genai

from dotenv import load_dotenv
import time
import json
import re
import os
import logging

# loading environment variables from .env file if it exists
if os.path.exists('.env'):
    load_dotenv(override=True)

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%I:%M:%S %p",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)
# loading Model and vectorizer
try:
    MODEL_PATH = os.getenv(
        'MODEL_PATH', './Sentiment_Analysis_Predictor.joblib')
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

GEMINI_API_KEY = os.getenv("KEY")
MONGO_URI = os.getenv("MONGO_URI")

# flask app setup
app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# mongo db setup
# mongo_client = MongoClient(MONGO_URI)
# db = mongo_client["Talent-Skills-Alliance"]
# collection = db["reviews"]

# setting though globals in process-reviews endpoint
mongo_client = None
db = None
collection = None

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    logging.warning("Gemini API key not found. Skipping Gemini integration.")

OUTCOMES = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}
LOCAL_RATING_MAP = {0: 5, 1: 2, 2: 5, 3: 8}


def check_mongo_connection():

    global mongo_client, db, collection

    try:
        mongo_client.admin.command('ping')
        return True
    except Exception as e:
        logging.error(f"MongoDB connection lost: {e}")
        try:
            mongo_client = MongoClient(MONGO_URI)
            db = mongo_client["Talent-Skills-Alliance"]
            collection = db["reviews"]
            logging.info("Reconnected to MongoDB")
            return True
        except Exception as re:
            logging.critical(f"Failed to reconnect MongoDB: {re}")
            return False


# route for processing reviews
@app.route("/process-reviews", methods=["GET"])
def process_reviews():

    if not check_mongo_connection():
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        if not model or not vectorizer:
            logging.error(
                "Model or vectorizer not loaded. Cannot process reviews.")
            return jsonify({"error": "Model not loaded"}), 500

        # unprocessed = list(collection.find({"isProcessed": False}))
        unprocessed = list(collection.find({
            "$or": [{"isProcessed": False}, {"isProcessed": {"$exists": False}}]
        }))
        print("unprocessed reviews:", len(unprocessed))
        if not unprocessed:
            logging.info("No unprocessed reviews found")
            return jsonify({"message": "No unprocessed reviews found"}), 200

        affected_review_for_ids = set()

        for review in unprocessed:
            text = review.get("review", "").strip()
            if not text:
                logging.warning(
                    f"Review ID {review['_id']} skipped due to missing or empty text")
                continue

            # Local prediction
            vec = vectorizer.transform([text])
            local_pred = model.predict(vec)[0]
            local_sentiment = OUTCOMES.get(local_pred, "Unknown")
            local_rating = LOCAL_RATING_MAP.get(local_pred, 5)

            gemini_sentiment, rating = gemini_sentiment_rating(text)
            if gemini_sentiment == "Unknown":
                rating = local_rating
                isProcessed = False
            else:
                isProcessed = True

            collection.update_one(
                {"_id": review["_id"]},
                {
                    "$set": {
                        "rating": rating,
                        "isProcessed": isProcessed,
                    }
                }
            )
            logging.info(
                f"Updated Review ID {review['_id']} → Comment: {text}, Rating: {rating}, Local: {local_sentiment}, Gemini: {gemini_sentiment}")

            review_for_id = review.get("reviewFor")
            if review_for_id:
                affected_review_for_ids.add(review_for_id)

        print("Affected reviewFor IDs:", affected_review_for_ids)
        # performing summarization for affected reviewFor IDs
        for review_for_id in affected_review_for_ids:
            try:
                all_reviews = list(collection.find({
                    "reviewFor": review_for_id,
                    "review": {"$ne": None}
                }))
                combined_text = "\n".join(
                    [r["review"] for r in all_reviews if r.get("review")])
                if not combined_text.strip():
                    logging.info(
                        f"Skipping summary for {review_for_id}: No valid comments")
                    continue

                summary_prompt = (
                    "You are a review summarizer AI. Summarize the overall opinion about the person below "
                    "in 1-2 concise sentences, no more than 200 characters in total. Focus on the tone and recurring themes. "
                    "Avoid repetition, lists, or generic phrases.\n\n"
                    f"Reviews:\n{combined_text}"
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=summary_prompt)
                summary_text = response.text.strip()
                print("combined text:\n", combined_text)
                print("Summary:\n", summary_text)
                collection.update_many(
                    {"reviewFor": review_for_id},
                    {"$set": {"extraRemarks": summary_text}}
                )
                logging.info(f"Updated summary for reviewFor: {review_for_id}")

            except Exception as e:
                logging.error(f"Summarization failed for {review_for_id}: {e}")

        return jsonify({"message": f"Processed {len(unprocessed)} reviews and summarized {len(affected_review_for_ids)} users."}), 200

    except Exception as e:
        logging.error(f"Error processing reviews: {e}")
        return jsonify({"error": "Failed to process reviews", "details": str(e)}), 500


def gemini_sentiment_rating(comment, retries=3, delay=2):
    prompt = (
        "You are a sentiment analysis expert. "
        "Given the following review comment, classify it and assign a numeric rating from 1 to 10. "
        "Base this on how positive/negative the review is.\n\n"
        "Output JSON with exactly this format:\n"
        "{\"sentiment\": \"Neutral\", \"rating\": 5}\n\n"
        f"Review: \"{comment}\"\n"
    )
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt)
            text = response.text.strip()
            # print(text)
            json_text = re.search(r"\{.*\}", text)
            if json_text:
                result = json.loads(json_text.group())
                sentiment = result.get("sentiment", "Unknown").capitalize()
                rating = int(result.get("rating", 5)) if result.get(
                    "rating") is not None else 5
            else:
                sentiment, rating = "Unknown", 5

            logging.info(
                f"Gemini response: {sentiment} (Rating: {rating}) in {time.perf_counter() - start:.2f}s")
            return sentiment, rating
        except Exception as e:
            logging.warning(f"Gemini API error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Gemini failed after {retries} retries.")
                return "Unknown", 5


# sentiment, rating = get_gemini_sentiment_rating("terribly good")
# print(sentiment, rating)
@app.route("/summary", methods=["GET"])
def summarize_user_reviews():
    if not check_mongo_connection():
        logging.error("Cannot summarize: MongoDB connection failed")
        return jsonify({"error": "MongoDB connection failed"}), 500

    try:
        review_groups = collection.aggregate([
            {"$match": {"reviewFor": {"$exists": True}, "review": {"$ne": None}}},
            {"$group": {
                "_id": "$reviewFor",
                "comments": {"$push": "$review"}
            }}
        ])
        # print(len(list(review_groups)), "groups found")
        updated_count = 0
        for group in list(review_groups):
            review_for_id = group["_id"]
            comments = group["comments"]

            if len(comments) < 1:
                logging.info(f"Skipping {review_for_id}: Not enough comments")
                continue
            # print("inloop")
            combined_text = "\n".join(comments)
            # prompt = (
            #     "You are a review summarizer. Summarize the overall opinion about this person "
            #     "based on multiple reviews below in 2-3 sentences. Avoid repetition or listing comments. "
            #     f"\n\nReviews:\n{combined_text}"
            # )
            prompt = (
                "You are a review summarizer AI. Summarize the overall opinion about the person below "
                "in 1-2 concise sentences, no more than 200 characters in total. Focus on the tone and recurring themes. "
                "Avoid repetition, lists, or generic phrases.\n\n"
                f"Reviews:\n{combined_text}"
            )
            # print(prompt)
            try:
                # response = client.models.generate_content(
                #     model="gemini-2.0", contents=prompt)
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt)
                summary_text = response.text.strip()
                print(summary_text)
                collection.update_many(
                    {"reviewFor": review_for_id},
                    {"$set": {"extraRemarks": summary_text}}
                )
                updated_count += 1
                logging.info(
                    f"Updated extraRemarks for reviewFor: {review_for_id}")
            except Exception as e:
                logging.error(
                    f"Gemini error while summarizing for {review_for_id}: {e}")
        return jsonify({"message": f"Summarization complete. Updated {updated_count} users."}), 200

    except Exception as e:
        logging.error(f"Failed to summarize user reviews: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/', methods=['GET'])
def main():
    return jsonify({"message": "Server is running!"}), 200


@app.route("/logs", methods=["GET"])
def fetch_logs():
    if not os.path.exists("logs/server.log"):
        return jsonify({"error": "No logs found."}), 404

    return send_file("logs/server.log", as_attachment=True, download_name="server.txt")


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed on this endpoint. Please check the API documentation."}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Something went wrong on our end'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': "Invalid endpoint. Please check the URL and try again."}), 404

# @app.teardown_appcontext
# def close_mongo_connection(exception):
#     mongo_client.close()


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

        outcomes = {0: "Irrelevant", 1: "Negative",
                    2: "Neutral", 3: "Positive"}
        result = outcomes.get(prediction[0], "Unknown")
        logging.info(f"Prediction result: {result} ({int(prediction[0])})")

        return jsonify({'Result': result, "Sentiment": int(prediction[0])}), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
