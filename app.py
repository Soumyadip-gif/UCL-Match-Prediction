from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from flask_cors import CORS

from src.predict import predict_match
from src.database import (
    initialize_database,
    save_prediction,
    get_predictions
)


app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
initialize_database()


# ==========================================
# LOAD TEAM LIST
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ucl_matches_all.csv"
)

matches = pd.read_csv(DATA_PATH)

teams = sorted(
    set(matches["HomeTeam"]).union(
        set(matches["AwayTeam"])
    )
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        teams=teams
    )


# ==========================================
# PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        home_team = data.get("home_team")
        away_team = data.get("away_team")

        if not home_team or not away_team:

            return jsonify({
                "success": False,
                "error": "Please select both teams."
            }), 400

        if home_team == away_team:

            return jsonify({
                "success": False,
                "error": "Home and away teams must be different."
            }), 400

        result = predict_match(
            home_team,
            away_team
        )

        save_prediction(result)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==========================================
# PREDICTION HISTORY API
# ==========================================

@app.route("/history", methods=["GET"])
def history():

    try:

        predictions = get_predictions()

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )