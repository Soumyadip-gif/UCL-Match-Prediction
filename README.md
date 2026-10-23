# ⚽ UCL Match Prediction

An AI-powered football analytics web application that predicts UEFA Champions League match outcomes, expected goals, and predicted scores using machine learning.

## Features
- Match outcome prediction: Home Win, Draw, or Away Win
- Expected home and away goals
- Predicted scoreline
- Match probabilities and analysis report
- Prediction history using SQLite
- Interactive, responsive football-themed dashboard

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/UCL-Match-Prediction.git
cd UCL-Match-Prediction
python -m venv venv
```

Activate the environment and install dependencies:

```bash
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Dataset
Historical UEFA Champions League match data from [OpenFootball](https://github.com/openfootball/champions-league).

## Disclaimer
Predictions are machine-learning estimates based on historical data and are not guaranteed match results.

## Author
**Soumyadip Porel**
