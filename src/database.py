import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "data" / "predictions.db"


def get_connection():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_FILE)


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            created_at TEXT NOT NULL,

            home_team TEXT NOT NULL,

            away_team TEXT NOT NULL,

            home_probability REAL NOT NULL,

            draw_probability REAL NOT NULL,

            away_probability REAL NOT NULL,

            prediction TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(result):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            created_at,
            home_team,
            away_team,
            home_probability,
            draw_probability,
            away_probability,
            prediction
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        result["home_team"],
        result["away_team"],

        result["home_win_probability"],
        result["draw_probability"],
        result["away_win_probability"],

        result["prediction"]
    ))

    connection.commit()
    connection.close()


def get_predictions():

    connection = get_connection()

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]

