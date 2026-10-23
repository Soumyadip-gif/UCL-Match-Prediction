from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "ucl_ml_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "ucl_match_model.pkl"
FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"
COMPARISON_FILE = MODEL_DIR / "model_comparison.csv"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ==========================================
# FEATURES
# ==========================================

exclude_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Result"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

X = df[feature_columns]
y = df["Result"]


# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

unique_dates = sorted(df["Date"].unique())

split_index = int(len(unique_dates) * 0.80)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_mask = df["Date"].isin(train_dates)
test_mask = df["Date"].isin(test_dates)

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("\n===================================")
print("DRAW-FOCUSED MODEL TRAINING")
print("===================================")

print("Training matches:", len(X_train))
print("Testing matches :", len(X_test))

print("\nTraining period:")
print(train_dates[0], "to", train_dates[-1])

print("\nTesting period:")
print(test_dates[0], "to", test_dates[-1])

print("\nFeatures:", len(feature_columns))


# ==========================================
# MODELS
# ==========================================

models = {

    "Logistic Regression Balanced": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest Balanced": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# TRAINING
# ==========================================

results = []

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print("\n-----------------------------------")
    print(name)
    print("-----------------------------------")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Macro F1 gives equal importance
    # to Home, Draw and Away.
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    draw_recall = recall_score(
        y_test,
        predictions,
        labels=["D"],
        average="macro",
        zero_division=0
    )

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Draw Recall: {draw_recall:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["H", "D", "A"]
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted_F1": f1,
        "Macro_F1": macro_f1,
        "Draw_Recall": draw_recall
    })

    # Select using Macro F1 because
    # we specifically want balanced H/D/A performance.
    if macro_f1 > best_f1:

        best_f1 = macro_f1
        best_model = model
        best_model_name = name


# ==========================================
# SAVE RESULTS
# ==========================================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    "Macro_F1",
    ascending=False
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_FILE
)

joblib.dump(
    feature_columns,
    FEATURE_FILE
)


# ==========================================
# FINAL
# ==========================================

print("\n===================================")
print("DRAW-FOCUSED MODEL COMPARISON")
print("===================================")

print(
    comparison_df.to_string(index=False)
)

print("\nSelected model:", best_model_name)
print(f"Macro F1      : {best_f1:.4f}")

print("\nSaved:")
print(MODEL_FILE)
print(FEATURE_FILE)
print(COMPARISON_FILE)

print("\n===================================")
print("STEP 15 COMPLETED")
print("===================================")