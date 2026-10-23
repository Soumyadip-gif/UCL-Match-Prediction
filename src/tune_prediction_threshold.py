from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "ucl_ml_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "ucl_match_model.pkl"
FEATURE_FILE = BASE_DIR / "models" / "feature_columns.pkl"
THRESHOLD_FILE = BASE_DIR / "models" / "prediction_settings.pkl"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

feature_columns = joblib.load(FEATURE_FILE)

X = df[feature_columns]
y = df["Result"]


# ==========================================
# CHRONOLOGICAL TEST SPLIT
# ==========================================

unique_dates = sorted(df["Date"].unique())

test_split = int(len(unique_dates) * 0.80)

train_dates = unique_dates[:test_split]
test_dates = unique_dates[test_split:]

train_mask = df["Date"].isin(train_dates)
test_mask = df["Date"].isin(test_dates)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# ==========================================
# VALIDATION SPLIT
# ==========================================

train_unique_dates = sorted(
    X_train.index.to_series()
    .map(df["Date"])
    .unique()
)

validation_split = int(
    len(train_unique_dates) * 0.80
)

fit_dates = train_unique_dates[:validation_split]
validation_dates = train_unique_dates[validation_split:]

fit_mask = df["Date"].isin(fit_dates)
validation_mask = df["Date"].isin(validation_dates)

X_fit = X[fit_mask]
y_fit = y[fit_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]


# ==========================================
# TRAIN VALIDATION MODEL
# ==========================================

validation_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

validation_model.fit(
    X_fit,
    y_fit
)


# ==========================================
# VALIDATION PROBABILITIES
# ==========================================

validation_probabilities = (
    validation_model.predict_proba(
        X_validation
    )
)

classes = list(
    validation_model.classes_
)

draw_index = classes.index("D")


# ==========================================
# NORMAL PREDICTION
# ==========================================

normal_indices = np.argmax(
    validation_probabilities,
    axis=1
)

normal_predictions = np.array(
    classes
)[normal_indices]

normal_macro_f1 = f1_score(
    y_validation,
    normal_predictions,
    average="macro",
    zero_division=0
)


# ==========================================
# FIND DRAW THRESHOLD
# ==========================================

best_threshold = 0.0
best_macro_f1 = normal_macro_f1

threshold_results = []

for threshold in np.arange(
    0.20,
    0.61,
    0.01
):

    predictions = []

    for probabilities in validation_probabilities:

        draw_probability = probabilities[
            draw_index
        ]

        if draw_probability >= threshold:
            prediction = "D"

        else:
            prediction = classes[
                np.argmax(probabilities)
            ]

        predictions.append(prediction)

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    draw_recall = f1_score(
        y_validation,
        predictions,
        labels=["D"],
        average="macro",
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Macro_F1": macro_f1,
        "Draw_F1": draw_recall
    })

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_threshold = threshold


# ==========================================
# RETRAIN ON ALL TRAINING DATA
# ==========================================

final_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_train,
    y_train
)


# ==========================================
# TEST PROBABILITIES
# ==========================================

test_probabilities = final_model.predict_proba(
    X_test
)

classes = list(
    final_model.classes_
)

draw_index = classes.index("D")


# ==========================================
# NORMAL TEST PREDICTION
# ==========================================

normal_test_indices = np.argmax(
    test_probabilities,
    axis=1
)

normal_test_predictions = np.array(
    classes
)[normal_test_indices]


# ==========================================
# THRESHOLD TEST PREDICTION
# ==========================================

threshold_predictions = []

for probabilities in test_probabilities:

    draw_probability = probabilities[
        draw_index
    ]

    if draw_probability >= best_threshold:
        prediction = "D"

    else:
        prediction = classes[
            np.argmax(probabilities)
        ]

    threshold_predictions.append(
        prediction
    )


# ==========================================
# RESULTS
# ==========================================

normal_accuracy = accuracy_score(
    y_test,
    normal_test_predictions
)

normal_macro_f1 = f1_score(
    y_test,
    normal_test_predictions,
    average="macro",
    zero_division=0
)

threshold_accuracy = accuracy_score(
    y_test,
    threshold_predictions
)

threshold_macro_f1 = f1_score(
    y_test,
    threshold_predictions,
    average="macro",
    zero_division=0
)


# ==========================================
# OUTPUT
# ==========================================

print("\n===================================")
print("PROBABILITY-BASED PREDICTION")
print("===================================")

print(
    "\nValidation period:",
    validation_dates[0],
    "to",
    validation_dates[-1]
)

print(
    "\nTest period:",
    test_dates[0],
    "to",
    test_dates[-1]
)

print("\nNormal validation Macro F1:")
print(f"{normal_macro_f1:.4f}")

print("\nSelected draw threshold:")
print(f"{best_threshold:.2f}")

print("\nValidation Macro F1 after tuning:")
print(f"{best_macro_f1:.4f}")


print("\n-----------------------------------")
print("TEST — NORMAL PREDICTION")
print("-----------------------------------")

print(f"Accuracy : {normal_accuracy:.4f}")
print(f"Macro F1 : {normal_macro_f1:.4f}")

print(
    classification_report(
        y_test,
        normal_test_predictions,
        labels=["H", "D", "A"],
        zero_division=0
    )
)

print(
    confusion_matrix(
        y_test,
        normal_test_predictions,
        labels=["H", "D", "A"]
    )
)


print("\n-----------------------------------")
print("TEST — THRESHOLD PREDICTION")
print("-----------------------------------")

print(f"Accuracy : {threshold_accuracy:.4f}")
print(f"Macro F1 : {threshold_macro_f1:.4f}")

print(
    classification_report(
        y_test,
        threshold_predictions,
        labels=["H", "D", "A"],
        zero_division=0
    )
)

print(
    confusion_matrix(
        y_test,
        threshold_predictions,
        labels=["H", "D", "A"]
    )
)


# ==========================================
# SAVE FINAL MODEL + SETTINGS
# ==========================================

joblib.dump(
    final_model,
    MODEL_FILE
)

joblib.dump(
    {
        "draw_threshold": float(best_threshold),
        "classes": classes
    },
    THRESHOLD_FILE
)


print("\n===================================")
print("PROBABILITY TUNING COMPLETED")
print("===================================")

print("Model saved:")
print(MODEL_FILE)

print("\nPrediction settings saved:")
print(THRESHOLD_FILE)