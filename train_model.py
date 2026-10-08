from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/placement_data.csv")
MODEL_PATH = Path("models/placement_model.pkl")
METRICS_PATH = Path("models/metrics.json")

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["placement_status"])
    y = df["placement_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "feature_importances": dict(zip(X.columns, model.feature_importances_.tolist()))
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
