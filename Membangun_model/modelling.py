import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings('ignore')


def main():
    # 1. Load preprocessed data
    data_path = os.path.join(os.path.dirname(__file__), 'heart_preprocessing.csv')
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")

    # 2. Split data
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Setup MLflow
    mlflow.set_tracking_uri("mlruns")  # Simpan lokal
    mlflow.set_experiment("Heart-Failure-Prediction")

    # 4. Training dengan Autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'='*50}")
        print(f"Model: Random Forest Classifier")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"{'='*50}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    main()