import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss
)
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')



# --- KONFIGURASI DagsHub (Advanced) ---
DAGSHUB_USER = 'Adh1mzz'
DAGSHUB_REPO = 'Eksperimen_SML_Ahmad_Fauzul_Adhim'

# import dagshub
# dagshub.init(repo_owner='Adh1mzz', repo_name='Eksperimen_SML_Ahmad_Fauzul_Adhim', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/Adh1mzz/Eksperimen_SML_Ahmad_Fauzul_Adhim.mlflow")


def load_data():
    """Load preprocessed dataset"""
    data_path = os.path.join(os.path.dirname(__file__), 'heart_preprocessing.csv')
    df = pd.read_csv(data_path)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """Artefak tambahan 1: Confusion Matrix Plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Heart Disease'],
                yticklabels=['Normal', 'Heart Disease'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    """Artefak tambahan 2: Feature Importance Plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices], color='#3498db', edgecolor='black')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
    return None


def train_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test):
    """
    Training model dengan MANUAL LOGGING ke MLflow
    Mencakup semua metriks autolog + 2 artefak tambahan
    """
    with mlflow.start_run(run_name=model_name):
        # ============================
        # Log Parameters (manual)
        # ============================
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # ============================
        # Training
        # ============================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ============================
        # Log Metrics (manual - sama dengan autolog)
        # ============================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("log_loss", logloss)
        
        # Training metrics juga
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        mlflow.log_metric("training_accuracy", train_accuracy)
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Log Loss:  {logloss:.4f}")
        print(f"{'='*50}")
        
        # ============================
        # Log Model (manual)
        # ============================
        mlflow.sklearn.log_model(model, "model")
        
        # ============================
        # ARTEFAK TAMBAHAN (Advanced)
        # ============================
        
        # Artefak 1: Confusion Matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, f"confusion_matrix_{model_name}.png")
        mlflow.log_artifact(cm_path)
        print(f"  ✓ Artefak logged: Confusion Matrix")
        
        # Artefak 2: Feature Importance
        fi_path = plot_feature_importance(model, list(X_train.columns), f"feature_importance_{model_name}.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
            print(f"  ✓ Artefak logged: Feature Importance")
        
        # Artefak 3: Classification Report (sebagai JSON)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = f"classification_report_{model_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        print(f"  ✓ Artefak logged: Classification Report (JSON)")
        
        # Artefak 4: Dataset info
        dataset_info = {
            "total_samples": len(X_train) + len(X_test),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "features": list(X_train.columns),
            "target_distribution_train": y_train.value_counts().to_dict(),
            "target_distribution_test": y_test.value_counts().to_dict()
        }
        info_path = f"dataset_info_{model_name}.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        mlflow.log_artifact(info_path)
        print(f"  ✓ Artefak logged: Dataset Info (JSON)")
        
        # Cleanup temp files
        for f in [cm_path, fi_path, report_path, info_path]:
            if f and os.path.exists(f):
                os.remove(f)
        
        return accuracy


def main():
    # Setup MLflow
    mlflow.set_tracking_uri("mlruns")  # Ganti dengan DagsHub URI untuk Advanced
    mlflow.set_experiment("Heart-Failure-Tuning")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ============================
    # MODEL 1: Random Forest + Hyperparameter Tuning
    # ============================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING - Random Forest")
    print("=" * 60)
    
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    print(f"\nBest params: {rf_grid.best_params_}")
    print(f"Best CV score: {rf_grid.best_score_:.4f}")
    
    # Log best model
    rf_accuracy = train_and_log_model(
        model=rf_grid.best_estimator_,
        model_name="RandomForest_Tuned",
        params=rf_grid.best_params_,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    
    # ============================
    # MODEL 2: Gradient Boosting + Hyperparameter Tuning
    # ============================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING - Gradient Boosting")
    print("=" * 60)
    
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    
    print(f"\nBest params: {gb_grid.best_params_}")
    print(f"Best CV score: {gb_grid.best_score_:.4f}")
    
    gb_accuracy = train_and_log_model(
        model=gb_grid.best_estimator_,
        model_name="GradientBoosting_Tuned",
        params=gb_grid.best_params_,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    
    # ============================
    # MODEL 3: Logistic Regression + Hyperparameter Tuning
    # ============================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING - Logistic Regression")
    print("=" * 60)
    
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }
    
    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(lr, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    
    print(f"\nBest params: {lr_grid.best_params_}")
    print(f"Best CV score: {lr_grid.best_score_:.4f}")
    
    lr_accuracy = train_and_log_model(
        model=lr_grid.best_estimator_,
        model_name="LogisticRegression_Tuned",
        params=lr_grid.best_params_,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    
    # ============================
    # SUMMARY
    # ============================
    print("\n" + "=" * 60)
    print("RINGKASAN HASIL")
    print("=" * 60)
    print(f"Random Forest:       {rf_accuracy:.4f}")
    print(f"Gradient Boosting:   {gb_accuracy:.4f}")
    print(f"Logistic Regression: {lr_accuracy:.4f}")
    print(f"\n✅ Buka MLflow UI: mlflow ui")
    print(f"   Lalu buka http://localhost:5000 di browser")


if __name__ == "__main__":
    main()