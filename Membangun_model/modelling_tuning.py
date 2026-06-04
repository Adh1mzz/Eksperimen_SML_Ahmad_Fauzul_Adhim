import mlflow
import mlflow.sklearn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
import dagshub
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI DAGSHUB (ADVANCE) ---
# Menggunakan username dan repo Anda (Tidak diubah)
DAGSHUB_USER = "Adh1mzz"
DAGSHUB_REPO = "Eksperimen_SML_Ahmad_Fauzul_Adhim"

# Inisialisasi Dagshub untuk MLflow tracking
dagshub.init(repo_owner='Adh1mzz', repo_name='Eksperimen_SML_Ahmad_Fauzul_Adhim', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Adh1mzz/Eksperimen_SML_Ahmad_Fauzul_Adhim.mlflow")

mlflow.set_experiment("Heart-Disease-Tuning")

# ==========================================
# DATA LOADING & SPLITTING
# ==========================================
# Load data hasil preprocessing (menggunakan pandas)
data_path = os.path.join(os.path.dirname(__file__), 'heart_preprocessing.csv')
df = pd.read_csv(data_path)

# Split data (Target = HeartDisease)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Ukuran X_train : {X_train.shape}")
print(f"Ukuran X_test  : {X_test.shape}")

# Definisikan Hyperparameter Space untuk Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'random_state': [42]
}

# --- MANUAL LOGGING (SKILLED/ADVANCE) ---
with mlflow.start_run(run_name="RF_Tuning_GridSearch"):

    # Grid Search
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log Best Params Manual
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_value)

    # Evaluasi
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log Metrics Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # --- ARTEFAK TAMBAHAN (ADVANCE) ---

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Normal (0)', 'Heart Disease (1)'],
        yticklabels=['Normal (0)', 'Heart Disease (1)']
    )
    plt.title('Confusion Matrix - Best Model')
    plt.savefig('tuning_cm.png')
    mlflow.log_artifact('tuning_cm.png')

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('tuning_roc.png')
    mlflow.log_artifact('tuning_roc.png')

    # 3. Model Save
    mlflow.sklearn.log_model(best_model, "best_rf_model")

    # Cleanup local files
    os.remove('tuning_cm.png')
    os.remove('tuning_roc.png')

    print("\n" + "="*45)
    print("   HASIL EVALUASI — Best Random Forest")
    print("="*45)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("="*45)
    print("✅ Training Selesai! Data telah dikirim ke DagsHub.")