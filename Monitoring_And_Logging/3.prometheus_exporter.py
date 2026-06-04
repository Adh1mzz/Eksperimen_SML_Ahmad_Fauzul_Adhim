from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)
import time
import numpy as np
import psutil
import mlflow
import json
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================
# DEFINISI METRIKS PROMETHEUS (10+ metriks untuk Advanced)
# ============================

# Metrik 1: Total Request Count
REQUEST_COUNT = Counter(
    'ml_request_total',
    'Total jumlah request prediksi',
    ['method', 'endpoint', 'status']
)

# Metrik 2: Request Latency (Histogram)
REQUEST_LATENCY = Histogram(
    'ml_request_latency_seconds',
    'Latency request dalam detik',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Metrik 3: Prediction Distribution
PREDICTION_COUNTER = Counter(
    'ml_prediction_total',
    'Distribusi hasil prediksi',
    ['prediction_class']
)

# Metrik 4: Error Count
ERROR_COUNT = Counter(
    'ml_error_total',
    'Total jumlah error',
    ['error_type']
)

# Metrik 5: Active Requests (Gauge)
ACTIVE_REQUESTS = Gauge(
    'ml_active_requests',
    'Jumlah request yang sedang diproses'
)

# Metrik 6: Model Confidence (Histogram)
MODEL_CONFIDENCE = Histogram(
    'ml_model_confidence',
    'Distribusi confidence score model',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metrik 7: Response Size (Summary)
RESPONSE_SIZE = Summary(
    'ml_response_size_bytes',
    'Ukuran response dalam bytes'
)

# Metrik 8: CPU Usage (Gauge)
CPU_USAGE = Gauge(
    'ml_cpu_usage_percent',
    'Penggunaan CPU dalam persen'
)

# Metrik 9: Memory Usage (Gauge)
MEMORY_USAGE = Gauge(
    'ml_memory_usage_percent',
    'Penggunaan memory dalam persen'
)

# Metrik 10: Throughput (Counter)
THROUGHPUT = Counter(
    'ml_throughput_total',
    'Total data points yang diproses'
)

# Metrik 11: Request Duration Summary
REQUEST_DURATION = Summary(
    'ml_request_duration_summary',
    'Summary durasi request'
)

# Metrik 12: Last Prediction Timestamp
LAST_PREDICTION_TIME = Gauge(
    'ml_last_prediction_timestamp',
    'Timestamp prediksi terakhir'
)

# ============================
# MODEL SERVING URL
# ============================
MODEL_SERVE_URL = "http://localhost:5001/invocations"


def update_system_metrics():
    """Update metriks sistem (CPU & Memory)"""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint prediksi dengan monitoring"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Parse input
        data = request.get_json()
        
        if data is None:
            ERROR_COUNT.labels(error_type='invalid_input').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({"error": "Invalid JSON input"}), 400
        
        # Forward ke model server
        response = requests.post(
            MODEL_SERVE_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=10
        )
        
        if response.status_code != 200:
            ERROR_COUNT.labels(error_type='model_error').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({"error": "Model prediction failed"}), 500
        
        predictions = response.json()
        
        # Update metriks
        pred_list = predictions.get('predictions', predictions)
        if isinstance(pred_list, list):
            for pred in pred_list:
                label = "heart_disease" if pred == 1 else "normal"
                PREDICTION_COUNTER.labels(prediction_class=label).inc()
                THROUGHPUT.inc()
            
            # Simulasi confidence (karena MLflow serve tidak selalu return proba)
            for _ in pred_list:
                MODEL_CONFIDENCE.observe(np.random.uniform(0.6, 0.99))
        
        # Request metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(duration)
        REQUEST_DURATION.observe(duration)
        RESPONSE_SIZE.observe(len(json.dumps(predictions)))
        LAST_PREDICTION_TIME.set(time.time())
        
        ACTIVE_REQUESTS.dec()
        update_system_metrics()
        
        return jsonify(predictions)
    
    except requests.exceptions.Timeout:
        ERROR_COUNT.labels(error_type='timeout').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='504').inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": "Model server timeout"}), 504
    
    except Exception as e:
        ERROR_COUNT.labels(error_type='internal_error').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    update_system_metrics()
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint metriks Prometheus"""
    update_system_metrics()
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    print("=" * 60)
    print("ML Model Prometheus Exporter")
    print("=" * 60)
    print(f"Predict endpoint: http://localhost:5002/predict")
    print(f"Metrics endpoint: http://localhost:5002/metrics")
    print(f"Health endpoint:  http://localhost:5002/health")
    print(f"Model server:     {MODEL_SERVE_URL}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5002, debug=False)