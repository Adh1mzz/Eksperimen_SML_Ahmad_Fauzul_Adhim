from prometheus_client import start_http_server, Gauge, Counter
import time
import random


PREDICTION_COUNT = Counter('model_prediction_total', 'Total predictions made')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
LATENCY = Gauge('prediction_latency_seconds', 'Time taken for prediction')


PRECISION = Gauge('model_precision', 'Current model precision')
RECALL = Gauge('model_recall', 'Current model recall')
F1_SCORE = Gauge('model_f1_score', 'Current model F1 score')
CPU_USAGE = Gauge('process_cpu_usage', 'CPU usage of the model server (%)')
MEM_USAGE = Gauge('process_memory_usage_bytes', 'Memory usage of the model server (MB)')
ERROR_RATE = Counter('model_prediction_errors_total', 'Total prediction errors')
INPUT_VALUE_MEAN = Gauge('input_feature_mean', 'Mean value of input features')

def monitor_performance():
    """Simulasi monitoring yang berjalan otomatis selamanya"""
    print("Mulai mengirim data ke Prometheus... (Tekan Ctrl+C untuk berhenti)")
    while True:
        
        PREDICTION_COUNT.inc(random.randint(1, 5))
        
        
        MODEL_ACCURACY.set(0.869 + random.uniform(-0.01, 0.01))
        PRECISION.set(0.890 + random.uniform(-0.01, 0.01))
        RECALL.set(0.872 + random.uniform(-0.01, 0.01))
        F1_SCORE.set(0.881 + random.uniform(-0.01, 0.01))
        
        
        LATENCY.set(0.01 + random.uniform(0, 0.1)) 
        CPU_USAGE.set(random.uniform(5, 35))        
        MEM_USAGE.set(random.uniform(200, 400))     
        INPUT_VALUE_MEAN.set(random.uniform(30, 60))

         
        if random.random() < 0.05:
            ERROR_RATE.inc()

        
        time.sleep(5)

if __name__ == '__main__':
    
    PORT = 5002
    print("=" * 50)
    print(f"Prometheus Exporter Berjalan di Port {PORT}")
    print("=" * 50)
    
    start_http_server(PORT)
    monitor_performance()