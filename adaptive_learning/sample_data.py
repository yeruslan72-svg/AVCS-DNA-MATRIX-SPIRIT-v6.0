# adaptive_learning/sample_data.py
"""
Synthetic data generator for local testing of adaptive layer.
"""
import random
from datetime import datetime
def generate_sample(mode='normal'):
    if mode == 'normal':
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'vibration': round(0.3 + random.random() * 0.7, 3),
            'temperature': 60 + random.random()*6,
            'pressure': 4 + random.random()*1,
            'rpm': 2950 + random.randint(-20,20),
            'load': 1.0 + random.random()*0.1
        }
    if mode == 'degraded':
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'vibration': round(3.0 + random.random() * 2.0, 3),
            'temperature': 75 + random.random()*10,
            'pressure': 6 + random.random()*2,
            'rpm': 2950 + random.randint(-50,50),
            'load': 1.2 + random.random()*0.3
        }
