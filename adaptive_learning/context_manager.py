# adaptive_learning/context_manager.py
"""
Context Manager: rule-based context inference + light ML hooks.
Returns a context dict describing the current operational mode.
"""
from datetime import datetime

class ContextManager:
    def __init__(self):
        # example thresholds / heuristic rules - configurable
        self.night_hours = (0, 6)
        self.high_load_threshold = 1.3  # multiplier above nominal

    def infer_context(self, telemetry: dict, metadata: dict = None):
        """
        telemetry: { 'vibration': float, 'temperature': float, 'pressure': float, 'rpm': float, 'load': float }
        metadata: optional { 'operator': str, 'shift': 'A' }
        """
        ctx = {}
        now = datetime.utcnow()
        ctx['hour'] = now.hour
        ctx['is_night'] = (self.night_hours[0] <= now.hour <= self.night_hours[1])
        load = telemetry.get('load', 1.0)
        ctx['high_load'] = load > self.high_load_threshold
        ctx['operator'] = metadata.get('operator') if metadata else None
        ctx['shift'] = metadata.get('shift') if metadata else None

        # mode selection
        if ctx['is_night'] and ctx['high_load']:
            ctx['mode'] = 'night_high_load'
        elif ctx['high_load']:
            ctx['mode'] = 'high_load'
        else:
            ctx['mode'] = 'normal'
        return ctx
