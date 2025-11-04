# adaptive_learning/feedback_controller.py
"""
Feedback Controller: safe suggestion generator for actuator setpoints.
DOES NOT APPLY CHANGES AUTOMATICALLY — returns proposed actions.
"""
from typing import Dict

class FeedbackController:
    def __init__(self):
        # tuning parameters (example)
        self.max_damper = 8000
        self.min_damper = 0

    def propose_actions(self, risk_index: float, context: Dict):
        """
        risk_index: 0-100
        context: result from ContextManager
        Returns dict { 'damper_force': int, 'notes': str }
        """
        if risk_index >= 85:
            force = self.max_damper
            note = "Emergency damping"
        elif risk_index >= 60:
            force = int(self.max_damper * 0.6)
            note = "High damping"
        elif risk_index >= 30:
            force = int(self.max_damper * 0.25)
            note = "Moderate damping"
        else:
            force = int(self.max_damper * 0.06)
            note = "Standby damping"

        # context adjustments
        if context.get('mode') == 'night_high_load':
            force = min(self.max_damper, int(force * 1.1))
            note += "; adjusted for night high load"

        # clamp
        force = max(self.min_damper, min(self.max_damper, force))
        return {'damper_force': force, 'note': note}
