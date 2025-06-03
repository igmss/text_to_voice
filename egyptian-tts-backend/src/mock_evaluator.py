"""
Mock TTS Evaluator for testing purposes
"""

import numpy as np
from typing import Dict

class VoiceOverEvaluator:
    """Mock voice over evaluator"""
    
    def evaluate(self, audio: np.ndarray) -> float:
        """Evaluate voice over quality (mock implementation)"""
        # Simple mock evaluation based on audio characteristics
        rms = np.sqrt(np.mean(audio**2))
        return min(0.95, max(0.5, rms * 10))  # Mock quality score

class EgyptianTTSEvaluator:
    """Mock Egyptian TTS evaluator"""
    
    def __init__(self, config: Dict):
        """Initialize the evaluator"""
        self.config = config
        self.voice_over_evaluator = VoiceOverEvaluator()
        print("✅ Mock TTS Evaluator initialized")
    
    def evaluate_synthesis(self, audio: np.ndarray, text: str) -> Dict:
        """Evaluate synthesis quality (mock implementation)"""
        quality_score = self.voice_over_evaluator.evaluate(audio)
        
        return {
            'overall_quality': quality_score,
            'pronunciation_accuracy': 0.9,
            'naturalness': 0.85,
            'clarity': 0.88,
            'voice_over_quality': quality_score
        }

