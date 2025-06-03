"""
Mock Audio Processor for testing purposes
"""

import numpy as np
import soundfile as sf
from typing import Dict, Tuple

class AudioProcessor:
    """Simplified audio processor for testing"""
    
    def __init__(self, target_sr: int = 22050, target_bit_depth: int = 16):
        """Initialize the audio processor"""
        self.target_sr = target_sr
        self.target_bit_depth = target_bit_depth
        print("✅ Mock Audio Processor initialized")
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            audio, sr = sf.read(filepath)
            return audio, sr
        except Exception as e:
            # Return mock audio if file can't be loaded
            duration = 2.0
            t = np.linspace(0, duration, int(self.target_sr * duration))
            audio = 0.1 * np.sin(2 * np.pi * 440 * t)
            return audio, self.target_sr
    
    def assess_quality(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Assess audio quality (mock implementation)"""
        # Simple mock quality metrics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        quality_metrics = {
            'rms_level': float(rms),
            'peak_level': float(peak),
            'dynamic_range': float(peak - rms),
            'overall_quality': 0.8,
            'clarity': 0.85,
            'noise_level': 0.1
        }
        
        return quality_metrics

