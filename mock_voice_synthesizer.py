"""
Simplified Voice Synthesizer for testing purposes
This is a mock implementation to test the Flask API structure
"""

import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Union

class VoiceSynthesizer:
    """Simplified voice synthesizer for testing"""
    
    def __init__(self):
        """Initialize the synthesizer"""
        self.sample_rate = 22050
        print("✅ Mock Voice Synthesizer initialized")
    
    def synthesize_voice_over(self, text: str, speaker_id: str = 'default', 
                            voice_preset: str = 'commercial-warm', 
                            custom_settings: Optional[Dict] = None) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate a mock voice over (sine wave for testing)
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker voice ID
            voice_preset: Voice preset name
            custom_settings: Custom synthesis settings
            
        Returns:
            Tuple of (audio_array, sample_rate, metadata)
        """
        # Generate a simple sine wave as mock audio
        duration = min(len(text) * 0.1, 10.0)  # Rough duration based on text length
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some variation based on text
        if len(text) > 50:
            audio = audio * np.linspace(1, 0.5, len(audio))
        
        metadata = {
            'text': text,
            'speaker_id': speaker_id,
            'voice_preset': voice_preset,
            'duration': duration,
            'quality_metrics': {
                'voice_over_quality': 0.85,
                'clarity': 0.9,
                'naturalness': 0.8
            }
        }
        
        return audio, self.sample_rate, metadata
    
    def save_audio(self, audio: np.ndarray, sample_rate: int, filepath: str):
        """Save audio to file"""
        sf.write(filepath, audio, sample_rate)
        return filepath

