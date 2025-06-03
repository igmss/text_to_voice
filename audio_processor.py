"""
Audio preprocessing module for Egyptian Arabic TTS training data.

This module handles audio quality enhancement, normalization, and feature extraction
specifically optimized for voice over quality Egyptian Arabic speech synthesis.
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from scipy.io import wavfile
import noisereduce as nr
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Professional audio processor for Egyptian Arabic TTS training data.
    Optimized for voice over quality standards.
    """
    
    def __init__(self, target_sr: int = 48000, target_bit_depth: int = 24):
        """
        Initialize audio processor with voice over quality standards.
        
        Args:
            target_sr: Target sampling rate (48kHz for voice over)
            target_bit_depth: Target bit depth (24-bit for voice over)
        """
        self.target_sr = target_sr
        self.target_bit_depth = target_bit_depth
        self.min_snr_db = 20.0  # Minimum signal-to-noise ratio
        self.max_silence_duration = 2.0  # Maximum silence duration in seconds
        self.voice_activity_threshold = 0.01  # Voice activity detection threshold
        
        # Voice over quality standards
        self.vo_standards = {
            'peak_level_db': -3.0,      # Peak normalization level
            'noise_floor_db': -50.0,    # Maximum noise floor
            'dynamic_range_db': 40.0,   # Minimum dynamic range
            'frequency_range': (80, 8000),  # Voice frequency range
        }
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Use librosa for robust audio loading
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            return audio, sr
        except Exception as e:
            # Fallback to soundfile
            try:
                audio, sr = sf.read(file_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                return audio, sr
            except Exception as e2:
                raise ValueError(f"Could not load audio file {file_path}: {e2}")
    
    def assess_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Assess audio quality metrics for voice over standards.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal-to-noise ratio estimation
        snr_db = self.estimate_snr(audio)
        
        # Dynamic range
        dynamic_range = self.calculate_dynamic_range(audio)
        
        # Peak level
        peak_level_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        
        # RMS level
        rms_level_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        
        # Frequency response assessment
        freq_response = self.assess_frequency_response(audio, sr)
        
        # Clipping detection
        clipping_percentage = self.detect_clipping(audio)
        
        # Silence ratio
        silence_ratio = self.calculate_silence_ratio(audio)
        
        return {
            'snr_db': snr_db,
            'dynamic_range_db': dynamic_range,
            'peak_level_db': peak_level_db,
            'rms_level_db': rms_level_db,
            'frequency_response_score': freq_response,
            'clipping_percentage': clipping_percentage,
            'silence_ratio': silence_ratio,
            'sample_rate': sr,
            'duration_seconds': len(audio) / sr,
            'meets_vo_standards': self.meets_vo_standards(snr_db, dynamic_range, peak_level_db, clipping_percentage)
        }
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation using voice activity detection
        voice_segments = self.detect_voice_activity(audio)
        
        if len(voice_segments) == 0:
            return 0.0
        
        # Calculate signal power from voice segments
        signal_power = np.mean([np.mean(audio[start:end]**2) for start, end in voice_segments])
        
        # Estimate noise from silent segments
        noise_segments = self.detect_silence_segments(audio)
        if len(noise_segments) > 0:
            noise_power = np.mean([np.mean(audio[start:end]**2) for start, end in noise_segments])
        else:
            noise_power = np.min(audio**2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return 10 * np.log10(snr_linear + 1e-10)
        else:
            return 60.0  # Very high SNR if no noise detected
    
    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        peak_level = np.max(np.abs(audio))
        noise_floor = np.percentile(np.abs(audio), 5)  # 5th percentile as noise floor
        
        if noise_floor > 0:
            return 20 * np.log10(peak_level / noise_floor)
        else:
            return 60.0  # Very high dynamic range
    
    def assess_frequency_response(self, audio: np.ndarray, sr: int) -> float:
        """Assess frequency response quality (0-1 score)."""
        # Calculate power spectral density
        freqs, psd = scipy.signal.welch(audio, sr, nperseg=2048)
        
        # Focus on voice frequency range
        voice_freq_mask = (freqs >= self.vo_standards['frequency_range'][0]) & \
                         (freqs <= self.vo_standards['frequency_range'][1])
        
        voice_psd = psd[voice_freq_mask]
        voice_freqs = freqs[voice_freq_mask]
        
        # Calculate spectral flatness in voice range
        geometric_mean = np.exp(np.mean(np.log(voice_psd + 1e-10)))
        arithmetic_mean = np.mean(voice_psd)
        
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Convert to quality score (higher flatness = better for voice)
        return min(spectral_flatness * 10, 1.0)
    
    def detect_clipping(self, audio: np.ndarray) -> float:
        """Detect clipping percentage."""
        # Detect samples at or near maximum amplitude
        threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        return (clipped_samples / len(audio)) * 100
    
    def calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silence to total duration."""
        silence_segments = self.detect_silence_segments(audio)
        silence_duration = sum(end - start for start, end in silence_segments)
        return silence_duration / len(audio)
    
    def meets_vo_standards(self, snr_db: float, dynamic_range_db: float, 
                          peak_level_db: float, clipping_percentage: float) -> bool:
        """Check if audio meets voice over quality standards."""
        return (snr_db >= self.min_snr_db and
                dynamic_range_db >= self.vo_standards['dynamic_range_db'] and
                peak_level_db <= self.vo_standards['peak_level_db'] and
                clipping_percentage < 1.0)
    
    def enhance_audio(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Enhance audio quality for voice over standards.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (enhanced_audio, enhancement_metadata)
        """
        enhanced = audio.copy()
        metadata = {'enhancements_applied': []}
        
        # 1. Noise reduction
        if self.estimate_snr(enhanced) < self.min_snr_db:
            enhanced = self.reduce_noise(enhanced, sr)
            metadata['enhancements_applied'].append('noise_reduction')
        
        # 2. Normalize audio levels
        enhanced = self.normalize_levels(enhanced)
        metadata['enhancements_applied'].append('level_normalization')
        
        # 3. Apply gentle compression for voice over
        enhanced = self.apply_voice_compression(enhanced)
        metadata['enhancements_applied'].append('voice_compression')
        
        # 4. High-pass filter to remove low-frequency noise
        enhanced = self.apply_highpass_filter(enhanced, sr, cutoff=80)
        metadata['enhancements_applied'].append('highpass_filter')
        
        # 5. De-essing (reduce harsh sibilants)
        enhanced = self.apply_deessing(enhanced, sr)
        metadata['enhancements_applied'].append('deessing')
        
        # 6. Final quality assessment
        final_quality = self.assess_quality(enhanced, sr)
        metadata['final_quality'] = final_quality
        
        return enhanced, metadata
    
    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction using spectral gating."""
        try:
            # Use noisereduce library for spectral noise reduction
            reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
            return reduced
        except:
            # Fallback: simple spectral subtraction
            return self.spectral_subtraction(audio, sr)
    
    def spectral_subtraction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple spectral subtraction for noise reduction."""
        # Estimate noise spectrum from first 0.5 seconds
        noise_samples = int(0.5 * sr)
        noise_spectrum = np.abs(np.fft.fft(audio[:noise_samples]))
        
        # Apply spectral subtraction
        window_size = 2048
        hop_size = window_size // 4
        enhanced = np.zeros_like(audio)
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            spectrum = np.fft.fft(window)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Subtract noise spectrum
            clean_magnitude = magnitude - 0.5 * noise_spectrum[:len(magnitude)]
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            clean_spectrum = clean_magnitude * np.exp(1j * phase)
            clean_window = np.real(np.fft.ifft(clean_spectrum))
            
            enhanced[i:i + window_size] += clean_window
        
        return enhanced
    
    def normalize_levels(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels for voice over standards."""
        # Peak normalization to -3dB
        peak_level = np.max(np.abs(audio))
        if peak_level > 0:
            target_peak = 10**(self.vo_standards['peak_level_db'] / 20)
            normalized = audio * (target_peak / peak_level)
        else:
            normalized = audio
        
        return normalized
    
    def apply_voice_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle compression optimized for voice."""
        # Simple soft compression
        threshold = 0.7
        ratio = 3.0
        
        compressed = audio.copy()
        mask = np.abs(compressed) > threshold
        
        # Apply compression above threshold
        excess = np.abs(compressed[mask]) - threshold
        compressed_excess = excess / ratio
        compressed[mask] = np.sign(compressed[mask]) * (threshold + compressed_excess)
        
        return compressed
    
    def apply_highpass_filter(self, audio: np.ndarray, sr: int, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth high-pass filter
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
        filtered = scipy.signal.filtfilt(b, a, audio)
        
        return filtered
    
    def apply_deessing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply de-essing to reduce harsh sibilants."""
        # Focus on sibilant frequency range (4-8 kHz)
        nyquist = sr / 2
        low_freq = 4000 / nyquist
        high_freq = min(8000 / nyquist, 0.99)
        
        # Design band-pass filter for sibilant range
        b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
        sibilant_band = scipy.signal.filtfilt(b, a, audio)
        
        # Apply gentle compression to sibilant band
        threshold = 0.3
        ratio = 4.0
        
        compressed_sibilants = sibilant_band.copy()
        mask = np.abs(compressed_sibilants) > threshold
        excess = np.abs(compressed_sibilants[mask]) - threshold
        compressed_excess = excess / ratio
        compressed_sibilants[mask] = np.sign(compressed_sibilants[mask]) * (threshold + compressed_excess)
        
        # Subtract over-compressed sibilants from original
        deessed = audio - (sibilant_band - compressed_sibilants) * 0.5
        
        return deessed
    
    def detect_voice_activity(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect voice activity segments."""
        # Simple energy-based voice activity detection
        frame_size = 1024
        hop_size = frame_size // 2
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            frame_energy = np.sum(frame**2)
            energy.append(frame_energy)
        
        energy = np.array(energy)
        threshold = np.mean(energy) * 0.1  # Adaptive threshold
        
        # Find voice segments
        voice_frames = energy > threshold
        segments = []
        
        in_voice = False
        start_frame = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_voice:
                start_frame = i
                in_voice = True
            elif not is_voice and in_voice:
                start_sample = start_frame * hop_size
                end_sample = i * hop_size
                segments.append((start_sample, end_sample))
                in_voice = False
        
        # Handle case where voice continues to end
        if in_voice:
            start_sample = start_frame * hop_size
            segments.append((start_sample, len(audio)))
        
        return segments
    
    def detect_silence_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect silence segments."""
        voice_segments = self.detect_voice_activity(audio)
        
        # Find gaps between voice segments
        silence_segments = []
        
        if len(voice_segments) == 0:
            return [(0, len(audio))]
        
        # Silence before first voice segment
        if voice_segments[0][0] > 0:
            silence_segments.append((0, voice_segments[0][0]))
        
        # Silence between voice segments
        for i in range(len(voice_segments) - 1):
            end_current = voice_segments[i][1]
            start_next = voice_segments[i + 1][0]
            if start_next > end_current:
                silence_segments.append((end_current, start_next))
        
        # Silence after last voice segment
        if voice_segments[-1][1] < len(audio):
            silence_segments.append((voice_segments[-1][1], len(audio)))
        
        return silence_segments
    
    def resample_audio(self, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if original_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    def save_audio(self, audio: np.ndarray, sr: int, file_path: str, bit_depth: int = 24):
        """Save audio with specified quality settings."""
        # Ensure audio is in correct range
        if bit_depth == 24:
            # 24-bit audio
            audio_int = (audio * (2**23 - 1)).astype(np.int32)
            sf.write(file_path, audio_int, sr, subtype='PCM_24')
        elif bit_depth == 16:
            # 16-bit audio
            audio_int = (audio * (2**15 - 1)).astype(np.int16)
            sf.write(file_path, audio_int, sr, subtype='PCM_16')
        else:
            # Float32 fallback
            sf.write(file_path, audio, sr, subtype='FLOAT')


def main():
    """Test the audio processor."""
    processor = AudioProcessor()
    
    # Create test audio signal
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate test signal with voice-like characteristics
    fundamental = 150  # Hz
    audio = (np.sin(2 * np.pi * fundamental * t) * 0.5 +
             np.sin(2 * np.pi * fundamental * 2 * t) * 0.3 +
             np.sin(2 * np.pi * fundamental * 3 * t) * 0.2)
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(audio))
    audio_with_noise = audio + noise
    
    print("Testing Audio Processor...")
    
    # Assess quality
    quality = processor.assess_quality(audio_with_noise, sr)
    print(f"Original quality metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    
    # Enhance audio
    enhanced, metadata = processor.enhance_audio(audio_with_noise, sr)
    print(f"\nEnhancements applied: {metadata['enhancements_applied']}")
    
    # Final quality
    final_quality = metadata['final_quality']
    print(f"\nEnhanced quality metrics:")
    for key, value in final_quality.items():
        print(f"  {key}: {value}")
    
    print(f"\nMeets voice over standards: {final_quality['meets_vo_standards']}")


if __name__ == "__main__":
    main()

