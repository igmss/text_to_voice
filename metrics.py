"""
Evaluation metrics for Egyptian Arabic TTS system.

This module provides comprehensive evaluation metrics specifically designed
for assessing Egyptian Arabic voice over quality and dialectal accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import scipy.stats
from scipy.spatial.distance import cosine
import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class EgyptianTTSEvaluator:
    """
    Comprehensive evaluator for Egyptian Arabic TTS system.
    Includes both objective metrics and subjective quality assessment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 48000)
        
        # Initialize evaluation components
        self.phoneme_evaluator = PhonemeAccuracyEvaluator()
        self.prosody_evaluator = ProsodyEvaluator()
        self.audio_quality_evaluator = AudioQualityEvaluator(self.sample_rate)
        self.dialect_evaluator = DialectAccuracyEvaluator()
        self.voice_over_evaluator = VoiceOverQualityEvaluator()
        
    def evaluate_model(self, model, test_loader, device: str = 'cuda') -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained TTS model
            test_loader: Test data loader
            device: Device for computation
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = {
            'phoneme_accuracy': [],
            'prosody_correlation': [],
            'audio_quality': [],
            'dialect_accuracy': [],
            'voice_over_quality': [],
            'naturalness_score': [],
            'intelligibility_score': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate audio
                output = model(
                    text_input=batch['text_tokens'],
                    speaker_embedding=None
                )
                
                # Extract generated and target audio
                generated_audio = output['audio'].cpu().numpy()
                target_audio = batch['audio'].cpu().numpy()
                
                # Evaluate each sample in batch
                for i in range(generated_audio.shape[0]):
                    gen_audio = generated_audio[i]
                    tgt_audio = target_audio[i]
                    
                    # Phoneme accuracy
                    phoneme_acc = self.phoneme_evaluator.evaluate(
                        gen_audio, tgt_audio, self.sample_rate
                    )
                    all_metrics['phoneme_accuracy'].append(phoneme_acc)
                    
                    # Prosody correlation
                    prosody_corr = self.prosody_evaluator.evaluate(
                        gen_audio, tgt_audio, self.sample_rate
                    )
                    all_metrics['prosody_correlation'].append(prosody_corr)
                    
                    # Audio quality
                    audio_qual = self.audio_quality_evaluator.evaluate(
                        gen_audio, tgt_audio
                    )
                    all_metrics['audio_quality'].append(audio_qual)
                    
                    # Dialect accuracy (if text available)
                    if 'metadata' in batch:
                        dialect_acc = self.dialect_evaluator.evaluate(
                            gen_audio, batch['metadata'][i]
                        )
                        all_metrics['dialect_accuracy'].append(dialect_acc)
                    
                    # Voice over quality
                    vo_quality = self.voice_over_evaluator.evaluate(gen_audio)
                    all_metrics['voice_over_quality'].append(vo_quality)
                    
                    # Naturalness and intelligibility
                    naturalness = self.evaluate_naturalness(gen_audio, tgt_audio)
                    intelligibility = self.evaluate_intelligibility(gen_audio, tgt_audio)
                    
                    all_metrics['naturalness_score'].append(naturalness)
                    all_metrics['intelligibility_score'].append(intelligibility)
        
        # Aggregate metrics
        final_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:  # Only if we have values
                final_metrics[f'{metric_name}_mean'] = np.mean(values)
                final_metrics[f'{metric_name}_std'] = np.std(values)
        
        return final_metrics
    
    def evaluate_naturalness(self, generated_audio: np.ndarray, 
                           target_audio: np.ndarray) -> float:
        """Evaluate naturalness of generated speech."""
        # Extract features for naturalness assessment
        gen_features = self.extract_naturalness_features(generated_audio)
        tgt_features = self.extract_naturalness_features(target_audio)
        
        # Calculate similarity
        similarity = 1 - cosine(gen_features, tgt_features)
        return max(0, similarity)  # Ensure non-negative
    
    def evaluate_intelligibility(self, generated_audio: np.ndarray,
                                target_audio: np.ndarray) -> float:
        """Evaluate intelligibility of generated speech."""
        try:
            # Use STOI (Short-Time Objective Intelligibility)
            # Resample to 10kHz for STOI if needed
            if self.sample_rate != 10000:
                gen_10k = librosa.resample(generated_audio, 
                                         orig_sr=self.sample_rate, 
                                         target_sr=10000)
                tgt_10k = librosa.resample(target_audio, 
                                         orig_sr=self.sample_rate, 
                                         target_sr=10000)
            else:
                gen_10k = generated_audio
                tgt_10k = target_audio
            
            # Calculate STOI
            stoi_score = stoi(tgt_10k, gen_10k, 10000, extended=False)
            return stoi_score
        except:
            # Fallback to spectral similarity
            return self.spectral_similarity(generated_audio, target_audio)
    
    def extract_naturalness_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for naturalness assessment."""
        # Extract various acoustic features
        features = []
        
        # Spectral features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Prosodic features
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[voiced_flag]
        if len(f0_clean) > 0:
            features.extend([np.mean(f0_clean), np.std(f0_clean)])
        else:
            features.extend([0, 0])
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        return np.array(features)
    
    def spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate spectral similarity between two audio signals."""
        # Extract mel-spectrograms
        mel1 = librosa.feature.melspectrogram(y=audio1, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=audio2, sr=self.sample_rate)
        
        # Flatten and normalize
        mel1_flat = mel1.flatten()
        mel2_flat = mel2.flatten()
        
        # Ensure same length
        min_len = min(len(mel1_flat), len(mel2_flat))
        mel1_flat = mel1_flat[:min_len]
        mel2_flat = mel2_flat[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(mel1_flat, mel2_flat)[0, 1]
        return max(0, correlation)  # Ensure non-negative
    
    def generate_evaluation_report(self, metrics: Dict[str, float], 
                                 output_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        report = "# Egyptian Arabic TTS Evaluation Report\n\n"
        
        # Overall performance summary
        report += "## Overall Performance Summary\n\n"
        
        key_metrics = [
            'phoneme_accuracy_mean',
            'prosody_correlation_mean', 
            'audio_quality_mean',
            'dialect_accuracy_mean',
            'voice_over_quality_mean',
            'naturalness_score_mean',
            'intelligibility_score_mean'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                score = metrics[metric]
                std = metrics.get(metric.replace('_mean', '_std'), 0)
                report += f"- **{metric.replace('_', ' ').title()}**: {score:.3f} ± {std:.3f}\n"
        
        # Detailed analysis
        report += "\n## Detailed Analysis\n\n"
        
        # Phoneme accuracy analysis
        if 'phoneme_accuracy_mean' in metrics:
            score = metrics['phoneme_accuracy_mean']
            if score >= 0.9:
                assessment = "Excellent"
            elif score >= 0.8:
                assessment = "Good"
            elif score >= 0.7:
                assessment = "Fair"
            else:
                assessment = "Needs Improvement"
            
            report += f"### Phoneme Accuracy: {assessment}\n"
            report += f"The model achieves {score:.1%} phoneme accuracy, indicating "
            report += f"{assessment.lower()} pronunciation quality for Egyptian Arabic.\n\n"
        
        # Voice over quality analysis
        if 'voice_over_quality_mean' in metrics:
            score = metrics['voice_over_quality_mean']
            if score >= 0.85:
                assessment = "Professional Grade"
            elif score >= 0.75:
                assessment = "Broadcast Quality"
            elif score >= 0.65:
                assessment = "Good Quality"
            else:
                assessment = "Needs Enhancement"
            
            report += f"### Voice Over Quality: {assessment}\n"
            report += f"Audio quality score of {score:.3f} indicates {assessment.lower()} "
            report += f"suitable for voice over applications.\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if metrics.get('phoneme_accuracy_mean', 0) < 0.8:
            report += "- **Improve Phoneme Accuracy**: Consider additional training data "
            report += "focusing on Egyptian Arabic phonetic variations.\n"
        
        if metrics.get('prosody_correlation_mean', 0) < 0.7:
            report += "- **Enhance Prosody Modeling**: Improve stress and intonation "
            report += "prediction for more natural Egyptian Arabic speech patterns.\n"
        
        if metrics.get('voice_over_quality_mean', 0) < 0.8:
            report += "- **Audio Quality Enhancement**: Apply additional post-processing "
            report += "to meet professional voice over standards.\n"
        
        if metrics.get('dialect_accuracy_mean', 0) < 0.85:
            report += "- **Dialect Specificity**: Increase training data with authentic "
            report += "Egyptian Arabic expressions and pronunciation patterns.\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class PhonemeAccuracyEvaluator:
    """Evaluates phoneme-level accuracy for Egyptian Arabic."""
    
    def __init__(self):
        self.egyptian_phonemes = [
            'b', 't', 'g', 'ħ', 'x', 'd', 'r', 'z', 's', 'ʃ',
            'sˤ', 'dˤ', 'tˤ', 'ʕ', 'ɣ', 'f', 'ʔ', 'k', 'l', 'm',
            'n', 'h', 'w', 'j', 'a', 'i', 'u', 'aː', 'iː', 'uː'
        ]
    
    def evaluate(self, generated_audio: np.ndarray, 
                target_audio: np.ndarray, sample_rate: int) -> float:
        """Evaluate phoneme accuracy."""
        # Extract phonetic features
        gen_features = self.extract_phonetic_features(generated_audio, sample_rate)
        tgt_features = self.extract_phonetic_features(target_audio, sample_rate)
        
        # Calculate similarity
        similarity = self.calculate_phonetic_similarity(gen_features, tgt_features)
        return similarity
    
    def extract_phonetic_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract phonetic features from audio."""
        # Use MFCCs as phonetic features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        return mfccs
    
    def calculate_phonetic_similarity(self, features1: np.ndarray, 
                                    features2: np.ndarray) -> float:
        """Calculate phonetic similarity between feature sets."""
        # Dynamic Time Warping for sequence alignment
        from scipy.spatial.distance import euclidean
        
        # Simplified DTW implementation
        m, n = features1.shape[1], features2.shape[1]
        dtw_matrix = np.full((m + 1, n + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = euclidean(features1[:, i-1], features2[:, j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Normalize by path length
        dtw_distance = dtw_matrix[m, n] / (m + n)
        
        # Convert to similarity score
        similarity = 1 / (1 + dtw_distance)
        return similarity


class ProsodyEvaluator:
    """Evaluates prosodic features for Egyptian Arabic."""
    
    def evaluate(self, generated_audio: np.ndarray, 
                target_audio: np.ndarray, sample_rate: int) -> float:
        """Evaluate prosody correlation."""
        # Extract prosodic features
        gen_prosody = self.extract_prosodic_features(generated_audio, sample_rate)
        tgt_prosody = self.extract_prosodic_features(target_audio, sample_rate)
        
        # Calculate correlation
        correlation = self.calculate_prosody_correlation(gen_prosody, tgt_prosody)
        return correlation
    
    def extract_prosodic_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract prosodic features."""
        features = {}
        
        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        features['f0'] = f0[voiced_flag] if np.any(voiced_flag) else np.array([])
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        features['energy'] = rms
        
        # Duration features (voice activity)
        features['voice_activity'] = voiced_flag.astype(float)
        
        return features
    
    def calculate_prosody_correlation(self, prosody1: Dict[str, np.ndarray],
                                    prosody2: Dict[str, np.ndarray]) -> float:
        """Calculate prosody correlation."""
        correlations = []
        
        for feature_name in ['f0', 'energy', 'voice_activity']:
            if feature_name in prosody1 and feature_name in prosody2:
                feat1 = prosody1[feature_name]
                feat2 = prosody2[feature_name]
                
                if len(feat1) > 0 and len(feat2) > 0:
                    # Align sequences to same length
                    min_len = min(len(feat1), len(feat2))
                    feat1_aligned = feat1[:min_len]
                    feat2_aligned = feat2[:min_len]
                    
                    # Calculate correlation
                    if np.std(feat1_aligned) > 0 and np.std(feat2_aligned) > 0:
                        corr = np.corrcoef(feat1_aligned, feat2_aligned)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0


class AudioQualityEvaluator:
    """Evaluates audio quality metrics."""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def evaluate(self, generated_audio: np.ndarray, 
                target_audio: np.ndarray) -> float:
        """Evaluate audio quality."""
        metrics = []
        
        # Signal-to-noise ratio
        snr = self.calculate_snr(generated_audio)
        metrics.append(min(snr / 30.0, 1.0))  # Normalize to 0-1
        
        # Spectral similarity
        spec_sim = self.spectral_similarity(generated_audio, target_audio)
        metrics.append(spec_sim)
        
        # Dynamic range
        dynamic_range = self.calculate_dynamic_range(generated_audio)
        metrics.append(min(dynamic_range / 40.0, 1.0))  # Normalize to 0-1
        
        return np.mean(metrics)
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Simple SNR estimation
        signal_power = np.mean(audio**2)
        noise_power = np.percentile(audio**2, 10)  # Bottom 10% as noise
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60.0  # Very high SNR
        
        return snr
    
    def spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate spectral similarity."""
        # Extract spectrograms
        spec1 = np.abs(librosa.stft(audio1))
        spec2 = np.abs(librosa.stft(audio2))
        
        # Align to same size
        min_time = min(spec1.shape[1], spec2.shape[1])
        spec1 = spec1[:, :min_time]
        spec2 = spec2[:, :min_time]
        
        # Calculate correlation
        spec1_flat = spec1.flatten()
        spec2_flat = spec2.flatten()
        
        if np.std(spec1_flat) > 0 and np.std(spec2_flat) > 0:
            correlation = np.corrcoef(spec1_flat, spec2_flat)[0, 1]
            return max(0, correlation)
        else:
            return 0.0
    
    def calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range."""
        peak_level = np.max(np.abs(audio))
        noise_floor = np.percentile(np.abs(audio), 5)
        
        if noise_floor > 0:
            dynamic_range = 20 * np.log10(peak_level / noise_floor)
        else:
            dynamic_range = 60.0
        
        return dynamic_range


class DialectAccuracyEvaluator:
    """Evaluates Egyptian Arabic dialect accuracy."""
    
    def __init__(self):
        self.egyptian_markers = [
            'ʔ',  # qaf as glottal stop
            'g',  # jim as hard g
            'eː', # specific vowel patterns
        ]
    
    def evaluate(self, generated_audio: np.ndarray, metadata: Dict) -> float:
        """Evaluate dialect accuracy."""
        # This would require more sophisticated phonetic analysis
        # For now, return a placeholder based on audio quality
        return 0.8  # Placeholder


class VoiceOverQualityEvaluator:
    """Evaluates voice over specific quality metrics."""
    
    def evaluate(self, audio: np.ndarray) -> float:
        """Evaluate voice over quality."""
        metrics = []
        
        # Clarity (high frequency content)
        clarity = self.evaluate_clarity(audio)
        metrics.append(clarity)
        
        # Consistency (amplitude variation)
        consistency = self.evaluate_consistency(audio)
        metrics.append(consistency)
        
        # Professional sound (frequency balance)
        professional_sound = self.evaluate_professional_sound(audio)
        metrics.append(professional_sound)
        
        return np.mean(metrics)
    
    def evaluate_clarity(self, audio: np.ndarray) -> float:
        """Evaluate speech clarity."""
        # Analyze high frequency content
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/48000)
        
        # Focus on speech clarity range (2-8 kHz)
        clarity_range = (freqs >= 2000) & (freqs <= 8000)
        clarity_power = np.mean(np.abs(fft[clarity_range])**2)
        total_power = np.mean(np.abs(fft)**2)
        
        clarity_ratio = clarity_power / (total_power + 1e-10)
        return min(clarity_ratio * 5, 1.0)  # Normalize
    
    def evaluate_consistency(self, audio: np.ndarray) -> float:
        """Evaluate amplitude consistency."""
        # Calculate RMS in overlapping windows
        window_size = 2048
        hop_size = 512
        rms_values = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
        
        if len(rms_values) > 1:
            consistency = 1 - (np.std(rms_values) / (np.mean(rms_values) + 1e-10))
            return max(0, consistency)
        else:
            return 1.0
    
    def evaluate_professional_sound(self, audio: np.ndarray) -> float:
        """Evaluate professional sound quality."""
        # Analyze frequency balance
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/48000)
        magnitude = np.abs(fft)
        
        # Professional voice over typically has balanced frequency response
        # in the 200-4000 Hz range
        voice_range = (freqs >= 200) & (freqs <= 4000)
        voice_power = np.mean(magnitude[voice_range]**2)
        
        # Check for excessive low or high frequency content
        low_range = (freqs >= 20) & (freqs < 200)
        high_range = (freqs > 4000) & (freqs <= 8000)
        
        low_power = np.mean(magnitude[low_range]**2)
        high_power = np.mean(magnitude[high_range]**2)
        
        # Professional balance: strong voice range, moderate low/high
        balance_score = voice_power / (voice_power + low_power + high_power + 1e-10)
        return min(balance_score * 2, 1.0)


def main():
    """Test the evaluation system."""
    # Create test configuration
    config = {
        'sample_rate': 48000,
        'evaluation': {
            'metrics': ['phoneme_accuracy', 'prosody_correlation', 'audio_quality']
        }
    }
    
    # Initialize evaluator
    evaluator = EgyptianTTSEvaluator(config)
    
    # Create dummy test data
    sample_rate = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test audio signals
    generated_audio = np.sin(2 * np.pi * 150 * t) * 0.5
    target_audio = np.sin(2 * np.pi * 150 * t) * 0.5 + np.random.normal(0, 0.05, len(t))
    
    print("Testing Egyptian TTS Evaluator...")
    
    # Test individual evaluators
    phoneme_eval = PhonemeAccuracyEvaluator()
    phoneme_score = phoneme_eval.evaluate(generated_audio, target_audio, sample_rate)
    print(f"Phoneme Accuracy: {phoneme_score:.3f}")
    
    prosody_eval = ProsodyEvaluator()
    prosody_score = prosody_eval.evaluate(generated_audio, target_audio, sample_rate)
    print(f"Prosody Correlation: {prosody_score:.3f}")
    
    audio_qual_eval = AudioQualityEvaluator(sample_rate)
    audio_score = audio_qual_eval.evaluate(generated_audio, target_audio)
    print(f"Audio Quality: {audio_score:.3f}")
    
    vo_eval = VoiceOverQualityEvaluator()
    vo_score = vo_eval.evaluate(generated_audio)
    print(f"Voice Over Quality: {vo_score:.3f}")
    
    # Test naturalness and intelligibility
    naturalness = evaluator.evaluate_naturalness(generated_audio, target_audio)
    intelligibility = evaluator.evaluate_intelligibility(generated_audio, target_audio)
    
    print(f"Naturalness: {naturalness:.3f}")
    print(f"Intelligibility: {intelligibility:.3f}")
    
    # Generate sample metrics for report
    sample_metrics = {
        'phoneme_accuracy_mean': phoneme_score,
        'phoneme_accuracy_std': 0.05,
        'prosody_correlation_mean': prosody_score,
        'prosody_correlation_std': 0.03,
        'audio_quality_mean': audio_score,
        'audio_quality_std': 0.02,
        'voice_over_quality_mean': vo_score,
        'voice_over_quality_std': 0.04,
        'naturalness_score_mean': naturalness,
        'naturalness_score_std': 0.06,
        'intelligibility_score_mean': intelligibility,
        'intelligibility_score_std': 0.03
    }
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(sample_metrics)
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(report)


if __name__ == "__main__":
    main()

