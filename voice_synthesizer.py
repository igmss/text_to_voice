"""
Voice Synthesis Interface for Egyptian Arabic TTS

This module provides a user-friendly interface for generating voice overs
using the Egyptian Arabic TTS system with professional quality controls.
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, List, Optional, Tuple, Union
import json
import os
from datetime import datetime
import gradio as gr
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import our TTS components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.egyptian_tts import EgyptianArabicTTS, create_model_config
from preprocessing.text_processor import EgyptianArabicProcessor
from preprocessing.audio_processor import AudioProcessor
from evaluation.metrics import EgyptianTTSEvaluator


class VoiceSynthesizer:
    """
    Main voice synthesis interface for Egyptian Arabic TTS.
    Provides high-level API for voice over generation.
    """
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize voice synthesizer.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
        """
        self.config = config or create_model_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.text_processor = EgyptianArabicProcessor()
        self.audio_processor = AudioProcessor(
            target_sr=48000,  # Voice over quality
            target_bit_depth=24
        )
        self.evaluator = EgyptianTTSEvaluator({'sample_rate': 48000})
        
        # Initialize model
        self.model = EgyptianArabicTTS(self.config).to(self.device)
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No trained model loaded. Using random weights for demonstration.")
        
        # Voice settings presets
        self.voice_presets = self.create_voice_presets()
        
        # Speaker database
        self.speakers = self.load_speaker_database()
        
    def load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def create_voice_presets(self) -> Dict[str, Dict]:
        """Create voice over presets for different use cases."""
        return {
            "Commercial - Energetic": {
                "speed": 1.1,
                "pitch": 1.05,
                "energy": 1.2,
                "emotion": "excited",
                "description": "High-energy commercial voice over"
            },
            "Commercial - Warm": {
                "speed": 0.95,
                "pitch": 0.98,
                "energy": 1.0,
                "emotion": "warm",
                "description": "Warm and friendly commercial voice"
            },
            "Educational - Clear": {
                "speed": 0.9,
                "pitch": 1.0,
                "energy": 0.9,
                "emotion": "neutral",
                "description": "Clear and measured educational delivery"
            },
            "Documentary - Authoritative": {
                "speed": 0.85,
                "pitch": 0.95,
                "energy": 1.1,
                "emotion": "serious",
                "description": "Authoritative documentary narration"
            },
            "Audiobook - Natural": {
                "speed": 1.0,
                "pitch": 1.0,
                "energy": 0.95,
                "emotion": "neutral",
                "description": "Natural storytelling voice"
            },
            "News - Professional": {
                "speed": 1.05,
                "pitch": 1.0,
                "energy": 1.05,
                "emotion": "professional",
                "description": "Professional news delivery"
            }
        }
    
    def load_speaker_database(self) -> Dict[str, Dict]:
        """Load available speaker voices."""
        # In a real implementation, this would load from a speaker database
        return {
            "default": {
                "name": "Default Egyptian Voice",
                "gender": "neutral",
                "age": "adult",
                "description": "Standard Egyptian Arabic voice"
            },
            "male_young": {
                "name": "Ahmed - Young Male",
                "gender": "male",
                "age": "young",
                "description": "Energetic young male voice"
            },
            "female_adult": {
                "name": "Fatima - Adult Female",
                "gender": "female", 
                "age": "adult",
                "description": "Professional adult female voice"
            },
            "male_mature": {
                "name": "Omar - Mature Male",
                "gender": "male",
                "age": "mature",
                "description": "Authoritative mature male voice"
            }
        }
    
    def synthesize_voice_over(self, 
                             text: str,
                             speaker_id: str = "default",
                             voice_preset: str = "Commercial - Warm",
                             custom_settings: Dict = None,
                             output_format: str = "wav") -> Tuple[np.ndarray, int, Dict]:
        """
        Generate voice over from Egyptian Arabic text.
        
        Args:
            text: Egyptian Arabic text to synthesize
            speaker_id: Speaker voice to use
            voice_preset: Voice preset for style
            custom_settings: Custom voice settings (overrides preset)
            output_format: Output audio format
            
        Returns:
            Tuple of (audio_array, sample_rate, metadata)
        """
        # Process text
        processed_text = self.text_processor.process_for_tts(text)
        
        # Get voice settings
        if custom_settings:
            voice_settings = custom_settings
        else:
            voice_settings = self.voice_presets.get(voice_preset, 
                                                   self.voice_presets["Commercial - Warm"])
        
        # Generate audio using model
        try:
            with torch.no_grad():
                # Convert text to model input
                text_input = self.text_to_tensor(processed_text['text'])
                
                # Get speaker embedding
                speaker_embedding = self.get_speaker_embedding(speaker_id)
                
                # Generate audio
                output = self.model(
                    text_input=text_input,
                    speaker_embedding=speaker_embedding,
                    prosody_control=voice_settings
                )
                
                # Extract audio
                audio_tensor = output['audio']
                audio_np = audio_tensor.cpu().numpy().squeeze()
                
        except Exception as e:
            print(f"Model inference error: {e}")
            # Fallback: generate dummy audio for demonstration
            audio_np = self.generate_demo_audio(text, voice_settings)
        
        # Post-process audio for voice over quality
        enhanced_audio, enhancement_metadata = self.audio_processor.enhance_audio(
            audio_np, 48000
        )
        
        # Apply voice over specific processing
        final_audio = self.apply_voice_over_processing(enhanced_audio, voice_settings)
        
        # Generate metadata
        metadata = {
            'text': text,
            'processed_text': processed_text['text'],
            'speaker_id': speaker_id,
            'voice_preset': voice_preset,
            'voice_settings': voice_settings,
            'audio_length_seconds': len(final_audio) / 48000,
            'sample_rate': 48000,
            'bit_depth': 24,
            'enhancement_applied': enhancement_metadata.get('enhancements_applied', []),
            'generation_timestamp': datetime.now().isoformat(),
            'quality_metrics': self.evaluate_output_quality(final_audio)
        }
        
        return final_audio, 48000, metadata
    
    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to model input tensor."""
        # Simplified tokenization for demonstration
        tokens = [ord(c) % 1000 for c in text[:100]]  # Limit length
        tokens = tokens + [0] * (100 - len(tokens))  # Pad to fixed length
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding for voice cloning."""
        # In real implementation, load from speaker database
        embedding_dim = self.config.get('speaker_dim', 256)
        
        # Generate consistent embedding based on speaker_id
        np.random.seed(hash(speaker_id) % 2**32)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        
        return torch.tensor(embedding).unsqueeze(0).to(self.device)
    
    def generate_demo_audio(self, text: str, voice_settings: Dict) -> np.ndarray:
        """Generate demonstration audio when model is not available."""
        # Create synthetic speech-like audio for demonstration
        duration = len(text) * 0.1  # Rough estimate: 0.1 seconds per character
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base frequency influenced by pitch setting
        base_freq = 150 * voice_settings.get('pitch', 1.0)
        
        # Generate speech-like signal with harmonics
        audio = np.zeros_like(t)
        for harmonic in range(1, 6):
            freq = base_freq * harmonic
            amplitude = 0.5 / harmonic * voice_settings.get('energy', 1.0)
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some variation to make it more speech-like
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
        audio *= modulation
        
        # Apply speed setting by resampling
        speed = voice_settings.get('speed', 1.0)
        if speed != 1.0:
            new_length = int(len(audio) / speed)
            audio = np.interp(np.linspace(0, len(audio)-1, new_length), 
                            np.arange(len(audio)), audio)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio
    
    def apply_voice_over_processing(self, audio: np.ndarray, 
                                   voice_settings: Dict) -> np.ndarray:
        """Apply voice over specific audio processing."""
        processed = audio.copy()
        
        # Apply EQ for voice over clarity
        processed = self.apply_voice_over_eq(processed)
        
        # Apply gentle compression for consistency
        processed = self.apply_voice_over_compression(processed, voice_settings)
        
        # Apply de-essing for professional sound
        processed = self.audio_processor.apply_deessing(processed, 48000)
        
        # Final normalization to voice over standards
        processed = self.audio_processor.normalize_levels(processed)
        
        return processed
    
    def apply_voice_over_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply EQ optimized for voice over work."""
        # High-pass filter to remove low-frequency rumble
        audio = self.audio_processor.apply_highpass_filter(audio, 48000, cutoff=80)
        
        # Gentle boost in presence range (2-5 kHz) for clarity
        # This would be implemented with proper EQ filters in production
        return audio
    
    def apply_voice_over_compression(self, audio: np.ndarray, 
                                   voice_settings: Dict) -> np.ndarray:
        """Apply compression optimized for voice over."""
        # Adjust compression based on voice settings
        energy_factor = voice_settings.get('energy', 1.0)
        
        # More compression for higher energy settings
        threshold = 0.6 / energy_factor
        ratio = 2.5 + energy_factor
        
        compressed = audio.copy()
        mask = np.abs(compressed) > threshold
        
        excess = np.abs(compressed[mask]) - threshold
        compressed_excess = excess / ratio
        compressed[mask] = np.sign(compressed[mask]) * (threshold + compressed_excess)
        
        return compressed
    
    def evaluate_output_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of generated audio."""
        quality_metrics = self.audio_processor.assess_quality(audio, 48000)
        
        # Add voice over specific metrics
        vo_quality = self.evaluator.voice_over_evaluator.evaluate(audio)
        quality_metrics['voice_over_quality'] = vo_quality
        
        return quality_metrics
    
    def batch_synthesize(self, texts: List[str], 
                        output_dir: str,
                        speaker_id: str = "default",
                        voice_preset: str = "Commercial - Warm") -> List[str]:
        """
        Batch synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Directory to save audio files
            speaker_id: Speaker voice to use
            voice_preset: Voice preset for style
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, text in enumerate(texts):
            # Generate audio
            audio, sr, metadata = self.synthesize_voice_over(
                text, speaker_id, voice_preset
            )
            
            # Save audio file
            output_file = os.path.join(output_dir, f"voice_over_{i+1:03d}.wav")
            self.save_audio(audio, sr, output_file)
            
            # Save metadata
            metadata_file = os.path.join(output_dir, f"voice_over_{i+1:03d}_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            output_files.append(output_file)
        
        return output_files
    
    def save_audio(self, audio: np.ndarray, sample_rate: int, 
                   output_path: str, format: str = "wav"):
        """Save audio to file with voice over quality settings."""
        if format.lower() == "wav":
            # Save as 24-bit WAV for voice over quality
            audio_24bit = (audio * (2**23 - 1)).astype(np.int32)
            sf.write(output_path, audio_24bit, sample_rate, subtype='PCM_24')
        else:
            # Use soundfile default
            sf.write(output_path, audio, sample_rate)
    
    def create_voice_sample(self, speaker_id: str, 
                           sample_text: str = "مرحبا، هذا مثال على الصوت المصري") -> Tuple[np.ndarray, int]:
        """Create a voice sample for speaker preview."""
        audio, sr, _ = self.synthesize_voice_over(
            sample_text, 
            speaker_id=speaker_id,
            voice_preset="Commercial - Warm"
        )
        return audio, sr


class VoiceOverStudio:
    """
    Complete voice over studio interface with Gradio UI.
    Provides professional tools for Egyptian Arabic voice over production.
    """
    
    def __init__(self, synthesizer: VoiceSynthesizer):
        """Initialize voice over studio."""
        self.synthesizer = synthesizer
        self.temp_dir = tempfile.mkdtemp()
        
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface for voice over studio."""
        
        def generate_voice_over(text, speaker_id, voice_preset, 
                              speed, pitch, energy, custom_emotion):
            """Generate voice over with custom settings."""
            if not text.strip():
                return None, "Please enter text to synthesize."
            
            # Create custom settings
            custom_settings = {
                "speed": speed,
                "pitch": pitch, 
                "energy": energy,
                "emotion": custom_emotion
            }
            
            try:
                # Generate audio
                audio, sr, metadata = self.synthesizer.synthesize_voice_over(
                    text=text,
                    speaker_id=speaker_id,
                    voice_preset=voice_preset,
                    custom_settings=custom_settings
                )
                
                # Save to temporary file
                temp_file = os.path.join(self.temp_dir, f"voice_over_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                self.synthesizer.save_audio(audio, sr, temp_file)
                
                # Create info text
                info = f"""
                **Generated Voice Over**
                - Duration: {metadata['audio_length_seconds']:.2f} seconds
                - Speaker: {self.synthesizer.speakers[speaker_id]['name']}
                - Preset: {voice_preset}
                - Quality Score: {metadata['quality_metrics'].get('voice_over_quality', 0):.3f}
                - Sample Rate: {sr} Hz
                """
                
                return temp_file, info
                
            except Exception as e:
                return None, f"Error generating voice over: {str(e)}"
        
        def preview_speaker(speaker_id):
            """Preview speaker voice."""
            try:
                audio, sr = self.synthesizer.create_voice_sample(speaker_id)
                temp_file = os.path.join(self.temp_dir, f"speaker_preview_{speaker_id}.wav")
                self.synthesizer.save_audio(audio, sr, temp_file)
                return temp_file
            except Exception as e:
                return None
        
        # Create interface
        with gr.Blocks(title="Egyptian Arabic Voice Over Studio", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 🎙️ Egyptian Arabic Voice Over Studio")
            gr.Markdown("Professional voice over generation for Egyptian Arabic content")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Text input
                    text_input = gr.Textbox(
                        label="Egyptian Arabic Text",
                        placeholder="أدخل النص المصري هنا...",
                        lines=5,
                        max_lines=10
                    )
                    
                    # Speaker selection
                    speaker_dropdown = gr.Dropdown(
                        choices=list(self.synthesizer.speakers.keys()),
                        value="default",
                        label="Speaker Voice",
                        info="Choose the voice character"
                    )
                    
                    # Speaker preview
                    preview_btn = gr.Button("🔊 Preview Speaker", size="sm")
                    speaker_preview = gr.Audio(label="Speaker Preview", visible=False)
                    
                    # Voice preset
                    preset_dropdown = gr.Dropdown(
                        choices=list(self.synthesizer.voice_presets.keys()),
                        value="Commercial - Warm",
                        label="Voice Preset",
                        info="Choose a preset style"
                    )
                    
                with gr.Column(scale=1):
                    # Custom controls
                    gr.Markdown("### Custom Voice Controls")
                    
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed",
                        info="Speaking rate"
                    )
                    
                    pitch_slider = gr.Slider(
                        minimum=0.7,
                        maximum=1.3,
                        value=1.0,
                        step=0.05,
                        label="Pitch",
                        info="Voice pitch"
                    )
                    
                    energy_slider = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Energy",
                        info="Voice energy/intensity"
                    )
                    
                    emotion_dropdown = gr.Dropdown(
                        choices=["neutral", "warm", "excited", "serious", "professional"],
                        value="neutral",
                        label="Emotion",
                        info="Emotional tone"
                    )
            
            # Generate button
            generate_btn = gr.Button("🎵 Generate Voice Over", variant="primary", size="lg")
            
            # Output
            with gr.Row():
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Generated Voice Over",
                        type="filepath"
                    )
                    
                with gr.Column():
                    output_info = gr.Markdown(label="Generation Info")
            
            # Event handlers
            preview_btn.click(
                fn=preview_speaker,
                inputs=[speaker_dropdown],
                outputs=[speaker_preview]
            ).then(
                lambda: gr.Audio(visible=True),
                outputs=[speaker_preview]
            )
            
            generate_btn.click(
                fn=generate_voice_over,
                inputs=[
                    text_input, speaker_dropdown, preset_dropdown,
                    speed_slider, pitch_slider, energy_slider, emotion_dropdown
                ],
                outputs=[output_audio, output_info]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["مرحبا بكم في برنامجنا الجديد", "default", "Commercial - Energetic"],
                    ["هذا درس تعليمي مهم جداً", "female_adult", "Educational - Clear"],
                    ["في هذا التقرير الإخباري", "male_mature", "News - Professional"],
                    ["كان يا ما كان في قديم الزمان", "default", "Audiobook - Natural"]
                ],
                inputs=[text_input, speaker_dropdown, preset_dropdown]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the voice over studio interface."""
        interface = self.create_interface()
        return interface.launch(**kwargs)


def main():
    """Main function to run the voice synthesis interface."""
    print("Initializing Egyptian Arabic Voice Over Studio...")
    
    # Initialize synthesizer
    synthesizer = VoiceSynthesizer()
    
    # Create studio interface
    studio = VoiceOverStudio(synthesizer)
    
    # Launch interface
    print("Launching Voice Over Studio...")
    studio.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()

