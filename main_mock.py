import os
import sys
import json
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import soundfile as sf

# Import our mock TTS system components for testing
from mock_voice_synthesizer import VoiceSynthesizer
from mock_evaluator import EgyptianTTSEvaluator
from mock_audio_processor import AudioProcessor

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'egyptian_tts_secret_key_2024'

# Enable CORS for frontend communication
CORS(app, origins=['*'])

# Initialize TTS system
try:
    synthesizer = VoiceSynthesizer()
    evaluator = EgyptianTTSEvaluator({'sample_rate': 48000})
    audio_processor = AudioProcessor(target_sr=48000, target_bit_depth=24)
    print("✅ Mock TTS system initialized successfully")
except Exception as e:
    print(f"⚠️ TTS system initialization failed: {e}")
    synthesizer = None
    evaluator = None
    audio_processor = None

# Temporary storage for generated audio files
TEMP_AUDIO_DIR = tempfile.mkdtemp()
print(f"📁 Temporary audio directory: {TEMP_AUDIO_DIR}")

# Voice presets configuration
VOICE_PRESETS = {
    'commercial-energetic': {
        'name': 'Commercial - Energetic',
        'description': 'High-energy commercial voice over',
        'settings': {'speed': 1.1, 'pitch': 1.05, 'energy': 1.2}
    },
    'commercial-warm': {
        'name': 'Commercial - Warm',
        'description': 'Warm and friendly commercial voice',
        'settings': {'speed': 0.95, 'pitch': 0.98, 'energy': 1.0}
    },
    'educational-clear': {
        'name': 'Educational - Clear',
        'description': 'Clear and measured educational delivery',
        'settings': {'speed': 0.9, 'pitch': 1.0, 'energy': 0.9}
    },
    'documentary-authoritative': {
        'name': 'Documentary - Authoritative',
        'description': 'Authoritative documentary narration',
        'settings': {'speed': 0.85, 'pitch': 0.95, 'energy': 1.1}
    },
    'audiobook-natural': {
        'name': 'Audiobook - Natural',
        'description': 'Natural storytelling voice',
        'settings': {'speed': 1.0, 'pitch': 1.0, 'energy': 0.95}
    },
    'news-professional': {
        'name': 'News - Professional',
        'description': 'Professional news delivery',
        'settings': {'speed': 1.05, 'pitch': 1.0, 'energy': 1.05}
    }
}

# Speaker voices
SPEAKERS = {
    'default': {'name': 'Default Egyptian Voice', 'gender': 'neutral', 'age': 'adult'},
    'male-young': {'name': 'Ahmed - Young Male', 'gender': 'male', 'age': 'young'},
    'female-adult': {'name': 'Fatima - Adult Female', 'gender': 'female', 'age': 'adult'},
    'male-mature': {'name': 'Omar - Mature Male', 'gender': 'male', 'age': 'mature'}
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tts_system': 'available' if synthesizer else 'unavailable',
        'version': '1.0.0-mock'
    })

@app.route('/api/presets', methods=['GET'])
def get_voice_presets():
    """Get available voice presets."""
    return jsonify({
        'presets': VOICE_PRESETS,
        'count': len(VOICE_PRESETS)
    })

@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """Get available speaker voices."""
    return jsonify({
        'speakers': SPEAKERS,
        'count': len(SPEAKERS)
    })

@app.route('/api/generate', methods=['POST'])
def generate_voice_over():
    """Generate voice over from Egyptian Arabic text."""
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        speaker_id = data.get('speaker_id', 'default')
        voice_preset = data.get('voice_preset', 'commercial-warm')
        custom_settings = data.get('custom_settings', {})
        
        # Validate inputs
        if speaker_id not in SPEAKERS:
            return jsonify({'error': f'Invalid speaker_id: {speaker_id}'}), 400
        
        if voice_preset not in VOICE_PRESETS:
            return jsonify({'error': f'Invalid voice_preset: {voice_preset}'}), 400
        
        # Check if TTS system is available
        if not synthesizer:
            return jsonify({'error': 'TTS system not available'}), 503
        
        # Generate voice over
        try:
            audio, sample_rate, metadata = synthesizer.synthesize_voice_over(
                text=text,
                speaker_id=speaker_id,
                voice_preset=voice_preset,
                custom_settings=custom_settings if custom_settings else None
            )
            
            # Generate unique filename
            audio_id = str(uuid.uuid4())
            audio_filename = f"voice_over_{audio_id}.wav"
            audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
            
            # Save audio file
            synthesizer.save_audio(audio, sample_rate, audio_path)
            
            # Prepare response
            response_data = {
                'audio_id': audio_id,
                'audio_url': f'/api/audio/{audio_id}',
                'metadata': {
                    'text': text,
                    'speaker': SPEAKERS[speaker_id]['name'],
                    'preset': VOICE_PRESETS[voice_preset]['name'],
                    'duration': len(audio) / sample_rate,
                    'sample_rate': sample_rate,
                    'quality_score': metadata.get('quality_metrics', {}).get('voice_over_quality', 0.8),
                    'timestamp': datetime.now().isoformat()
                },
                'quality_metrics': metadata.get('quality_metrics', {}),
                'generation_settings': {
                    'speaker_id': speaker_id,
                    'voice_preset': voice_preset,
                    'custom_settings': custom_settings
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Generation error: {e}")
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
    
    except Exception as e:
        print(f"Request processing error: {e}")
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_audio_file(audio_id):
    """Serve generated audio file."""
    try:
        audio_filename = f"voice_over_{audio_id}.wav"
        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=audio_filename
        )
    
    except Exception as e:
        print(f"Audio serving error: {e}")
        return jsonify({'error': f'Failed to serve audio: {str(e)}'}), 500

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information and capabilities."""
    return jsonify({
        'system': 'Egyptian Arabic TTS Voice Over System (Mock)',
        'version': '1.0.0-mock',
        'capabilities': {
            'voice_generation': synthesizer is not None,
            'quality_evaluation': evaluator is not None,
            'audio_processing': audio_processor is not None,
            'batch_processing': True,
            'real_time_generation': True
        },
        'supported_formats': ['wav'],
        'sample_rates': [22050, 44100, 48000],
        'languages': ['Egyptian Arabic'],
        'voice_presets': list(VOICE_PRESETS.keys()),
        'speakers': list(SPEAKERS.keys()),
        'quality_standards': {
            'sample_rate': 48000,
            'bit_depth': 24,
            'voice_over_optimized': True
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve static files."""
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return jsonify({
                'message': 'Egyptian Arabic TTS Voice Over API (Mock Version)',
                'version': '1.0.0-mock',
                'endpoints': {
                    'health': '/api/health',
                    'generate': '/api/generate',
                    'presets': '/api/presets',
                    'speakers': '/api/speakers',
                    'system': '/api/system-info'
                }
            })

if __name__ == '__main__':
    print("🎙️ Starting Egyptian Arabic TTS Voice Over API Server (Mock Version)")
    print(f"📁 Temporary audio directory: {TEMP_AUDIO_DIR}")
    print("🌐 Server will be available at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

