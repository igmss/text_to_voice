import os
import sys
import json
import tempfile
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.exceptions import BadRequest

# Import TTS components
from src.enhanced_voice_synthesizer import EnhancedVoiceSynthesizer
from src.mock_audio_processor import MockAudioProcessor
from src.mock_evaluator import MockTTSEvaluator

# Create blueprint
tts_bp = Blueprint('tts', __name__)

# Global variables for TTS system
voice_synthesizer = None
audio_processor = None
tts_evaluator = None
temp_audio_dir = None

def initialize_tts_system():
    """Initialize the TTS system components"""
    global voice_synthesizer, audio_processor, tts_evaluator, temp_audio_dir
    
    try:
        # Create temporary directory for audio files
        temp_audio_dir = tempfile.mkdtemp(prefix='tts_audio_')
        
        # Initialize components
        voice_synthesizer = EnhancedVoiceSynthesizer()
        audio_processor = MockAudioProcessor()
        tts_evaluator = MockTTSEvaluator()
        
        print("✅ Enhanced TTS system initialized successfully")
        print(f"📁 Temporary audio directory: {temp_audio_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize TTS system: {e}")
        return False

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
        'settings': {'speed': 0.9, 'pitch': 1.0, 'energy': 0.8}
    },
    'documentary-authoritative': {
        'name': 'Documentary - Authoritative',
        'description': 'Authoritative documentary narration',
        'settings': {'speed': 0.85, 'pitch': 0.95, 'energy': 0.9}
    },
    'audiobook-natural': {
        'name': 'Audiobook - Natural',
        'description': 'Natural storytelling voice',
        'settings': {'speed': 0.9, 'pitch': 1.0, 'energy': 0.7}
    },
    'news-professional': {
        'name': 'News - Professional',
        'description': 'Professional news delivery',
        'settings': {'speed': 1.0, 'pitch': 1.0, 'energy': 1.0}
    }
}

# Speaker voices configuration
SPEAKERS = {
    'default': {
        'name': 'Default Egyptian Voice',
        'gender': 'Mixed',
        'age': 'Adult',
        'description': 'Standard Egyptian Arabic voice'
    },
    'male-young': {
        'name': 'Ahmed',
        'gender': 'Male',
        'age': 'Young Adult',
        'description': 'Energetic young male voice'
    },
    'female-adult': {
        'name': 'Fatima',
        'gender': 'Female',
        'age': 'Adult',
        'description': 'Professional female voice'
    },
    'male-mature': {
        'name': 'Omar',
        'gender': 'Male',
        'age': 'Mature',
        'description': 'Authoritative mature male voice'
    }
}

@tts_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = "healthy" if voice_synthesizer else "initializing"
        
        return jsonify({
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'tts_system': 'Enhanced Egyptian Arabic TTS',
            'features': {
                'espeak_available': voice_synthesizer.espeak_available if voice_synthesizer else False,
                'arabic_support': True,
                'voice_presets': len(VOICE_PRESETS),
                'speakers': len(SPEAKERS)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get detailed system information"""
    try:
        return jsonify({
            'system': 'Enhanced Egyptian Arabic TTS System',
            'version': '2.0.0',
            'languages': ['Arabic (Egyptian)', 'English'],
            'capabilities': {
                'text_to_speech': True,
                'arabic_processing': True,
                'voice_presets': True,
                'quality_evaluation': True,
                'batch_processing': True,
                'espeak_integration': voice_synthesizer.espeak_available if voice_synthesizer else False
            },
            'voice_presets': list(VOICE_PRESETS.keys()),
            'speakers': list(SPEAKERS.keys()),
            'sample_rates': [22050, 44100],
            'audio_formats': ['wav'],
            'temp_directory': temp_audio_dir
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/presets', methods=['GET'])
def get_voice_presets():
    """Get available voice presets"""
    try:
        return jsonify({
            'presets': VOICE_PRESETS,
            'count': len(VOICE_PRESETS)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/speakers', methods=['GET'])
def get_speakers():
    """Get available speaker voices"""
    try:
        return jsonify({
            'speakers': SPEAKERS,
            'count': len(SPEAKERS)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/generate', methods=['POST'])
def generate_voice():
    """Generate voice over from text"""
    try:
        if not voice_synthesizer:
            return jsonify({'error': 'TTS system not initialized'}), 503
        
        # Parse request data
        data = request.get_json()
        if not data:
            raise BadRequest('No JSON data provided')
        
        text = data.get('text', '').strip()
        speaker_id = data.get('speaker_id', 'default')
        voice_preset = data.get('voice_preset', 'commercial-warm')
        
        # Validate input
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if speaker_id not in SPEAKERS:
            return jsonify({'error': f'Invalid speaker_id: {speaker_id}'}), 400
        
        if voice_preset not in VOICE_PRESETS:
            return jsonify({'error': f'Invalid voice_preset: {voice_preset}'}), 400
        
        # Generate unique audio ID
        audio_id = str(uuid.uuid4())
        audio_filename = f"voice_{audio_id}.wav"
        audio_path = os.path.join(temp_audio_dir, audio_filename)
        
        # Generate voice over
        success = voice_synthesizer.synthesize_speech(
            text=text,
            output_path=audio_path,
            voice_preset=voice_preset,
            speaker_id=speaker_id
        )
        
        if not success:
            return jsonify({'error': 'Failed to generate voice over'}), 500
        
        # Process audio (mock processing)
        processed_path = audio_processor.process_audio(audio_path)
        
        # Evaluate quality
        quality_metrics = tts_evaluator.evaluate_audio(processed_path, text)
        
        # Prepare metadata
        metadata = {
            'audio_id': audio_id,
            'text': text,
            'speaker': SPEAKERS[speaker_id]['name'],
            'preset': VOICE_PRESETS[voice_preset]['name'],
            'duration': quality_metrics.get('duration', 0),
            'sample_rate': 22050,
            'quality_score': quality_metrics.get('overall_score', 0.85),
            'synthesis_method': 'espeak-ng' if voice_synthesizer.espeak_available else 'mock',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'audio_id': audio_id,
            'audio_url': f'/api/audio/{audio_id}',
            'metadata': metadata,
            'quality_metrics': quality_metrics
        })
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@tts_bp.route('/audio/<audio_id>', methods=['GET'])
def get_audio(audio_id):
    """Download generated audio file"""
    try:
        audio_filename = f"voice_{audio_id}.wav"
        audio_path = os.path.join(temp_audio_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_file(
            audio_path,
            as_attachment=True,
            download_name=audio_filename,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/evaluate', methods=['POST'])
def evaluate_audio():
    """Evaluate audio quality"""
    try:
        if not tts_evaluator:
            return jsonify({'error': 'TTS evaluator not initialized'}), 503
        
        data = request.get_json()
        if not data:
            raise BadRequest('No JSON data provided')
        
        audio_id = data.get('audio_id')
        text = data.get('text', '')
        
        if not audio_id:
            return jsonify({'error': 'audio_id is required'}), 400
        
        audio_filename = f"voice_{audio_id}.wav"
        audio_path = os.path.join(temp_audio_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Evaluate audio quality
        quality_metrics = tts_evaluator.evaluate_audio(audio_path, text)
        
        return jsonify({
            'audio_id': audio_id,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tts_bp.route('/batch-generate', methods=['POST'])
def batch_generate():
    """Generate multiple voice overs in batch"""
    try:
        if not voice_synthesizer:
            return jsonify({'error': 'TTS system not initialized'}), 503
        
        data = request.get_json()
        if not data:
            raise BadRequest('No JSON data provided')
        
        texts = data.get('texts', [])
        speaker_id = data.get('speaker_id', 'default')
        voice_preset = data.get('voice_preset', 'commercial-warm')
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts allowed per batch'}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            
            try:
                # Generate unique audio ID
                audio_id = str(uuid.uuid4())
                audio_filename = f"voice_{audio_id}.wav"
                audio_path = os.path.join(temp_audio_dir, audio_filename)
                
                # Generate voice over
                success = voice_synthesizer.synthesize_speech(
                    text=text.strip(),
                    output_path=audio_path,
                    voice_preset=voice_preset,
                    speaker_id=speaker_id
                )
                
                if success:
                    # Process and evaluate
                    processed_path = audio_processor.process_audio(audio_path)
                    quality_metrics = tts_evaluator.evaluate_audio(processed_path, text)
                    
                    results.append({
                        'index': i,
                        'text': text.strip(),
                        'audio_id': audio_id,
                        'audio_url': f'/api/audio/{audio_id}',
                        'success': True,
                        'quality_score': quality_metrics.get('overall_score', 0.85)
                    })
                else:
                    results.append({
                        'index': i,
                        'text': text.strip(),
                        'success': False,
                        'error': 'Failed to generate voice over'
                    })
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text.strip(),
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_id': str(uuid.uuid4()),
            'total_requested': len(texts),
            'total_generated': len([r for r in results if r.get('success')]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize TTS system when blueprint is imported
initialize_tts_system()

