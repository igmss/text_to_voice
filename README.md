# Egyptian Arabic Text-to-Speech Voice Over System

A comprehensive text-to-speech system specifically designed for Egyptian Arabic dialect, with support for Egyptian slang and voice over production.

## Features

- Egyptian Arabic dialect support
- Egyptian slang processing
- High-quality voice synthesis
- Real-time text-to-speech conversion
- Quality metrics and monitoring
- Web-based user interface
- Comprehensive testing framework
- Multiple voice presets for different use cases
- Batch processing capabilities
- Professional audio quality standards

## Project Structure

```
├── frontend/                 # React-based web interface
│   ├── App.jsx              # Main application component
│   ├── App.css              # Styling
│   ├── index.html           # HTML template
│   └── vite.config.js       # Vite configuration
│
├── backend/                  # Python-based backend
│   ├── main.py              # Main application entry point
│   ├── main_mock.py         # Mock version for testing
│   ├── egyptian_tts.py      # Core TTS implementation
│   ├── audio_processor.py   # Audio processing utilities
│   ├── text_processor.py    # Text processing utilities
│   ├── voice_synthesizer.py # Voice synthesis engine
│   ├── metrics.py           # Quality metrics
│   ├── quality_dashboard.py # Quality monitoring
│   └── test_framework.py    # Testing framework
│
├── mock_implementations/     # Mock versions for testing
│   ├── mock_voice_synthesizer.py
│   ├── mock_audio_processor.py
│   └── mock_evaluator.py
│
└── docs/                    # Documentation
    ├── Complete Documentation.md
    ├── Architecture Implementation.md
    ├── Research Findings.md
    ├── Deployment Guide.md
    ├── Quick Start Guide.md
    └── project_analysis_and_enhancement_plan.md
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher (for frontend)
- espeak-ng system package

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/igmss/text_to_voice.git
cd text_to_voice
```

2. Install system dependencies:
```bash
sudo apt update
sudo apt install -y espeak-ng
```

3. Install Python dependencies:
```bash
# Install basic dependencies first
pip install flask flask-cors soundfile librosa pydub pyarabic arabic-reshaper python-bidi phonemizer

# For full TTS functionality (requires more memory):
# pip install torch torchaudio TTS
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
# Create package.json if it doesn't exist
npm init -y
npm install vite @vitejs/plugin-react react react-dom
```

2. Configure Vite (vite.config.js should already exist)

## Usage

### Quick Start (Mock Version)

For testing and development, you can use the mock version:

```bash
python main_mock.py
```

This will start a mock TTS server at `http://localhost:5000` with simulated audio generation.

### Full Version

For production use with actual TTS:

```bash
python main.py
```

### Frontend Development

```bash
# Start the frontend development server
npm run dev
```

Access the web interface at `http://localhost:5173`

## API Endpoints

### Core Endpoints

- `GET /api/health` - Health check
- `POST /api/generate` - Generate voice over from text
- `GET /api/presets` - Get available voice presets
- `GET /api/speakers` - Get available speaker voices
- `GET /api/audio/<audio_id>` - Download generated audio
- `GET /api/system-info` - Get system capabilities

### Example API Usage

```bash
# Generate voice over
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا بكم في نظام تحويل النص إلى كلام المصري",
    "speaker_id": "default",
    "voice_preset": "commercial-warm"
  }'
```

## Voice Presets

- **commercial-energetic**: High-energy commercial voice over
- **commercial-warm**: Warm and friendly commercial voice
- **educational-clear**: Clear and measured educational delivery
- **documentary-authoritative**: Authoritative documentary narration
- **audiobook-natural**: Natural storytelling voice
- **news-professional**: Professional news delivery

## Speaker Voices

- **default**: Default Egyptian Voice (neutral)
- **male-young**: Ahmed - Young Male
- **female-adult**: Fatima - Adult Female
- **male-mature**: Omar - Mature Male

## Development Status

### Current Implementation
- ✅ Flask API server with comprehensive endpoints
- ✅ Mock TTS system for testing and development
- ✅ Voice presets and speaker configurations
- ✅ Audio processing pipeline
- ✅ Quality metrics and evaluation
- ✅ React frontend interface
- ✅ Comprehensive documentation

### In Progress
- 🔄 Full TTS engine integration
- 🔄 Arabic text processing optimization
- 🔄 Frontend-backend integration testing
- 🔄 Production deployment configuration

### Planned Features
- 📋 User authentication system
- 📋 Audio history and management
- 📋 Real-time audio streaming
- 📋 Voice cloning capabilities
- 📋 SSML support
- 📋 Batch processing optimization

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Complete Documentation](Egyptian%20Arabic%20Text-to-Speech%20Voice%20Over%20System_%20Complete%20Documentation.md)
- [Architecture Implementation](Egyptian%20Arabic%20TTS%20Architecture%20Implementation.md)
- [Research Findings](Egyptian%20Arabic%20TTS%20Research%20Findings.md)
- [Deployment Guide](Egyptian%20Arabic%20TTS%20Voice%20Over%20System_%20Deployment%20Guide.md)
- [Quick Start Guide](Egyptian%20Arabic%20TTS%20Voice%20Over%20System_%20Quick%20Start%20Guide.md)
- [Project Analysis and Enhancement Plan](project_analysis_and_enhancement_plan.md)

## Troubleshooting

### Common Issues

1. **Memory errors during installation**: Use the mock version for development
2. **Import errors**: Ensure all dependencies are installed correctly
3. **Audio generation fails**: Check that espeak-ng is installed system-wide
4. **Frontend not loading**: Verify Node.js dependencies are installed

### Getting Help

- Check the documentation in the `docs/` folder
- Review the project analysis document for known issues
- Open an issue on GitHub for bug reports

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and supporters of this project
- Special thanks to the Egyptian Arabic language community
- Built with modern web technologies and AI/ML frameworks

