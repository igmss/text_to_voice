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

## Project Structure

```
├── frontend/                 # React-based web interface
│   ├── App.jsx              # Main application component
│   ├── App.css              # Styling
│   └── vite.config.js       # Vite configuration
│
├── backend/                  # Python-based backend
│   ├── main.py              # Main application entry point
│   ├── egyptian_tts.py      # Core TTS implementation
│   ├── audio_processor.py   # Audio processing utilities
│   ├── text_processor.py    # Text processing utilities
│   ├── voice_synthesizer.py # Voice synthesis engine
│   ├── metrics.py           # Quality metrics
│   ├── quality_dashboard.py # Quality monitoring
│   └── test_framework.py    # Testing framework
│
└── docs/                    # Documentation
    ├── Complete Documentation.md
    ├── Architecture Implementation.md
    ├── Research Findings.md
    ├── Deployment Guide.md
    └── Quick Start Guide.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/igmss/text_to_voice.git
cd text_to_voice
```

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

## Usage

1. Start the backend server:
```bash
python main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the web interface at `http://localhost:5173`

## Documentation

- [Complete Documentation](docs/Complete%20Documentation.md)
- [Architecture Implementation](docs/Architecture%20Implementation.md)
- [Research Findings](docs/Research%20Findings.md)
- [Deployment Guide](docs/Deployment%20Guide.md)
- [Quick Start Guide](docs/Quick%20Start%20Guide.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[Your Name/GitHub Username]

## Acknowledgments

- Thanks to all contributors and supporters of this project
- Special thanks to the Egyptian Arabic language community 