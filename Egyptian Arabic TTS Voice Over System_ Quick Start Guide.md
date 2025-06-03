# Egyptian Arabic TTS Voice Over System: Quick Start Guide

**Version:** 1.0.0  
**Target Audience:** Content Creators, Developers, System Administrators  
**Estimated Setup Time:** 15-30 minutes  

## Overview

This guide provides step-by-step instructions for setting up and using the Egyptian Arabic TTS Voice Over System. Whether you're a content creator looking to generate voice overs or a developer integrating TTS capabilities into your application, this guide will get you started quickly.

## System Requirements

### Minimum Requirements
- **Operating System:** Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **RAM:** 8GB (16GB recommended)
- **Storage:** 10GB free space
- **CPU:** Intel i5 / AMD Ryzen 5 or equivalent
- **Network:** Stable internet connection for initial setup

### Recommended Requirements
- **RAM:** 16GB or more
- **Storage:** 20GB+ free space (SSD recommended)
- **CPU:** Intel i7 / AMD Ryzen 7 or equivalent
- **GPU:** NVIDIA GPU with 4GB+ VRAM (optional, for faster processing)

### Software Dependencies
- **Python:** 3.8 or higher
- **Node.js:** 16.0 or higher
- **npm/pnpm:** Latest version
- **Git:** For cloning the repository

## Quick Installation

### Option 1: Using Pre-built Release (Recommended)

1. **Download the latest release:**
   ```bash
   wget https://github.com/egyptian-tts/releases/latest/egyptian-tts-system.tar.gz
   tar -xzf egyptian-tts-system.tar.gz
   cd egyptian-tts-system
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start the system:**
   ```bash
   ./start.sh
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://localhost:3000`

### Option 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/egyptian-tts/egyptian-voice-studio.git
   cd egyptian-voice-studio
   ```

2. **Set up the backend:**
   ```bash
   cd voice_api
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend:**
   ```bash
   cd ../web_app/egyptian-voice-studio
   npm install
   ```

4. **Start the backend:**
   ```bash
   cd ../../voice_api
   source venv/bin/activate
   python src/main.py
   ```

5. **Start the frontend (in a new terminal):**
   ```bash
   cd web_app/egyptian-voice-studio
   npm run dev
   ```

## First Steps

### 1. Verify Installation

Once the system is running, verify that everything is working correctly:

1. Open your browser and go to `http://localhost:3000`
2. You should see the Egyptian Voice Studio interface
3. Check that the API is responding by visiting `http://localhost:5000/api/health`

### 2. Generate Your First Voice Over

1. **Enter Egyptian Arabic text:**
   - Click on the text area in the main interface
   - Type or paste Egyptian Arabic text (e.g., "مرحبا بكم في برنامجنا الجديد")
   - Or click one of the sample text buttons for quick testing

2. **Choose voice settings:**
   - Select a voice preset (e.g., "Commercial - Warm")
   - Choose a speaker voice (e.g., "Fatima - Adult Female")
   - Adjust speed, pitch, and energy if needed

3. **Generate audio:**
   - Click the "Generate Voice Over" button
   - Wait for processing (usually 5-15 seconds)
   - Listen to the generated audio using the built-in player

4. **Download your audio:**
   - Click the download button to save the audio file
   - Choose your preferred format (WAV or MP3)

### 3. Explore Advanced Features

- **Batch Processing:** Upload multiple texts for simultaneous processing
- **Quality Assessment:** Review quality scores and metrics
- **History:** Access previously generated voice overs
- **Analytics:** View usage statistics and quality trends

## Common Use Cases

### Content Creators

**Creating YouTube Video Narration:**
1. Prepare your script in Egyptian Arabic
2. Use the "Educational - Clear" or "Commercial - Energetic" preset
3. Generate and download high-quality WAV files
4. Import into your video editing software

**Podcast Production:**
1. Use the "Audiobook - Natural" preset for storytelling
2. Generate in WAV format for best quality
3. Apply additional post-processing if needed

### Developers

**API Integration:**
```python
import requests

# Generate voice over via API
response = requests.post('http://localhost:5000/api/generate', json={
    'text': 'مرحبا بكم في تطبيقنا الجديد',
    'speaker_id': 'female-adult',
    'voice_preset': 'commercial-warm'
})

audio_url = response.json()['audio_url']
```

**Batch Processing:**
```python
# Generate multiple voice overs
texts = [
    'النص الأول',
    'النص الثاني', 
    'النص الثالث'
]

response = requests.post('http://localhost:5000/api/batch-generate', json={
    'texts': texts,
    'speaker_id': 'male-young',
    'voice_preset': 'news-professional'
})
```

### Educators

**Creating Learning Materials:**
1. Use the "Educational - Clear" preset
2. Ensure text includes proper diacritics for pronunciation
3. Generate at high quality (48kHz WAV)
4. Test with target audience for clarity

## Troubleshooting

### Common Issues

**"TTS system not available" error:**
- Ensure all Python dependencies are installed
- Check that the backend server is running
- Verify system requirements are met

**Poor audio quality:**
- Check input text for proper Arabic encoding
- Try different voice presets
- Ensure adequate system resources

**Slow generation:**
- Close unnecessary applications
- Consider using a GPU if available
- Reduce text length for testing

**Web interface not loading:**
- Verify both frontend and backend are running
- Check for port conflicts (3000, 5000)
- Clear browser cache and cookies

### Getting Help

- **Documentation:** Full system documentation available in `/documentation/`
- **API Reference:** Available at `http://localhost:5000/api/system-info`
- **Community:** Join our Discord server for community support
- **Issues:** Report bugs on GitHub Issues page

## Next Steps

Once you're comfortable with basic usage:

1. **Explore Voice Customization:** Experiment with different presets and settings
2. **Integrate with Your Workflow:** Use the API for automated content generation
3. **Optimize for Your Use Case:** Fine-tune settings for your specific requirements
4. **Scale Your Usage:** Implement batch processing for larger projects

## Security Considerations

- The system runs locally by default for privacy
- No audio data is sent to external servers
- Generated files are stored locally and can be deleted
- API access can be restricted using standard web security practices

## Performance Optimization

- **For better performance:** Use SSD storage and adequate RAM
- **For faster generation:** Consider GPU acceleration
- **For batch processing:** Increase system resources and use API directly
- **For production use:** Consider deploying on dedicated server hardware

This quick start guide covers the essential steps to get you up and running with the Egyptian Arabic TTS Voice Over System. For detailed technical information, advanced configuration options, and comprehensive API documentation, please refer to the complete system documentation.

