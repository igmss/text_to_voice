import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.tts import tts_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'egyptian-tts-production-key-2024'

# Configure CORS for production
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://*.amplifyapp.com",
    "https://*.amazonaws.com"
])

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(tts_bp, url_prefix='/api')

# Database configuration (disabled for TTS system)
# app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)
# with app.app_context():
#     db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve static files and React app"""
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
            return "index.html not found", 404

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return {
        'status': 'running',
        'service': 'Egyptian Arabic TTS API',
        'version': '2.0.0',
        'endpoints': {
            'health': '/api/health',
            'generate': '/api/generate',
            'presets': '/api/presets',
            'speakers': '/api/speakers',
            'system': '/api/system-info'
        }
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print("🎙️ Starting Egyptian Arabic TTS Production Server")
    print(f"🌐 Server will be available at http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

