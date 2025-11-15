#!/bin/bash

##############################################################################
# Raspberry Pi 5 Meeting Recorder - Complete Setup Script
# This script installs and configures everything needed for the IoT recorder
##############################################################################

set -e  # Exit on any error

echo "========================================================================"
echo "  RASPBERRY PI 5 MEETING RECORDER - AUTOMATED SETUP"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$HOME/meeting_recorder"
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_NAME="vosk-model-small-en-us-0.15"
SERVICE_NAME="meeting-recorder"

##############################################################################
# FUNCTIONS
##############################################################################

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

##############################################################################
# SYSTEM UPDATE
##############################################################################

print_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
print_status "System updated"

##############################################################################
# INSTALL DEPENDENCIES
##############################################################################

print_info "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    git \
    wget \
    unzip \
    curl \
    build-essential \
    alsa-utils \
    pulseaudio

print_status "System dependencies installed"

##############################################################################
# CREATE PROJECT DIRECTORY
##############################################################################

print_info "Creating project directory at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
print_status "Project directory created"

##############################################################################
# CREATE PYTHON VIRTUAL ENVIRONMENT
##############################################################################

print_info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
print_status "Virtual environment created"

##############################################################################
# INSTALL PYTHON PACKAGES
##############################################################################

print_info "Installing Python packages..."

pip install --upgrade pip

cat > requirements.txt << 'EOF'
vosk==0.3.45
pyaudio==0.2.14
pyyaml==6.0.1
numpy==1.24.3
nltk==3.8.1
scikit-learn==1.3.0
sentence-transformers==2.2.2
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
EOF

pip install -r requirements.txt
print_status "Python packages installed"

##############################################################################
# DOWNLOAD VOSK MODEL
##############################################################################

print_info "Downloading Vosk model (this may take a few minutes)..."
if [ ! -d "$PROJECT_DIR/$VOSK_MODEL_NAME" ]; then
    wget -q --show-progress "$VOSK_MODEL_URL" -O vosk_model.zip
    unzip -q vosk_model.zip
    rm vosk_model.zip
    print_status "Vosk model downloaded and extracted"
else
    print_status "Vosk model already exists"
fi

##############################################################################
# DOWNLOAD NLTK DATA
##############################################################################

print_info "Downloading NLTK data..."
python3 << 'PYTHON_EOF'
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data downloaded")
PYTHON_EOF
print_status "NLTK data ready"

##############################################################################
# CREATE CONFIG DIRECTORY AND FILE
##############################################################################

print_info "Creating configuration..."
mkdir -p configs
mkdir -p recordings

cat > configs/recorder_config.yml << EOF
model_path: $PROJECT_DIR/$VOSK_MODEL_NAME
sample_rate: 16000
channels: 1
block_duration_ms: 500
wav_format: PCM_16
save_dir: $PROJECT_DIR/recordings
summarizer: detailed
extractive_sentences: 5
auto_summary_interval_seconds: 0
mic_device_name: null
server_port: 5000
server_host: 0.0.0.0
EOF

print_status "Configuration created"

##############################################################################
# CREATE CORE MODULES
##############################################################################

print_info "Creating application modules..."

# recorder.py
cat > recorder.py << 'RECORDER_EOF'
"""Audio Recorder Module - Handles microphone input and WAV file writing"""

import pyaudio
import wave
import queue
import time
import os


class AudioRecorder:
    def __init__(self, config, session_folder, session_name):
        self.config = config
        self.session_folder = session_folder
        self.session_name = session_name
        
        self.sample_rate = config['sample_rate']
        self.channels = config['channels']
        self.block_duration_ms = config['block_duration_ms']
        self.chunk_size = int(self.sample_rate * self.block_duration_ms / 1000)
        
        self.audio = None
        self.stream = None
        self.wav_file = None
        self.audio_queue = queue.Queue()
        self.recording = False
        self.start_time = None
        self.frames_recorded = 0
        
        self._init_audio()
        self._init_wav_file()
    
    def _init_audio(self):
        self.audio = pyaudio.PyAudio()
        device_index = None
        mic_name = self.config.get('mic_device_name')
        
        if mic_name:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if mic_name.lower() in info['name'].lower():
                    device_index = i
                    break
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
    
    def _init_wav_file(self):
        wav_filename = os.path.join(self.session_folder, f"{self.session_name}.wav")
        self.wav_file = wave.open(wav_filename, 'wb')
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        self.wav_file.setframerate(self.sample_rate)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.audio_queue.put(in_data)
            self.wav_file.writeframes(in_data)
            self.frames_recorded += frame_count
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        self.recording = True
        self.start_time = time.time()
        if self.stream:
            self.stream.start_stream()
    
    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.wav_file:
            self.wav_file.close()
        if self.audio:
            self.audio.terminate()
    
    def get_audio_block(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_duration(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_frames_recorded(self):
        return self.frames_recorded
RECORDER_EOF

# stt_engine.py
cat > stt_engine.py << 'STT_EOF'
"""Vosk STT Engine Module"""

import json
from vosk import Model, KaldiRecognizer


class VoskSTTEngine:
    def __init__(self, model_path, sample_rate):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)
        self.partial_count = 0
        self.final_count = 0
    
    def process_audio(self, audio_data):
        if not audio_data:
            return None
        
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            if result.get('text', '').strip():
                self.final_count += 1
                return {
                    'type': 'final',
                    'text': result['text'],
                    'words': result.get('result', [])
                }
        else:
            result = json.loads(self.recognizer.PartialResult())
            if result.get('partial', '').strip():
                self.partial_count += 1
                return {
                    'type': 'partial',
                    'text': result['partial']
                }
        return None
    
    def get_final_result(self):
        try:
            result = json.loads(self.recognizer.FinalResult())
            if result.get('text', '').strip():
                return {
                    'type': 'final',
                    'text': result['text'],
                    'words': result.get('result', [])
                }
        except Exception:
            pass
        return None
    
    def reset(self):
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
STT_EOF

# transcript_aggregator.py
cat > transcript_aggregator.py << 'TRANS_EOF'
"""Transcript Aggregator Module"""

import os
from datetime import datetime, timedelta


class TranscriptAggregator:
    def __init__(self, session_folder, session_name):
        self.session_folder = session_folder
        self.session_name = session_name
        self.segments = []
        self.start_time = datetime.now()
        self.transcript_file = os.path.join(session_folder, f"{session_name}.txt")
    
    def add_segment(self, text, words=None):
        if not text or not text.strip():
            return
        
        elapsed = datetime.now() - self.start_time
        timestamp = self._format_timestamp(elapsed.total_seconds())
        
        segment = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed.total_seconds(),
            'text': text.strip(),
            'words': words or []
        }
        self.segments.append(segment)
    
    def _format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_transcript(self):
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"Transcript: {self.session_name}\n")
            f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for segment in self.segments:
                f.write(f"[{segment['timestamp']}] {segment['text']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Total segments: {len(self.segments)}\n")
        
        return self.transcript_file
    
    def get_full_transcript(self):
        return ' '.join(segment['text'] for segment in self.segments)
    
    def get_timestamped_transcript(self):
        return self.segments.copy()
TRANS_EOF

# summarizer.py
cat > summarizer.py << 'SUMM_EOF'
"""Advanced Speech-Aware Summarizer"""

import os
import re
from datetime import datetime
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except:
    EMBEDDINGS_AVAILABLE = False


class Summarizer:
    def __init__(self, mode='detailed', num_sentences=5):
        self.mode = mode
        self.num_sentences = num_sentences
        
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
        
        self.filler_words = {
            "okay", "uh", "um", "you know", "like", "so", "basically",
            "actually", "right", "hmm", "alright"
        }
    
    def generate_summary(self, text):
        if not text or len(text.strip()) < 20:
            return "Text too short to summarize."
        
        cleaned = self._clean_speech_text(text)
        
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(cleaned)
        except:
            sentences = cleaned.split('. ')
        
        if len(sentences) <= 3:
            return cleaned
        
        return self._semantic_summary(sentences, min(5, len(sentences)))
    
    def _clean_speech_text(self, text):
        lowered = text.lower()
        for w in self.filler_words:
            lowered = re.sub(rf"\b{w}\b", "", lowered)
        lowered = re.sub(r"\b(\w+)\s+\1\b", r"\1", lowered)
        return lowered.strip()
    
    def _semantic_summary(self, sentences, top_k=3):
        if not sentences:
            return ""
        
        if len(sentences) <= top_k:
            return " ".join(sentences)
        
        if EMBEDDINGS_AVAILABLE and self.embedding_model:
            embeddings = self.embedding_model.encode(sentences)
            sim_matrix = cosine_similarity(embeddings)
            scores = sim_matrix.sum(axis=1)
        else:
            scores = np.array([len(s.split()) for s in sentences])
        
        ranked = np.argsort(scores)[::-1][:top_k]
        ranked = sorted(ranked)
        selected = [sentences[i] for i in ranked]
        
        return "\n".join(selected)
    
    def save_summary(self, summary, session_folder, session_name):
        summary_file = os.path.join(session_folder, f"{session_name}_summary.txt")
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Summary: {session_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "=" * 60 + "\n")
        
        return summary_file
SUMM_EOF

# logger.py
cat > logger.py << 'LOGGER_EOF'
"""Session Logger Module"""

import os
from datetime import datetime


class SessionLogger:
    def __init__(self, session_folder, session_name):
        self.session_folder = session_folder
        self.session_name = session_name
        self.log_file = os.path.join(session_folder, f"{session_name}_log.txt")
        self.errors = []
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self._write_header()
    
    def _write_header(self):
        self.file_handle.write("=" * 60 + "\n")
        self.file_handle.write(f"Session Log: {self.session_name}\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 60 + "\n\n")
        self.file_handle.flush()
    
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        self.file_handle.write(log_line)
        self.file_handle.flush()
        
        if level == "ERROR":
            self.errors.append({'timestamp': timestamp, 'message': message})
    
    def get_errors(self):
        return self.errors.copy()
    
    def close(self):
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.write("\n" + "=" * 60 + "\n")
            self.file_handle.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.close()
LOGGER_EOF

print_status "Core modules created"

##############################################################################
# CREATE FLASK SERVER
##############################################################################

print_info "Creating Flask API server..."

cat > server.py << 'SERVER_EOF'
#!/usr/bin/env python3
"""
Flask API Server for Raspberry Pi Meeting Recorder
Provides REST API for remote recording control
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import yaml
import threading
import json
from datetime import datetime
from pathlib import Path
import base64

from recorder import AudioRecorder
from stt_engine import VoskSTTEngine
from transcript_aggregator import TranscriptAggregator
from summarizer import Summarizer
from logger import SessionLogger

app = Flask(__name__)
CORS(app)

# Global state
current_session = None
session_lock = threading.Lock()


class RecordingSession:
    def __init__(self, config):
        self.config = config
        self.session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_name = f"session_{self.session_timestamp}"
        
        save_dir = config['save_dir']
        self.session_folder = os.path.join(save_dir, self.session_name)
        os.makedirs(self.session_folder, exist_ok=True)
        
        self.logger = SessionLogger(self.session_folder, self.session_name)
        self.recorder = AudioRecorder(config, self.session_folder, self.session_name)
        self.stt_engine = VoskSTTEngine(config['model_path'], config['sample_rate'])
        self.aggregator = TranscriptAggregator(self.session_folder, self.session_name)
        self.summarizer = Summarizer(config['summarizer'], config['extractive_sentences'])
        
        self.running = False
        self.process_thread = None
    
    def start(self):
        self.running = True
        self.recorder.start()
        self.logger.log("Recording started")
        
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()
    
    def _process_loop(self):
        while self.running:
            audio_block = self.recorder.get_audio_block()
            if audio_block is None:
                continue
            
            result = self.stt_engine.process_audio(audio_block)
            if result and result['type'] == 'final':
                self.aggregator.add_segment(result['text'])
                self.logger.log(f"Transcribed: {result['text'][:50]}...")
    
    def stop(self):
        self.running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2)
        
        self.recorder.stop()
        
        final_result = self.stt_engine.get_final_result()
        if final_result and final_result.get('text'):
            self.aggregator.add_segment(final_result['text'])
        
        transcript_file = self.aggregator.save_transcript()
        transcript_text = self.aggregator.get_full_transcript()
        
        summary = ""
        summary_file = ""
        if transcript_text.strip():
            summary = self.summarizer.generate_summary(transcript_text)
            summary_file = self.summarizer.save_summary(
                summary, self.session_folder, self.session_name
            )
        
        self.logger.log("Recording stopped")
        self.logger.close()
        
        return {
            'session_name': self.session_name,
            'session_folder': self.session_folder,
            'duration': self.recorder.get_duration(),
            'transcript_file': transcript_file,
            'summary_file': summary_file,
            'summary': summary,
            'audio_file': os.path.join(self.session_folder, f"{self.session_name}.wav")
        }
    
    def get_status(self):
        return {
            'session_name': self.session_name,
            'running': self.running,
            'duration': self.recorder.get_duration(),
            'segments': len(self.aggregator.segments)
        }


def load_config():
    config_path = 'configs/recorder_config.yml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Meeting Recorder API is running'}), 200


@app.route('/start', methods=['POST'])
def start_recording():
    global current_session
    
    with session_lock:
        if current_session and current_session.running:
            return jsonify({'error': 'Recording already in progress'}), 400
        
        try:
            config = load_config()
            current_session = RecordingSession(config)
            current_session.start()
            
            return jsonify({
                'status': 'started',
                'session_name': current_session.session_name,
                'message': 'Recording started successfully'
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_recording():
    global current_session
    
    with session_lock:
        if not current_session or not current_session.running:
            return jsonify({'error': 'No active recording'}), 400
        
        try:
            result = current_session.stop()
            
            # Read files for response
            with open(result['transcript_file'], 'r') as f:
                transcript_content = f.read()
            
            with open(result['audio_file'], 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'status': 'stopped',
                'session_name': result['session_name'],
                'duration': result['duration'],
                'summary': result['summary'],
                'transcript': transcript_content,
                'audio_base64': audio_base64,
                'files': {
                    'transcript': result['transcript_file'],
                    'summary': result['summary_file'],
                    'audio': result['audio_file']
                }
            }
            
            current_session = None
            return jsonify(response_data), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    global current_session
    
    with session_lock:
        if not current_session:
            return jsonify({'running': False, 'message': 'No active session'}), 200
        
        return jsonify(current_session.get_status()), 200


@app.route('/download/<session_name>/<file_type>', methods=['GET'])
def download_file(session_name, file_type):
    config = load_config()
    session_folder = os.path.join(config['save_dir'], session_name)
    
    if file_type == 'audio':
        file_path = os.path.join(session_folder, f"{session_name}.wav")
        mimetype = 'audio/wav'
    elif file_type == 'transcript':
        file_path = os.path.join(session_folder, f"{session_name}.txt")
        mimetype = 'text/plain'
    elif file_type == 'summary':
        file_path = os.path.join(session_folder, f"{session_name}_summary.txt")
        mimetype = 'text/plain'
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, mimetype=mimetype, as_attachment=True)


@app.route('/sessions', methods=['GET'])
def list_sessions():
    config = load_config()
    recordings_dir = config['save_dir']
    
    if not os.path.exists(recordings_dir):
        return jsonify({'sessions': []}), 200
    
    sessions = []
    for folder in os.listdir(recordings_dir):
        folder_path = os.path.join(recordings_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('session_'):
            sessions.append({
                'name': folder,
                'path': folder_path,
                'created': os.path.getctime(folder_path)
            })
    
    sessions.sort(key=lambda x: x['created'], reverse=True)
    return jsonify({'sessions': sessions}), 200


if __name__ == '__main__':
    config = load_config()
    port = config.get('server_port', 5000)
    host = config.get('server_host', '0.0.0.0')
    
    print(f"Starting Meeting Recorder API on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
SERVER_EOF

chmod +x server.py
print_status "Flask server created"

##############################################################################
# CREATE SYSTEMD SERVICE
##############################################################################

print_info "Creating systemd service for auto-start..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Meeting Recorder API Server
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python3 $PROJECT_DIR/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service created"

##############################################################################
# CONFIGURE AUDIO
##############################################################################

print_info "Configuring audio system..."

# Set default audio device
sudo tee -a /etc/asound.conf > /dev/null << 'ASOUND_EOF'
pcm.!default {
    type asym
    playback.pcm "hw:0,0"
    capture.pcm "hw:0,0"
}
ASOUND_EOF

print_status "Audio configured"

##############################################################################
# ENABLE AND START SERVICE
##############################################################################

print_info "Enabling and starting service..."

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service
sudo systemctl start ${SERVICE_NAME}.service

sleep 3

if sudo systemctl is-active --quiet ${SERVICE_NAME}.service; then
    print_status "Service is running!"
else
    print_error "Service failed to start. Check logs with: sudo journalctl -u ${SERVICE_NAME}.service"
fi

##############################################################################
# GET RASPBERRY PI IP ADDRESS
##############################################################################

print_info "Getting network information..."
PI_IP=$(hostname -I | awk '{print $1}')

##############################################################################
# CREATE FRONTEND README
##############################################################################

print_info "Creating frontend documentation..."

cat > FRONTEND_README.md << 'FRONTEND_EOF'
# Frontend Integration Guide

## API Endpoints

Your Raspberry Pi is running a REST API server. Connect to it using:

**Base URL:** `http://<RASPBERRY_PI_IP>:5000`

### Available Endpoints

#### 1. Health Check
```
GET /health
```
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "Meeting Recorder API is running"
}
```

---

#### 2. Start Recording
```
POST /start
```
Start a new recording session.

**Response:**
```json
{
  "status": "started",
  "session_name": "session_2025-11-15_14-30-45",
  "message": "Recording started successfully"
}
```

---

#### 3. Stop Recording
```
POST /stop
```
Stop the current recording and get results.

**Response:**
```json
{
  "status": "stopped",
  "session_name": "session_2025-11-15_14-30-45",
  "duration": 125.5,
  "summary": "Meeting discussed project timeline and budget allocation...",
  "transcript": "[00:00:15] We need to finalize the budget\n[00:00:32] The deadline is next week...",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "files": {
    "transcript": "/home/pi/meeting_recorder/recordings/session_2025-11-15_14-30-45/session_2025-11-15_14-30-45.txt",
    "summary": "/home/pi/meeting_recorder/recordings/session_2025-11-15_14-30-45/session_2025-11-15_14-30-45_summary.txt",
    "audio": "/home/pi/meeting_recorder/recordings/session_2025-11-15_14-30-45/session_2025-11-15_14-30-45.wav"
  }
}
```

---

#### 4. Get Status
```
GET /status
```
Check current recording status.

**Response (Recording Active):**
```json
{
  "session_name": "session_2025-11-15_14-30-45",
  "running": true,
  "duration": 45.2,
  "segments": 12
}
```

**Response (No Active Recording):**
```json
{
  "running": false,
  "message": "No active session"
}
```

---

#### 5. Download Files
```
GET /download/<session_name>/<file_type>
```

**File Types:**
- `audio` - WAV audio file
- `transcript` - Full transcript with timestamps
- `summary` - Generated summary

**Example:**
```
GET /download/session_2025-11-15_14-30-45/audio
GET /download/session_2025-11-15_14-30-45/transcript
GET /download/session_2025-11-15_14-30-45/summary
```

---

#### 6. List All Sessions
```
GET /sessions
```
Get list of all recorded sessions.

**Response:**
```json
{
  "sessions": [
    {
      "name": "session_2025-11-15_14-30-45",
      "path": "/home/pi/meeting_recorder/recordings/session_2025-11-15_14-30-45",
      "created": 1731675045.123
    }
  ]
}
```

---

## Frontend Example Code

### Using Vanilla JavaScript (Fetch API)

```javascript
const PI_IP = '192.168.1.100'; // Replace with your Raspberry Pi IP
const API_URL = `http://${PI_IP}:5000`;

// Start Recording
async function startRecording() {
  try {
    const response = await fetch(`${API_URL}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    const data = await response.json();
    console.log('Recording started:', data);
    alert(`Recording started: ${data.session_name}`);
  } catch (error) {
    console.error('Error starting recording:', error);
    alert('Failed to start recording');
  }
}

// Stop Recording
async function stopRecording() {
  try {
    const response = await fetch(`${API_URL}/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    const data = await response.json();
    console.log('Recording stopped:', data);
    
    // Display summary
    document.getElementById('summary').textContent = data.summary;
    
    // Display transcript
    document.getElementById('transcript').textContent = data.transcript;
    
    // Create audio player
    const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.src = audioUrl;
    
    alert('Recording stopped successfully!');
  } catch (error) {
    console.error('Error stopping recording:', error);
    alert('Failed to stop recording');
  }
}

// Get Status
async function getStatus() {
  try {
    const response = await fetch(`${API_URL}/status`);
    const data = await response.json();
    
    if (data.running) {
      document.getElementById('status').textContent = 
        `Recording... Duration: ${data.duration.toFixed(1)}s | Segments: ${data.segments}`;
    } else {
      document.getElementById('status').textContent = 'Not recording';
    }
  } catch (error) {
    console.error('Error getting status:', error);
  }
}

// Helper: Convert base64 to Blob
function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

// Download File
async function downloadFile(sessionName, fileType) {
  const url = `${API_URL}/download/${sessionName}/${fileType}`;
  window.open(url, '_blank');
}

// Poll status every 2 seconds while recording
setInterval(getStatus, 2000);
```

---

### Using React

```jsx
import React, { useState, useEffect } from 'react';

const PI_IP = '192.168.1.100'; // Replace with your Raspberry Pi IP
const API_URL = `http://${PI_IP}:5000`;

function MeetingRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        setStatus(data);
        setIsRecording(data.running);
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      console.log('Started:', data);
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Failed to start recording');
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch(`${API_URL}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      setResult(data);
      setIsRecording(false);
    } catch (error) {
      console.error('Error stopping recording:', error);
      alert('Failed to stop recording');
    }
  };

  return (
    <div className="meeting-recorder">
      <h1>Meeting Recorder</h1>
      
      <div className="controls">
        <button onClick={handleStart} disabled={isRecording}>
          Start Recording
        </button>
        <button onClick={handleStop} disabled={!isRecording}>
          Stop Recording
        </button>
      </div>

      {status && status.running && (
        <div className="status">
          <p>Recording... Duration: {status.duration.toFixed(1)}s</p>
          <p>Segments: {status.segments}</p>
        </div>
      )}

      {result && (
        <div className="results">
          <h2>Summary</h2>
          <p>{result.summary}</p>

          <h2>Transcript</h2>
          <pre>{result.transcript}</pre>

          <h2>Audio</h2>
          <audio controls src={`data:audio/wav;base64,${result.audio_base64}`} />
          
          <div className="downloads">
            <a href={`${API_URL}/download/${result.session_name}/audio`} download>
              Download Audio
            </a>
            <a href={`${API_URL}/download/${result.session_name}/transcript`} download>
              Download Transcript
            </a>
            <a href={`${API_URL}/download/${result.session_name}/summary`} download>
              Download Summary
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

export default MeetingRecorder;
```

---

## HTML Example (Complete Single Page)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Meeting Recorder</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 900px;
      margin: 50px auto;
      padding: 20px;
      background: #f5f5f5;
    }
    .container {
      background: white;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 { color: #333; }
    .controls {
      margin: 20px 0;
      display: flex;
      gap: 10px;
    }
    button {
      padding: 12px 24px;
      font-size: 16px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      transition: all 0.3s;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-start {
      background: #4CAF50;
      color: white;
    }
    .btn-start:hover:not(:disabled) {
      background: #45a049;
    }
    .btn-stop {
      background: #f44336;
      color: white;
    }
    .btn-stop:hover:not(:disabled) {
      background: #da190b;
    }
    .status {
      padding: 15px;
      background: #e3f2fd;
      border-left: 4px solid #2196F3;
      margin: 20px 0;
      border-radius: 4px;
    }
    .results {
      margin-top: 30px;
    }
    .section {
      margin: 20px 0;
      padding: 15px;
      background: #f9f9f9;
      border-radius: 5px;
    }
    .section h2 {
      margin-top: 0;
      color: #555;
    }
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #fff;
      padding: 15px;
      border-radius: 5px;
      border: 1px solid #ddd;
    }
    .downloads {
      display: flex;
      gap: 10px;
      margin-top: 15px;
    }
    .downloads a {
      padding: 10px 20px;
      background: #2196F3;
      color: white;
      text-decoration: none;
      border-radius: 5px;
      transition: background 0.3s;
    }
    .downloads a:hover {
      background: #0b7dda;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎤 Meeting Recorder</h1>
    
    <div class="controls">
      <button id="startBtn" class="btn-start" onclick="startRecording()">
        Start Recording
      </button>
      <button id="stopBtn" class="btn-stop" onclick="stopRecording()" disabled>
        Stop Recording
      </button>
    </div>

    <div id="statusDiv" class="status" style="display: none;">
      <strong>Status:</strong> <span id="statusText"></span>
    </div>

    <div id="resultsDiv" class="results" style="display: none;">
      <div class="section">
        <h2>📋 Summary</h2>
        <p id="summary"></p>
      </div>

      <div class="section">
        <h2>📝 Transcript</h2>
        <pre id="transcript"></pre>
      </div>

      <div class="section">
        <h2>🔊 Audio Recording</h2>
        <audio id="audioPlayer" controls style="width: 100%;"></audio>
      </div>

      <div class="downloads" id="downloadLinks"></div>
    </div>
  </div>

  <script>
    const PI_IP = '192.168.1.100'; // CHANGE THIS TO YOUR RASPBERRY PI IP
    const API_URL = `http://${PI_IP}:5000`;
    
    let statusInterval = null;
    let currentSessionName = null;

    async function startRecording() {
      try {
        const response = await fetch(`${API_URL}/start`, { method: 'POST' });
        const data = await response.json();
        
        if (response.ok) {
          currentSessionName = data.session_name;
          document.getElementById('startBtn').disabled = true;
          document.getElementById('stopBtn').disabled = false;
          document.getElementById('resultsDiv').style.display = 'none';
          
          // Start status polling
          statusInterval = setInterval(updateStatus, 2000);
          updateStatus();
        } else {
          alert('Error: ' + data.error);
        }
      } catch (error) {
        alert('Failed to connect to Raspberry Pi');
        console.error(error);
      }
    }

    async function stopRecording() {
      try {
        const response = await fetch(`${API_URL}/stop`, { method: 'POST' });
        const data = await response.json();
        
        if (response.ok) {
          document.getElementById('startBtn').disabled = false;
          document.getElementById('stopBtn').disabled = true;
          document.getElementById('statusDiv').style.display = 'none';
          
          // Stop status polling
          if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
          }
          
          // Display results
          displayResults(data);
        } else {
          alert('Error: ' + data.error);
        }
      } catch (error) {
        alert('Failed to stop recording');
        console.error(error);
      }
    }

    async function updateStatus() {
      try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        
        if (data.running) {
          document.getElementById('statusDiv').style.display = 'block';
          document.getElementById('statusText').textContent = 
            `Recording... Duration: ${data.duration.toFixed(1)}s | Segments: ${data.segments}`;
        } else {
          document.getElementById('statusDiv').style.display = 'none';
        }
      } catch (error) {
        console.error('Status update failed:', error);
      }
    }

    function displayResults(data) {
      // Show results section
      document.getElementById('resultsDiv').style.display = 'block';
      
      // Summary
      document.getElementById('summary').textContent = data.summary;
      
      // Transcript
      document.getElementById('transcript').textContent = data.transcript;
      
      // Audio
      const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
      const audioUrl = URL.createObjectURL(audioBlob);
      document.getElementById('audioPlayer').src = audioUrl;
      
      // Download links
      const downloadDiv = document.getElementById('downloadLinks');
      downloadDiv.innerHTML = `
        <a href="${API_URL}/download/${data.session_name}/audio" download>
          📥 Download Audio
        </a>
        <a href="${API_URL}/download/${data.session_name}/transcript" download>
          📥 Download Transcript
        </a>
        <a href="${API_URL}/download/${data.session_name}/summary" download>
          📥 Download Summary
        </a>
      `;
    }

    function base64ToBlob(base64, mimeType) {
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], { type: mimeType });
    }

    // Check connection on page load
    window.addEventListener('load', async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
          console.log('Connected to Raspberry Pi successfully');
        }
      } catch (error) {
        alert('Cannot connect to Raspberry Pi. Please check the IP address and network connection.');
      }
    });
  </script>
</body>
</html>
```

---

## Network Configuration

### Finding Your Raspberry Pi IP Address

**On Raspberry Pi terminal:**
```bash
hostname -I
```

**Or check your router's DHCP client list.**

### Setting Static IP (Recommended)

Edit `/etc/dhcpcd.conf`:
```bash
sudo nano /etc/dhcpcd.conf
```

Add at the end:
```
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

Reboot:
```bash
sudo reboot
```

---

## CORS Configuration

The API server has CORS enabled for all origins. If you need to restrict it, edit `server.py`:

```python
from flask_cors import CORS

# Allow only specific origin
CORS(app, origins=["http://192.168.1.50:3000"])
```

---

## Troubleshooting

### Cannot Connect to API
1. Check Raspberry Pi is on same WiFi network
2. Verify IP address: `hostname -I`
3. Check firewall: `sudo ufw status`
4. Test API: `curl http://<PI_IP>:5000/health`

### CORS Errors
- Ensure frontend is using `http://` not `https://`
- Check browser console for exact error
- Verify CORS is enabled in server.py

### Audio Not Playing
- Check base64 decoding is working
- Try downloading the file directly via `/download` endpoint
- Verify browser supports WAV playback

---

## Production Deployment Tips

1. **Use HTTPS**: Set up reverse proxy with Nginx + Let's Encrypt
2. **Authentication**: Add JWT or API key authentication
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Monitoring**: Set up logging and monitoring
5. **Backup**: Regularly backup recordings folder

---

## Example Frontend Folder Structure

```
frontend/
├── index.html          # Main HTML file (use example above)
├── css/
│   └── styles.css      # Custom styles
├── js/
│   └── app.js          # API integration logic
└── config.js           # Configuration (PI_IP, etc.)
```

Save the HTML example above as `index.html` and open in browser. Update the `PI_IP` variable to match your Raspberry Pi's IP address.

---

## API Response Examples

### Success Response
```json
{
  "status": "success",
  "data": { ... }
}
```

### Error Response
```json
{
  "error": "Description of error"
}
```

---

## Support

For issues with the API server:
```bash
# Check service status
sudo systemctl status meeting-recorder

# View logs
sudo journalctl -u meeting-recorder -f

# Restart service
sudo systemctl restart meeting-recorder
```
FRONTEND_EOF

print_status "Frontend documentation created"

##############################################################################
# CREATE TEST SCRIPT
##############################################################################

print_info "Creating API test script..."

cat > test_api.sh << 'TEST_EOF'
#!/bin/bash

echo "Testing Meeting Recorder API..."
echo ""

# Health check
echo "1. Health Check:"
curl -s http://localhost:5000/health | python3 -m json.tool
echo ""
echo ""

# Start recording
echo "2. Starting Recording:"
curl -s -X POST http://localhost:5000/start | python3 -m json.tool
echo ""
echo ""

# Wait 10 seconds
echo "3. Recording for 10 seconds... (speak now!)"
sleep 10
echo ""

# Status check
echo "4. Checking Status:"
curl -s http://localhost:5000/status | python3 -m json.tool
echo ""
echo ""

# Stop recording
echo "5. Stopping Recording:"
curl -s -X POST http://localhost:5000/stop | python3 -m json.tool
echo ""
echo ""

echo "Test completed!"
TEST_EOF

chmod +x test_api.sh
print_status "Test script created"

##############################################################################
# FINAL INFORMATION
##############################################################################

echo ""
echo "========================================================================"
echo "  INSTALLATION COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
print_status "Project directory: $PROJECT_DIR"
print_status "Recordings saved to: $PROJECT_DIR/recordings"
print_status "Service name: ${SERVICE_NAME}.service"
echo ""
print_info "Network Information:"
echo "   Raspberry Pi IP: $PI_IP"
echo "   API Server: http://${PI_IP}:5000"
echo ""
print_info "API Endpoints:"
echo "   Health Check:    GET  http://${PI_IP}:5000/health"
echo "   Start Recording: POST http://${PI_IP}:5000/start"
echo "   Stop Recording:  POST http://${PI_IP}:5000/stop"
echo "   Get Status:      GET  http://${PI_IP}:5000/status"
echo ""
print_info "Service Management:"
echo "   Status:  sudo systemctl status ${SERVICE_NAME}"
echo "   Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "   Start:   sudo systemctl start ${SERVICE_NAME}"
echo "   Restart: sudo systemctl restart ${SERVICE_NAME}"
echo "   Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
print_info "Testing:"
echo "   Run: cd $PROJECT_DIR && ./test_api.sh"
echo ""
print_info "Frontend Integration:"
echo "   Read: $PROJECT_DIR/FRONTEND_README.md"
echo ""
print_status "The API server is now running and will auto-start on boot!"
echo ""
echo "========================================================================"
echo "  NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Note your Raspberry Pi IP: $PI_IP"
echo "2. Read FRONTEND_README.md for API integration"
echo "3. Test the API: ./test_api.sh"
echo "4. Connect from your frontend using the IP above"
echo ""
echo "The system is ready to accept recording requests!"
echo ""a