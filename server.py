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
