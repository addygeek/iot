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
