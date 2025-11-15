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
