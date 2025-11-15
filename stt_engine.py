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
