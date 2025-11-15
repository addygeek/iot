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
