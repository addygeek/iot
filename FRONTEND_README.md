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
