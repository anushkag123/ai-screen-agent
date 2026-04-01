# AI Screen Assistant

This project is an app that turns a desktop overlay into a simple AI assistant. It can:

- capture your screen
- answer questions about what is visible
- route current-events questions to web search
- accept voice input
- serve as a starter project for memory, RAG, and tool-calling extensions

## 1. Install Python First

You need Python 3.10 or newer.

### macOS

Option A: Download Python from the official site

1. Go to https://www.python.org/downloads/
2. Download the latest Python 3 release
3. Run the installer
4. Verify:

```bash
python3 --version
```

Option B: Install with Homebrew

```bash
brew install python
python3 --version
```

### Windows

1. Go to https://www.python.org/downloads/
2. Download Python 3
3. During installation, check `Add Python to PATH`
4. Verify in PowerShell:

```powershell
python --version
```

If `python` does not work, try:

```powershell
py --version
```

### Linux

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

## 2. Clone and Enter the Project

```bash
git clone 
cd sp25-full-stack-ai
```

## 3. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 4. Install Project Dependencies

Install the packages used by the base app:

```bash
pip install fastapi uvicorn pywebview mss pillow httpx python-dotenv python-multipart sounddevice numpy
```

Optional packages for workshop extensions:

```bash
pip install chromadb sentence-transformers pymupdf
```

## 5. Add Your API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can get a key from https://console.groq.com

## 6. Run the App

```bash
python main.py
```

This starts:

- a FastAPI backend at `http://127.0.0.1:8123`
- a floating desktop overlay window powered by PyWebView

## 7. macOS Permissions

If you are on macOS, you may need to allow:

- Screen Recording for your Python app or terminal
- Microphone access for voice input

If screen capture fails, check:

`System Settings -> Privacy & Security -> Screen Recording`

## 8. What the App Does

At a high level, the app works like this:

1. The UI sends a question to the backend.
2. The backend captures the current screen.
3. The agent decides how to answer:
   - vision for screen-based questions
   - web search for live/current questions
   - voice flow when audio is recorded
4. The answer is returned to the overlay.

## 9. Code Structure

### Entry Point

- [main.py](/Users/spotta/Desktop/Projects/workshop-cluely/main.py)
  Starts FastAPI, launches the floating overlay, handles `/ask` and `/voice`, and wires together capture, recording, and the agent.

### Core Assistant Modules

- [assistant/agent.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/agent.py)
  The starter workshop agent. It currently handles the base routing flow and is the one used by `main.py`.

- [assistant/solution_agent.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/solution_agent.py)
  A completed reference implementation with all workshop stages, including classification, memory, RAG, and tool actions.

- [assistant/capture.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/capture.py)
  Captures the current screen and returns PNG bytes plus metadata like width, height, and timestamp.

- [assistant/vision.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/vision.py)
  Sends a screenshot and prompt to Groq's vision-capable chat API.

- [assistant/search.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/search.py)
  Sends current-information questions to Groq's web-enabled model.

- [assistant/speech.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/speech.py)
  Transcribes recorded audio using Groq's transcription API.

- [assistant/recorder.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/recorder.py)
  Records microphone audio locally and returns WAV bytes.

- [assistant/overlay/index.html](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/overlay/index.html)
  The simple frontend for the floating overlay.

### Output / Workspace Folders

- `notes/`
  Used by the tool-calling helper to save notes created by the assistant.

- `reminders.json`
  Created when the reminder tool is used.

- `.chroma_db/`
  Created if you use the RAG extension.

## 11. Example Questions To Try

Vision:

- "What app is open on my screen?"
- "What color is this header?"
- "Summarize what you see."

Search:

- "What's the weather today?"
- "What is the latest OpenAI news?"

Voice:

- Start recording from the overlay and ask a screen-related question out loud.

## 12. Run Tests

The repo includes tests for the completed agent:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test*.py' -v
```

## 13. Common Troubleshooting

### `Missing GROQ_API_KEY in .env`

Create the `.env` file in the project root and add your API key.

### Screen capture errors on macOS

Grant Screen Recording permission to Python, your terminal, or the app used to launch Python.

### Microphone recording does not work

Grant microphone permission and make sure your default input device is available.

### `ModuleNotFoundError`

Make sure your virtual environment is activated and the dependencies were installed with `pip install ...`.
