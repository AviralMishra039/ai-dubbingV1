# DubGraph — AI-Powered Multi-Agent Video Dubbing

A LangGraph-orchestrated multi-agent pipeline for dubbing videos between **Hindi** and **English**. Upload a video, select a direction, and the system handles analysis, transcription, translation, voice synthesis, and assembly automatically.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                       │
│                                                             │
│  ┌──────────┐  ┌───────────────┐  ┌─────────────┐          │
│  │ Analysis  │→│ Transcription │→│ Translation  │          │
│  │ (Gemini)  │  │(Demucs+Whis) │  │  (Gemini)   │          │
│  └──────────┘  └───────────────┘  └──────┬──────┘          │
│                                          ↓                  │
│  ┌──────────┐  ┌───────────────┐  ┌─────────────┐          │
│  │ Assembly  │←│   Director    │←│    TTS       │          │
│  │ (FFmpeg)  │  │  (Routing)   │  │(ElevenLabs) │          │
│  └──────────┘  └───────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Agents:**
| Node | Responsibility | Tech |
|------|---------------|------|
| Analysis | Scene context, speaker profiles | Gemini 2.5 Flash Vision |
| Transcription | Speech-to-text + diarization | Demucs + pyannote + faster-whisper |
| Translation | Context-aware dubbing translation | Gemini 2.5 Flash |
| TTS | Voice synthesis | ElevenLabs (primary) / Google TTS (fallback) |
| Assembly | Audio mixing + video merge | FFmpeg (CPU only) |
| Director | Routing & validation | Conditional edges (no LLM) |

## Prerequisites

- **OS:** WSL2 Ubuntu (Windows) or native Linux
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4050 6GB)
- **System packages:** FFmpeg, git-lfs
- **Python:** 3.10+
- **Node.js:** 18+

### Install System Dependencies

```bash
# FFmpeg
sudo apt update && sudo apt install ffmpeg

# git-lfs (needed for pyannote model downloads)
sudo apt install git-lfs
git lfs install
```

## Setup

### 1. Install PyTorch with CUDA (WSL2)

**Do this first** — PyTorch must be installed with CUDA support before the other ML packages.

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Verify CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

### 2. Install Python Dependencies

```bash
cd backend
uv pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

#### Gemini API Key
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Create an API key for Gemini 2.5 Flash

#### Pyannote Auth Token
- Go to [Hugging Face](https://huggingface.co/settings/tokens)
- Create a token with `read` access
- Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

#### ElevenLabs API Key + Voice IDs
- Sign up at [ElevenLabs](https://elevenlabs.io)
- Get your API key from Profile → API Keys
- **Finding Voice IDs for Hindi:**
  1. Go to [ElevenLabs Voice Library](https://elevenlabs.io/voice-library)
  2. Search for "Hindi" in the voice library
  3. Click on a voice → the URL will contain the voice ID
  4. Or use the API: `GET https://api.elevenlabs.io/v1/voices` with your API key
  5. Look for voices tagged with Hindi language support
  6. Copy the `voice_id` values into your `.env`

#### Google Cloud TTS (Fallback)
- Create a service account at [Google Cloud Console](https://console.cloud.google.com/)
- Enable the Text-to-Speech API
- Download the JSON credentials file
- Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of this file

### 5. Whisper Model Size

Set `WHISPER_MODEL_SIZE` in `.env`. Options:
- `tiny` — fastest, lowest accuracy
- `base` — fast
- `small` — balanced
- `medium` — recommended (default)
- `large-v3` — best accuracy, highest VRAM (~5GB)

## Running

### Backend

```bash
# From the project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Development)

```bash
cd frontend
npm run dev
```

The Vite dev server proxies API calls to `localhost:8000`.

### Access

Open **http://localhost:5173** (Vite dev) or **http://localhost:8000** (production with built frontend).

## VRAM Management

The pipeline enforces strict sequential GPU usage for the 6GB VRAM budget:

1. **Demucs** loads → runs → `del model` → `torch.cuda.empty_cache()`
2. **Pyannote** loads → runs → `del pipeline` → `torch.cuda.empty_cache()`
3. **Faster-Whisper** loads → runs → `del model` → `torch.cuda.empty_cache()`

No two models are ever loaded simultaneously.

## Project Structure

```
dubbing-agent/
├── backend/
│   ├── main.py                  # FastAPI app + endpoints
│   ├── graph.py                 # LangGraph graph definition
│   ├── state.py                 # DubbingState TypedDict
│   ├── config.py                # Environment variable loading
│   ├── agents/
│   │   ├── director.py          # Routing / conditional edges
│   │   ├── analysis.py          # Gemini Vision scene analysis
│   │   ├── transcription.py     # Demucs + pyannote + whisper
│   │   ├── translation.py       # Gemini translation + SSML
│   │   ├── tts.py               # ElevenLabs + Google TTS
│   │   └── assembly.py          # FFmpeg audio + video merge
│   ├── utils/
│   │   ├── ffmpeg.py            # FFmpeg subprocess wrappers
│   │   ├── gpu.py               # CUDA utilities
│   │   └── cleanup.py           # Temp file cleanup
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app with view state
│   │   ├── index.css            # Design system
│   │   └── components/
│   │       ├── Uploader.jsx     # Upload + direction selector
│   │       ├── Processing.jsx   # Progress view
│   │       └── Result.jsx       # Video player + download
│   └── package.json
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT
