"""
DubGraph Configuration
All environment variables loaded and validated in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


# ── API Keys ────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
PYANNOTE_AUTH_TOKEN: str = os.getenv("PYANNOTE_AUTH_TOKEN", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# ── ElevenLabs Voice IDs ────────────────────────────────────────────────
ELEVENLABS_VOICE_EN_MALE: str = os.getenv("ELEVENLABS_VOICE_EN_MALE", "")
ELEVENLABS_VOICE_EN_FEMALE: str = os.getenv("ELEVENLABS_VOICE_EN_FEMALE", "")
ELEVENLABS_VOICE_HI_MALE: str = os.getenv("ELEVENLABS_VOICE_HI_MALE", "")
ELEVENLABS_VOICE_HI_FEMALE: str = os.getenv("ELEVENLABS_VOICE_HI_FEMALE", "")

# ── Model Settings ──────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
GEMINI_MODEL: str = "gemini-2.5-flash"

# ── Directories ─────────────────────────────────────────────────────────
TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "./temp")).resolve()
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Upload Limits ───────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = 500
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ── Pipeline Settings ──────────────────────────────────────────────────
MAX_RETRY_COUNT: int = 2
DURATION_STRETCH_THRESHOLD: float = 1.5   # flag segment if TTS > 1.5x original

# ── Accepted video MIME types ───────────────────────────────────────────
ACCEPTED_VIDEO_MIMES: set = {
    "video/mp4",
    "video/x-matroska",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}

ACCEPTED_VIDEO_EXTENSIONS: set = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}
