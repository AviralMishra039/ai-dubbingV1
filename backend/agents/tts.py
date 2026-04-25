"""
TTS Agent — Gemini 3.1 Flash TTS + Google Cloud TTS Fallback
LangGraph node name: "tts"

Generates speech audio for each translated segment.
Primary: Gemini 3.1 Flash TTS API via REST.
Fallback: Google Cloud Text-to-Speech (on error).
"""

import logging
import base64
from pathlib import Path
from typing import Optional

import requests

from backend.state import DubbingState
from backend.config import (
    GEMINI_API_KEY,
    DURATION_STRETCH_THRESHOLD,
)
from backend.utils.ffmpeg import get_duration
from backend.utils.gpu import clear_gpu_memory

logger = logging.getLogger(__name__)

# ── Voice Mappings ──────────────────────────────────────────────────────

GEMINI_VOICES = {
    "male": "Puck",     # Other options: Charon, Fenrir
    "female": "Kore",   # Other options: Aoede
}

GOOGLE_TTS_VOICES = {
    "en": {
        "male": "en-US-Neural2-D",
        "female": "en-US-Neural2-F",
    },
    "hi": {
        "male": "hi-IN-Neural2-B",
        "female": "hi-IN-Neural2-D",
    },
}

def _get_speaker_gender(speaker_profiles: dict, speaker_id: str) -> str:
    """Get the gender for a speaker from profiles."""
    profile = speaker_profiles.get(speaker_id, {})
    gender = profile.get("gender", "male").lower()
    return gender if gender in ("male", "female") else "male"


# ── Gemini TTS ──────────────────────────────────────────────────────────

def _generate_gemini_tts(
    text: str,
    gender: str,
    output_path: str,
) -> Optional[str]:
    """Generate audio using Gemini 3.1 Flash TTS API via REST.
    
    Returns:
        Path to the generated audio file, or None on failure.
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not configured")
        return None

    voice_name = GEMINI_VOICES.get(gender, GEMINI_VOICES["male"])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=45)

        if response.status_code == 429:
            logger.warning("Gemini TTS quota exceeded (429) — falling back")
            return None

        if response.status_code != 200:
            logger.warning(
                f"Gemini TTS error {response.status_code}: "
                f"{response.text[:200]}"
            )
            return None

        data = response.json()
        
        # Extract base64 audio
        try:
            audio_base64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            audio_bytes = base64.b64decode(audio_base64)
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to parse Gemini TTS response: {e}")
            return None

        # Gemini returns raw PCM audio (audio/l16; rate=24000; channels=1)
        # We save it to a .pcm file and convert it to .wav using FFmpeg
        pcm_path = output_path.replace(".wav", ".pcm")
        with open(pcm_path, "wb") as f:
            f.write(audio_bytes)

        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "s16le",       # raw 16-bit PCM
                "-ar", "24000",      # 24kHz
                "-ac", "1",          # mono
                "-i", pcm_path,
                "-filter:a", "volume=4.0",  # Boost quiet Gemini audio
                "-acodec", "pcm_s16le",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"Gemini PCM→WAV conversion failed: {result.stderr}")
            return None

        # Clean up PCM file
        Path(pcm_path).unlink(missing_ok=True)

        logger.info(f"Gemini TTS generated: {output_path}")
        return output_path

    except requests.exceptions.Timeout:
        logger.warning("Gemini TTS request timed out")
        return None
    except Exception as e:
        logger.warning(f"Gemini TTS failed: {e}")
        return None


# ── Google Cloud TTS ────────────────────────────────────────────────────

def _generate_google_tts(
    text: str,
    target_language: str,
    gender: str,
    output_path: str,
    ssml_hints: Optional[str] = None,
) -> Optional[str]:
    """Generate audio using Google Cloud Text-to-Speech."""
    try:
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()

        if ssml_hints and "<prosody" in ssml_hints:
            ssml_text = f"<speak>{ssml_hints}{text}</prosody></speak>"
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        else:
            synthesis_input = texttospeech.SynthesisInput(text=text)

        lang_code_map = {"en": "en-US", "hi": "hi-IN"}
        lang_code = lang_code_map.get(target_language, "en-US")
        
        lang_voices = GOOGLE_TTS_VOICES.get(target_language, GOOGLE_TTS_VOICES["en"])
        voice_name = lang_voices.get(gender, lang_voices["male"])

        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with open(output_path, "wb") as f:
            f.write(response.audio_content)

        logger.info(f"Google TTS generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Google TTS failed: {e}")
        return None


# ── Main TTS Agent ──────────────────────────────────────────────────────

def tts_agent(state: DubbingState) -> dict:
    """Generate TTS audio for all translated segments.
    
    Tries Gemini 3.1 Flash TTS first; falls back to Google Cloud TTS on failure.
    """
    logger.info("=== TTS AGENT START ===")

    translated_segments = state["translated_segments"]
    speaker_profiles = state.get("speaker_profiles", {})
    target_language = state["target_language"]
    temp_dir = state["temp_dir"]
    warnings = list(state.get("warnings", []))
    retry_count = state.get("retry_count", 0)

    segments_dir = str(Path(temp_dir) / "segments")
    Path(segments_dir).mkdir(parents=True, exist_ok=True)

    gemini_tts_exhausted = False
    audio_segments = []

    for idx, seg in enumerate(translated_segments):
        seg_id = seg["id"]
        speaker_id = seg["speaker_id"]
        translated_text = seg["translated"]
        ssml_hints = seg.get("ssml_hints", "")
        original_duration = seg["end"] - seg["start"]
        
        # Smart Overlapping: Calculate how much silence exists until the NEXT person talks
        available_duration = original_duration
        if idx + 1 < len(translated_segments):
            next_start = translated_segments[idx + 1]["start"]
            if next_start > seg["start"]:
                available_duration = next_start - seg["start"]
        else:
            available_duration = original_duration + 5.0  # Give last segment extra breathing room

        output_path = str(
            Path(segments_dir) / f"segment_{seg_id}_{speaker_id}.wav"
        )

        gender = _get_speaker_gender(speaker_profiles, speaker_id)

        logger.info(
            f"TTS segment {seg_id}: speaker={speaker_id}, gender={gender}, "
            f"text='{translated_text[:40]}...'"
        )

        generated_path = None

        # Try Gemini TTS first
        if not gemini_tts_exhausted:
            generated_path = _generate_gemini_tts(
                text=translated_text,
                gender=gender,
                output_path=output_path,
            )
            if generated_path is None:
                gemini_tts_exhausted = True
                warnings.append(
                    f"Gemini TTS failed at segment {seg_id} — "
                    f"switching to Google TTS fallback for remaining segments"
                )

        # Fallback to Google TTS
        if generated_path is None:
            generated_path = _generate_google_tts(
                text=translated_text,
                target_language=target_language,
                gender=gender,
                output_path=output_path,
                ssml_hints=ssml_hints,
            )

        if generated_path is None:
            logger.error(f"Both TTS providers failed for segment {seg_id}")
            warnings.append(f"TTS failed for segment {seg_id} — skipping")
            continue

        # Measure actual duration
        try:
            actual_duration = get_duration(generated_path)
        except Exception as e:
            logger.warning(f"Could not measure duration of {generated_path}: {e}")
            actual_duration = original_duration

        needs_stretch = (
            available_duration > 0
            and actual_duration > available_duration * DURATION_STRETCH_THRESHOLD
        )

        audio_segments.append({
            "id": seg_id,
            "start": seg["start"],
            "end": seg["end"],
            "file_path": generated_path,
            "duration": actual_duration,
            "original_duration": original_duration,
            "available_duration": available_duration,
            "speaker_id": speaker_id,
            "needs_stretch": needs_stretch,
        })

        if needs_stretch:
            logger.warning(
                f"Segment {seg_id} is {actual_duration:.2f}s vs "
                f"{available_duration:.2f}s available — flagged for stretch"
            )

    if not audio_segments:
        raise RuntimeError(
            "TTS produced zero audio segments — both Gemini TTS and "
            "Google TTS fallback failed for all segments"
        )

    logger.info(
        f"TTS complete: {len(audio_segments)}/{len(translated_segments)} "
        f"segments generated"
    )

    clear_gpu_memory()

    return {
        "audio_segments": audio_segments,
        "current_step": "tts_complete",
        "warnings": warnings,
    }
