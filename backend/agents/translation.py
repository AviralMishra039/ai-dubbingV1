"""
Translation Agent — Gemini-Powered Dubbing Translation
LangGraph node name: "translation"

Translates each transcription segment individually via Gemini,
preserving spoken rhythm, emotional register, and cultural context.
Also generates SSML prosody hints for the TTS agent.
"""

import json
import logging
import re

import google.generativeai as genai

from backend.state import DubbingState
from backend.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
}

SYSTEM_PROMPT = """You are a professional dubbing translator specializing in Hindi and English cinema. You translate dialogue for dubbing, which means the translated text will be spoken aloud by a voice actor and must sound completely natural when spoken, not when read.

Scene context: {scene_context}
Speaker profile: {speaker_profile}

Translation rules:
- CRITICAL: The translated text MUST be extremely concise. Do not use more syllables than the original English sentence. If you must summarize the meaning to keep it short, do so.
- Preserve natural spoken rhythm — translated text should take approximately the same time or LESS to speak as the original
- Match the emotional register exactly: angry stays angry, whispered stays intimate, shouted stays intense
- Keep proper nouns, character names, place names as-is
- Do NOT translate literally word for word
- If a phrase has a cultural equivalent in the target language, use it instead of a literal translation
- After the translation, on a new line provide SSML hints in this format:
  SSML: <prosody rate="X" pitch="Y">
  where X is slow/medium/fast and Y is low/medium/high based on the speaker's emotion

Respond with exactly two lines:
Line 1: The translated text only
Line 2: SSML: <prosody rate="X" pitch="Y">"""


def _get_speaker_profile_str(speaker_profiles: dict, speaker_id: str) -> str:
    """Get a human-readable speaker profile string.
    
    Args:
        speaker_profiles: Dict of speaker_id -> {gender, emotion, style}.
        speaker_id: The speaker to look up.
    
    Returns:
        Formatted string describing the speaker.
    """
    profile = speaker_profiles.get(speaker_id, {})
    if not profile:
        return "Unknown speaker — use neutral conversational tone"

    gender = profile.get("gender", "unknown")
    emotion = profile.get("emotion", "neutral")
    style = profile.get("style", "conversational")

    return f"Gender: {gender}, Emotion: {emotion}, Style: {style}"


def _parse_translation_response(response_text: str) -> tuple[str, str]:
    """Parse Gemini response into translated text and SSML hints.
    
    Expected format:
        <translated text>
        SSML: <prosody rate="medium" pitch="medium">
    
    Args:
        response_text: Raw text from Gemini.
    
    Returns:
        Tuple of (translated_text, ssml_hints).
    """
    lines = response_text.strip().split("\n")

    translated = ""
    ssml_hints = ""

    ssml_lines = []
    text_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("SSML:"):
            ssml_part = stripped[5:].strip()
            ssml_lines.append(ssml_part)
        else:
            text_lines.append(stripped)

    translated = " ".join(text_lines).strip()

    if ssml_lines:
        ssml_hints = ssml_lines[0]
    else:
        # Default SSML if Gemini didn't provide one
        ssml_hints = '<prosody rate="medium" pitch="medium">'

    # Clean up: remove any accidental quotes around the translation
    if translated.startswith('"') and translated.endswith('"'):
        translated = translated[1:-1]

    return translated, ssml_hints


def _translate_segment(
    segment: dict,
    source_language: str,
    target_language: str,
    scene_context: str,
    speaker_profiles: dict,
    model: genai.GenerativeModel,
) -> dict:
    """Translate a single segment via Gemini.
    
    Args:
        segment: Dict with {id, start, end, text, speaker_id}.
        source_language: Source language code.
        target_language: Target language code.
        scene_context: Scene context from analysis agent.
        speaker_profiles: Speaker profiles from analysis agent.
        model: Gemini GenerativeModel instance.
    
    Returns:
        Dict with original fields plus translated, ssml_hints.
    """
    speaker_id = segment["speaker_id"]
    profile_str = _get_speaker_profile_str(speaker_profiles, speaker_id)

    src_name = LANGUAGE_NAMES.get(source_language, source_language)
    tgt_name = LANGUAGE_NAMES.get(target_language, target_language)

    system_prompt = SYSTEM_PROMPT.format(
        scene_context=scene_context,
        speaker_profile=profile_str,
    )

    user_prompt = f"Translate this from {src_name} to {tgt_name}: {segment['text']}"

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]},
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )

        translated, ssml_hints = _parse_translation_response(response.text)

        if not translated:
            logger.warning(
                f"Empty translation for segment {segment['id']}, "
                f"using original text"
            )
            translated = segment["text"]

        return {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "original": segment["text"],
            "translated": translated,
            "speaker_id": speaker_id,
            "ssml_hints": ssml_hints,
        }

    except Exception as e:
        logger.error(f"Translation failed for segment {segment['id']}: {e}")
        # Fallback: return original text as "translation"
        return {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "original": segment["text"],
            "translated": segment["text"],
            "speaker_id": speaker_id,
            "ssml_hints": '<prosody rate="medium" pitch="medium">',
        }


def translation_agent(state: DubbingState) -> dict:
    """Translate all transcribed segments via Gemini.
    
    This is a CRITICAL agent. If it fails entirely, the pipeline
    cannot continue.
    
    Args:
        state: Current pipeline state.
    
    Returns:
        Dict of state updates: translated_segments.
    
    Raises:
        RuntimeError: If zero segments could be translated.
    """
    logger.info("=== TRANSLATION AGENT START ===")

    segments = state["segments"]
    source_language = state["source_language"]
    target_language = state["target_language"]
    scene_context = state.get("scene_context", "general content")
    speaker_profiles = state.get("speaker_profiles", {})

    logger.info(
        f"Translating {len(segments)} segments: "
        f"{LANGUAGE_NAMES.get(source_language)} → {LANGUAGE_NAMES.get(target_language)}"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    translated_segments = []

    for i, segment in enumerate(segments):
        logger.info(
            f"Translating segment {i + 1}/{len(segments)}: "
            f"'{segment['text'][:50]}...'"
        )

        translated = _translate_segment(
            segment=segment,
            source_language=source_language,
            target_language=target_language,
            scene_context=scene_context,
            speaker_profiles=speaker_profiles,
            model=model,
        )

        translated_segments.append(translated)

    if not translated_segments:
        raise RuntimeError("Translation produced zero translated segments")

    logger.info(f"Translation complete: {len(translated_segments)} segments translated")

    return {
        "translated_segments": translated_segments,
        "current_step": "translation_complete",
    }
