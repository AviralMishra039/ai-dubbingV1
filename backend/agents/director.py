"""
Director Agent — Graph Routing & Conditional Edges
Implements should_continue-style routing functions for the LangGraph graph.
The director does NOT call any LLM — it only inspects state and routes.
"""

import logging
from backend.state import DubbingState
from backend.config import MAX_RETRY_COUNT, DURATION_STRETCH_THRESHOLD

logger = logging.getLogger(__name__)


def validate_analysis(state: DubbingState) -> str:
    """Validate analysis agent output and decide next step.
    
    Returns:
        'transcription' to proceed, or 'transcription' with defaults
        if analysis had issues (analysis is non-critical).
    """
    scene_context = state.get("scene_context", "")
    speaker_count = state.get("speaker_count", 0)

    if not scene_context or speaker_count <= 0:
        logger.warning(
            "Analysis output incomplete — using defaults. "
            f"scene_context='{scene_context}', speaker_count={speaker_count}"
        )
        # Analysis is non-critical: we continue regardless
    else:
        logger.info(
            f"Analysis validated: {speaker_count} speakers, "
            f"context='{scene_context[:80]}...'"
        )

    return "transcription"


def validate_transcription(state: DubbingState) -> str:
    """Validate transcription agent output and decide next step.
    
    Returns:
        'translation' to proceed, or 'end' if transcription is empty
        (transcription is critical).
    """
    segments = state.get("segments", [])

    if not segments:
        logger.error("Transcription produced zero segments — cannot continue")
        return "end"

    logger.info(f"Transcription validated: {len(segments)} segments")
    return "translation"


def validate_translation(state: DubbingState) -> str:
    """Validate translation agent output and decide next step.
    
    Returns:
        'tts' to proceed, or 'end' if translation is empty
        (translation is critical).
    """
    translated = state.get("translated_segments", [])

    if not translated:
        logger.error("Translation produced zero segments — cannot continue")
        return "end"

    logger.info(f"Translation validated: {len(translated)} segments")
    return "tts"


def validate_tts(state: DubbingState) -> str:
    """Validate TTS agent output and decide: assembly or retry.
    
    After the TTS agent runs, the director checks each segment:
    - If audio_duration > original_duration * DURATION_STRETCH_THRESHOLD,
      flag it (needs_stretch = True).
    - If flagged segments exist AND retry_count < MAX_RETRY_COUNT,
      route back to TTS for retry with slower speech rate.
    - Otherwise, proceed to assembly.
    
    Returns:
        'tts' to retry, 'assembly' to proceed, or 'end' on failure.
    """
    audio_segments = state.get("audio_segments", [])

    if not audio_segments:
        logger.error("TTS produced zero audio segments — cannot continue")
        return "end"

    retry_count = state.get("retry_count", 0)

    # Flag segments that are too long
    flagged = []
    for seg in audio_segments:
        original_dur = seg.get("original_duration", 0)
        actual_dur = seg.get("duration", 0)
        if original_dur > 0 and actual_dur > original_dur * DURATION_STRETCH_THRESHOLD:
            seg["needs_stretch"] = True
            flagged.append(seg["id"])
        else:
            seg["needs_stretch"] = seg.get("needs_stretch", False)

    if flagged and retry_count < MAX_RETRY_COUNT:
        logger.warning(
            f"TTS segments too long ({len(flagged)} flagged): {flagged}. "
            f"Retrying (attempt {retry_count + 1}/{MAX_RETRY_COUNT})"
        )
        return "tts"

    if flagged:
        logger.warning(
            f"TTS segments still too long after {retry_count} retries. "
            f"Proceeding to assembly with stretch applied."
        )

    logger.info(f"TTS validated: {len(audio_segments)} segments ready for assembly")
    return "assembly"


def route_after_analysis(state: DubbingState) -> str:
    """Conditional edge function after analysis node."""
    return validate_analysis(state)


def route_after_transcription(state: DubbingState) -> str:
    """Conditional edge function after transcription node."""
    return validate_transcription(state)


def route_after_translation(state: DubbingState) -> str:
    """Conditional edge function after translation node."""
    return validate_translation(state)


def route_after_tts(state: DubbingState) -> str:
    """Conditional edge function after TTS node."""
    return validate_tts(state)


def increment_retry(state: DubbingState) -> dict:
    """Called when TTS is being retried — increments the retry counter."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "current_step": "tts_retry",
    }
