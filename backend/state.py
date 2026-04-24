"""
DubGraph State Definition
Shared TypedDict state that flows through all LangGraph nodes.
Every agent reads from and writes to this single state object.
"""

from typing import TypedDict, Optional


class DubbingState(TypedDict):
    """Central state object for the dubbing pipeline.
    
    This TypedDict is the single source of truth passed between
    all LangGraph nodes. Each agent reads its inputs and writes
    its outputs to fields in this state.
    """

    # ── Input (set by FastAPI before graph.invoke) ──────────────────────
    video_path: str                    # absolute path to uploaded video
    source_language: str               # "en" or "hi"
    target_language: str               # "en" or "hi"
    job_id: str                        # unique job identifier (uuid4)

    # ── Analysis Agent output ───────────────────────────────────────────
    scene_context: str                 # tone, mood, setting description
    speaker_count: int                 # number of detected speakers
    speaker_profiles: dict             # speaker_id -> {gender, emotion, style}

    # ── Transcription Agent output ──────────────────────────────────────
    segments: list                     # [{id, start, end, text, speaker_id}]

    # ── Translation Agent output ────────────────────────────────────────
    translated_segments: list          # [{id, start, end, original, translated,
                                       #   speaker_id, ssml_hints}]

    # ── TTS Agent output ────────────────────────────────────────────────
    audio_segments: list               # [{id, start, end, file_path, duration,
                                       #   original_duration, speaker_id,
                                       #   needs_stretch}]

    # ── Assembly Agent output ───────────────────────────────────────────
    final_video_path: str              # path to the finished dubbed video

    # ── Director / Pipeline metadata ────────────────────────────────────
    current_step: str                  # name of the step currently executing
    errors: list                       # list of error messages encountered
    warnings: list                     # list of non-fatal warning messages
    retry_count: int                   # number of TTS retries performed

    # ── Internal paths (set during pipeline) ────────────────────────────
    temp_dir: str                      # job-specific temp directory
    vocals_path: Optional[str]         # path to demucs-separated vocals
    background_path: Optional[str]     # path to demucs-separated background
