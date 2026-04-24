"""
Transcription Agent — Demucs + Pyannote + Faster-Whisper
LangGraph node name: "transcription"

This is the most GPU-intensive agent. It runs three models sequentially,
clearing VRAM between each to stay within the 6 GB budget:
  1. Demucs (vocal separation)
  2. Pyannote (speaker diarization)
  3. Faster-Whisper (speech-to-text)
"""

import logging
from pathlib import Path
from typing import Optional

from backend.state import DubbingState
from backend.config import (
    WHISPER_MODEL_SIZE,
    PYANNOTE_AUTH_TOKEN,
)
from backend.utils.ffmpeg import extract_audio
from backend.utils.gpu import clear_gpu_memory, is_cuda_available, get_compute_type

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Step 1: Background separation with Demucs
# ──────────────────────────────────────────────────────────────────────────

def _run_demucs(audio_path: str, output_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Separate vocals from background using Demucs htdemucs model.
    
    Args:
        audio_path: Path to the raw audio WAV file.
        output_dir: Directory where Demucs will save separated tracks.
    
    Returns:
        Tuple of (vocals_path, background_path), either may be None on failure.
    """
    try:
        import torch
        import demucs.separate

        logger.info("Loading Demucs htdemucs model...")

        # Run demucs separation
        demucs.separate.main([
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "--out", output_dir,
            audio_path,
        ])

        # Demucs saves to: output_dir/htdemucs/<filename_without_ext>/vocals.wav
        audio_name = Path(audio_path).stem
        demucs_out = Path(output_dir) / "htdemucs" / audio_name

        vocals_path = demucs_out / "vocals.wav"
        no_vocals_path = demucs_out / "no_vocals.wav"

        vocals = str(vocals_path) if vocals_path.exists() else None
        background = str(no_vocals_path) if no_vocals_path.exists() else None

        if vocals:
            logger.info(f"Demucs vocals extracted: {vocals}")
        else:
            logger.warning("Demucs did not produce vocals track")

        return vocals, background

    except Exception as e:
        logger.warning(f"Demucs separation failed: {e}")
        return None, None

    finally:
        # CRITICAL: Free VRAM before next model
        try:
            import torch
            if torch.cuda.is_available():
                # Force cleanup of any demucs model references
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Demucs VRAM cleared")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Step 2: Speaker diarization with pyannote
# ──────────────────────────────────────────────────────────────────────────

def _run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> dict:
    """Run speaker diarization using pyannote.audio.
    
    Args:
        audio_path: Path to the audio file (preferably vocals-only).
        num_speakers: Optional hint for expected number of speakers.
    
    Returns:
        Dict of {speaker_id: [(start, end), ...]} with time intervals.
    """
    pipeline = None
    try:
        from pyannote.audio import Pipeline
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading pyannote diarization pipeline on {device}...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=PYANNOTE_AUTH_TOKEN,
        )
        pipeline = pipeline.to(device)

        # Run diarization
        kwargs = {}
        if num_speakers and num_speakers > 0:
            kwargs["num_speakers"] = num_speakers

        logger.info("Running speaker diarization...")
        diarization = pipeline(audio_path, **kwargs)

        # Parse results into {speaker_id: [(start, end), ...]}
        speaker_segments: dict[str, list[tuple[float, float]]] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))

        logger.info(
            f"Diarization complete: {len(speaker_segments)} speakers detected"
        )
        return speaker_segments

    except Exception as e:
        logger.warning(f"Diarization failed: {e}")
        return {}

    finally:
        # CRITICAL: Free VRAM before next model
        del pipeline
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Pyannote VRAM cleared")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Step 3: Speech-to-text with faster-whisper
# ──────────────────────────────────────────────────────────────────────────

def _run_whisper(audio_path: str, language: str) -> list[dict]:
    """Transcribe audio using faster-whisper.
    
    Args:
        audio_path: Path to the audio file.
        language: Source language code ('en' or 'hi').
    
    Returns:
        List of dicts: [{start, end, text}, ...]
    """
    model = None
    try:
        from faster_whisper import WhisperModel

        compute_type = get_compute_type()
        device = "cuda" if is_cuda_available() else "cpu"

        logger.info(
            f"Loading faster-whisper {WHISPER_MODEL_SIZE} on {device} "
            f"({compute_type})..."
        )
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )

        logger.info(f"Transcribing with language={language}...")
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        raw_segments = []
        for seg in segments_iter:
            raw_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        logger.info(
            f"Whisper transcription complete: {len(raw_segments)} segments, "
            f"language={info.language}, probability={info.language_probability:.2f}"
        )
        return raw_segments

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise

    finally:
        # CRITICAL: Free VRAM
        del model
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Whisper VRAM cleared")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Step 4: Merge transcription + diarization
# ──────────────────────────────────────────────────────────────────────────

def _find_speaker_for_segment(
    seg_start: float,
    seg_end: float,
    speaker_segments: dict[str, list[tuple[float, float]]],
) -> str:
    """Match a transcription segment to a speaker via timestamp overlap.
    
    Uses the maximum overlap heuristic: the speaker whose diarization
    intervals overlap most with [seg_start, seg_end] wins.
    
    Args:
        seg_start: Segment start time in seconds.
        seg_end: Segment end time in seconds.
        speaker_segments: Dict from diarization {speaker_id: [(start, end), ...]}.
    
    Returns:
        The speaker_id with maximum overlap, or 'SPEAKER_00' if none found.
    """
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0

    for speaker_id, intervals in speaker_segments.items():
        total_overlap = 0.0
        for interval_start, interval_end in intervals:
            overlap_start = max(seg_start, interval_start)
            overlap_end = min(seg_end, interval_end)
            overlap = max(0.0, overlap_end - overlap_start)
            total_overlap += overlap

        if total_overlap > best_overlap:
            best_overlap = total_overlap
            best_speaker = speaker_id

    return best_speaker


def _merge_transcription_diarization(
    whisper_segments: list[dict],
    speaker_segments: dict[str, list[tuple[float, float]]],
) -> list[dict]:
    """Merge whisper transcription segments with pyannote speaker labels.
    
    Args:
        whisper_segments: List of {start, end, text} from whisper.
        speaker_segments: Dict of {speaker_id: [(start, end), ...]} from pyannote.
    
    Returns:
        List of {id, start, end, text, speaker_id}.
    """
    merged = []
    for idx, seg in enumerate(whisper_segments):
        if not seg["text"]:
            continue  # skip empty segments

        if speaker_segments:
            speaker_id = _find_speaker_for_segment(
                seg["start"], seg["end"], speaker_segments
            )
        else:
            speaker_id = "SPEAKER_00"

        merged.append({
            "id": idx,
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"],
            "speaker_id": speaker_id,
        })

    return merged


# ──────────────────────────────────────────────────────────────────────────
# Main agent function
# ──────────────────────────────────────────────────────────────────────────

def transcription_agent(state: DubbingState) -> dict:
    """Full transcription pipeline: extract audio → demucs → pyannote → whisper → merge.
    
    This is a CRITICAL agent. If transcription fails entirely, the
    pipeline cannot continue.
    
    Args:
        state: Current pipeline state.
    
    Returns:
        Dict of state updates: segments, vocals_path, background_path.
    
    Raises:
        RuntimeError: If whisper transcription produces no output.
    """
    logger.info("=== TRANSCRIPTION AGENT START ===")

    video_path = state["video_path"]
    source_language = state["source_language"]
    speaker_count = state.get("speaker_count", 1)
    temp_dir = state["temp_dir"]
    warnings = list(state.get("warnings", []))

    audio_dir = str(Path(temp_dir) / "audio")
    Path(audio_dir).mkdir(parents=True, exist_ok=True)

    raw_audio_path = str(Path(audio_dir) / "raw_audio.wav")

    # ── Step 1: Extract audio from video ────────────────────────────────
    logger.info("Step 1: Extracting audio from video...")
    extract_audio(video_path, raw_audio_path)

    # ── Step 2: Separate vocals with Demucs ─────────────────────────────
    logger.info("Step 2: Running Demucs vocal separation...")
    demucs_out = str(Path(temp_dir) / "demucs")
    vocals_path, background_path = _run_demucs(raw_audio_path, demucs_out)

    # Use vocals for transcription if available, else fall back to raw audio
    transcription_audio = vocals_path if vocals_path else raw_audio_path
    if not vocals_path:
        warnings.append("Demucs failed — using raw audio for transcription")

    # ── Step 3: Speaker diarization ─────────────────────────────────────
    logger.info("Step 3: Running speaker diarization...")
    speaker_segments = _run_diarization(
        transcription_audio,
        num_speakers=speaker_count if speaker_count > 1 else None,
    )

    if not speaker_segments:
        warnings.append(
            "Diarization failed or found no speakers — "
            "all segments assigned to SPEAKER_00"
        )

    # ── Step 4: Transcribe with faster-whisper ──────────────────────────
    logger.info("Step 4: Running faster-whisper transcription...")
    whisper_segments = _run_whisper(transcription_audio, source_language)

    if not whisper_segments:
        raise RuntimeError(
            "Whisper produced zero transcription segments. "
            "The audio may be silent or corrupted."
        )

    # ── Step 5: Merge transcription + diarization ───────────────────────
    logger.info("Step 5: Merging transcription with diarization...")
    merged_segments = _merge_transcription_diarization(
        whisper_segments, speaker_segments
    )

    if not merged_segments:
        raise RuntimeError(
            "Merge produced zero segments after filtering empty text."
        )

    logger.info(f"Transcription complete: {len(merged_segments)} segments")

    return {
        "segments": merged_segments,
        "vocals_path": vocals_path,
        "background_path": background_path,
        "current_step": "transcription_complete",
        "warnings": warnings,
    }
