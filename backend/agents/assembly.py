"""
Assembly Agent — FFmpeg Audio Assembly & Video Merge
LangGraph node name: "assembly"

Assembles the final dubbed video using only FFmpeg (no GPU).
Steps:
  1. Fix each segment duration (pad/stretch)
  2. Position each segment at exact timestamp
  3. Create silent base track
  4. Mix all delayed segments onto base
  5. Mix with Demucs background audio
  6. Merge with original video
  7. Cleanup temp files
"""

import logging
from pathlib import Path

from backend.state import DubbingState
from backend.config import OUTPUT_DIR
from backend.utils.ffmpeg import (
    pad_audio_with_silence,
    stretch_audio,
    delay_audio,
    create_silent_track,
    mix_audio_tracks,
    merge_video_audio,
    get_video_duration,
    get_duration,
)
from backend.utils.cleanup import cleanup_job_temp

logger = logging.getLogger(__name__)


def assembly_agent(state: DubbingState) -> dict:
    """Assemble the final dubbed video from TTS audio segments.
    
    This agent uses ONLY FFmpeg subprocess calls — zero GPU usage.
    
    Args:
        state: Current pipeline state.
    
    Returns:
        Dict of state updates: final_video_path.
    
    Raises:
        RuntimeError: If assembly produces no output video.
    """
    logger.info("=== ASSEMBLY AGENT START ===")

    audio_segments = state["audio_segments"]
    video_path = state["video_path"]
    background_path = state.get("background_path")
    job_id = state["job_id"]
    temp_dir = state["temp_dir"]
    warnings = list(state.get("warnings", []))

    assembly_dir = str(Path(temp_dir) / "assembly")
    Path(assembly_dir).mkdir(parents=True, exist_ok=True)

    # Get total video duration for the silent base track
    total_duration = get_video_duration(video_path)
    logger.info(f"Video duration: {total_duration:.2f}s")

    # ── Step 1: Fix each segment duration ───────────────────────────────
    logger.info("Step 1: Fixing segment durations...")
    fixed_paths = []

    for seg in audio_segments:
        seg_id = seg["id"]
        input_path = seg["file_path"]
        actual_dur = seg["duration"]
        original_dur = seg["original_duration"]
        needs_stretch = seg.get("needs_stretch", False)

        fixed_path = str(Path(assembly_dir) / f"fixed_{seg_id}.wav")

        if actual_dur < original_dur:
            # Pad with silence to match original duration
            logger.info(
                f"Segment {seg_id}: padding {actual_dur:.2f}s → "
                f"{original_dur:.2f}s"
            )
            pad_audio_with_silence(input_path, fixed_path, original_dur)

        elif needs_stretch and actual_dur > original_dur:
            # Speed up to fit original duration
            ratio = actual_dur / original_dur
            logger.info(
                f"Segment {seg_id}: stretching {actual_dur:.2f}s → "
                f"{original_dur:.2f}s (ratio={ratio:.2f})"
            )
            stretch_audio(input_path, fixed_path, ratio)

        else:
            # Use as-is (duration is acceptable)
            logger.info(f"Segment {seg_id}: duration OK ({actual_dur:.2f}s)")
            # Copy to assembly dir for consistency
            import shutil
            shutil.copy2(input_path, fixed_path)

        fixed_paths.append({
            "id": seg_id,
            "path": fixed_path,
            "start": seg["start"],
        })

    # ── Step 2: Position each segment at exact timestamp ────────────────
    logger.info("Step 2: Positioning segments at timestamps...")
    delayed_paths = []

    for item in fixed_paths:
        seg_id = item["id"]
        start_ms = int(item["start"] * 1000)
        delayed_path = str(Path(assembly_dir) / f"delayed_{seg_id}.wav")

        if start_ms > 0:
            delay_audio(item["path"], delayed_path, start_ms)
        else:
            # No delay needed for segments starting at 0
            import shutil
            shutil.copy2(item["path"], delayed_path)

        delayed_paths.append(delayed_path)

    # ── Step 3: Create silent base track ────────────────────────────────
    logger.info("Step 3: Creating silent base track...")
    base_path = str(Path(assembly_dir) / "base.wav")
    create_silent_track(base_path, total_duration)

    # ── Step 4: Mix all delayed segments onto base track ────────────────
    logger.info(f"Step 4: Mixing {len(delayed_paths)} segments onto base track...")
    dubbed_voice_path = str(Path(assembly_dir) / "dubbed_voice.wav")

    all_inputs = [base_path] + delayed_paths
    mix_audio_tracks(all_inputs, dubbed_voice_path)

    # ── Step 5: Mix with background audio (from Demucs) ─────────────────
    logger.info("Step 5: Mixing with background audio...")
    final_audio_path = str(Path(assembly_dir) / "final_audio.wav")

    if background_path and Path(background_path).exists():
        logger.info(f"Mixing dubbed voice with background: {background_path}")
        mix_audio_tracks(
            [dubbed_voice_path, background_path],
            final_audio_path,
            weights="1 0.7",
        )
    else:
        logger.info("No background track available — using dubbed voice only")
        import shutil
        shutil.copy2(dubbed_voice_path, final_audio_path)
        if not background_path:
            warnings.append("No background audio from Demucs — voice-only output")

    # ── Step 6: Merge with original video ───────────────────────────────
    logger.info("Step 6: Merging audio with original video...")
    output_filename = f"{job_id}_dubbed.mp4"
    final_video_path = str(OUTPUT_DIR / output_filename)

    merge_video_audio(video_path, final_audio_path, final_video_path)

    if not Path(final_video_path).exists():
        raise RuntimeError(
            f"Assembly failed — output video not found at {final_video_path}"
        )

    file_size_mb = Path(final_video_path).stat().st_size / (1024 * 1024)
    logger.info(
        f"Assembly complete: {final_video_path} ({file_size_mb:.1f} MB)"
    )

    # ── Step 7: Cleanup temp files ──────────────────────────────────────
    logger.info("Step 7: Cleaning up temp files...")
    try:
        cleanup_job_temp(temp_dir)
    except Exception as e:
        logger.warning(f"Cleanup failed (non-critical): {e}")

    return {
        "final_video_path": final_video_path,
        "current_step": "assembly_complete",
        "warnings": warnings,
    }
