"""
FFmpeg Subprocess Wrappers
All FFmpeg and FFprobe calls go through this module.
Uses subprocess directly — no ffmpeg-python wrapper.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Raised when an FFmpeg/FFprobe command fails."""

    def __init__(self, command: str, returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"FFmpeg command failed (exit {returncode}):\n"
            f"Command: {command}\n"
            f"Stderr: {stderr}"
        )


def _run(cmd: list[str], description: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run a subprocess command with full error handling.
    
    Args:
        cmd: Command and arguments as a list of strings.
        description: Human-readable description for logging.
    
    Returns:
        CompletedProcess result on success.
    
    Raises:
        FFmpegError: If the command exits with a non-zero code.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"Running {description}: {cmd_str}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max per command
    )

    if result.returncode != 0:
        logger.error(f"{description} failed: {result.stderr}")
        raise FFmpegError(
            command=cmd_str,
            returncode=result.returncode,
            stderr=result.stderr,
        )

    return result


def extract_frames(video_path: str, output_dir: str, interval_seconds: int = 30) -> list[str]:
    """Extract one frame every N seconds from a video.
    
    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frame images.
        interval_seconds: Interval between frame captures.
    
    Returns:
        List of paths to extracted frame images.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "frame_%03d.jpg")

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps=1/{interval_seconds}",
            "-q:v", "2",
            pattern,
        ],
        description="extract_frames",
    )

    frames = sorted(out.glob("frame_*.jpg"))
    return [str(f) for f in frames]


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video as 16kHz mono WAV.
    
    Args:
        video_path: Path to the input video.
        output_path: Path for the output WAV file.
    
    Returns:
        Path to the extracted audio file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        description="extract_audio",
    )

    return output_path


def get_duration(file_path: str) -> float:
    """Get the duration of a media file in seconds using ffprobe.
    
    Args:
        file_path: Path to the audio or video file.
    
    Returns:
        Duration in seconds as a float.
    """
    result = _run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(file_path),
        ],
        description="get_duration",
    )

    return float(result.stdout.strip())


def get_video_duration(video_path: str) -> float:
    """Alias for get_duration, specifically for video files."""
    return get_duration(video_path)


def pad_audio_with_silence(input_path: str, output_path: str, target_duration: float) -> str:
    """Pad an audio file with silence to reach a target duration.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Path for the padded output file.
        target_duration: Desired total duration in seconds.
    
    Returns:
        Path to the padded audio file.
    """
    current_duration = get_duration(input_path)
    pad_dur = max(0, target_duration - current_duration)

    if pad_dur <= 0:
        # No padding needed — just copy
        _run(["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)],
             description="copy_audio")
        return output_path

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", f"apad=pad_dur={pad_dur:.3f}",
            "-ar", "24000",
            "-ac", "1",
            str(output_path),
        ],
        description="pad_audio",
    )

    return output_path


def stretch_audio(input_path: str, output_path: str, ratio: float) -> str:
    """Speed up audio using atempo filter to fit original duration.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Path for the stretched output file.
        ratio: Speed ratio (e.g. 1.3 means play 1.3x faster).
    
    Returns:
        Path to the stretched audio file.
    """
    if ratio <= 0:
        raise ValueError(f"Invalid atempo ratio: {ratio}")

    # atempo filter accepts values between 0.5 and 100.0
    # For ratios > 2.0, chain two atempo filters
    if ratio <= 2.0:
        af = f"atempo={ratio:.4f}"
    else:
        # Chain: first 2.0x, then remaining
        second = ratio / 2.0
        if second > 2.0:
            # For very large ratios, triple-chain
            third = second / 2.0
            af = f"atempo=2.0,atempo=2.0,atempo={third:.4f}"
        else:
            af = f"atempo=2.0,atempo={second:.4f}"

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", af,
            "-ar", "24000",
            "-ac", "1",
            str(output_path),
        ],
        description="stretch_audio",
    )

    return output_path


def delay_audio(input_path: str, output_path: str, delay_ms: int) -> str:
    """Delay audio by a given number of milliseconds using adelay filter.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Path for the delayed output file.
        delay_ms: Delay in milliseconds.
    
    Returns:
        Path to the delayed audio file.
    """
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", f"adelay={delay_ms}|{delay_ms}",
            "-ar", "24000",
            "-ac", "1",
            str(output_path),
        ],
        description="delay_audio",
    )

    return output_path


def create_silent_track(output_path: str, duration: float, sample_rate: int = 24000) -> str:
    """Create a silent audio track of a specified duration.
    
    Args:
        output_path: Path for the silent audio file.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
    
    Returns:
        Path to the silent audio file.
    """
    _run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", f"{duration:.3f}",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_path),
        ],
        description="create_silent_track",
    )

    return output_path


def mix_audio_tracks(input_paths: list[str], output_path: str, weights: Optional[str] = None) -> str:
    """Mix multiple audio tracks together using amix filter.
    
    Args:
        input_paths: List of paths to audio files to mix.
        output_path: Path for the mixed output file.
        weights: Optional weight string for amix (e.g. "1 0.7").
    
    Returns:
        Path to the mixed audio file.
    """
    cmd = ["ffmpeg", "-y"]
    for p in input_paths:
        cmd.extend(["-i", str(p)])

    n = len(input_paths)
    filter_str = f"amix=inputs={n}:duration=first:dropout_transition=0"
    if weights:
        filter_str = f"amix=inputs={n}:duration=first:weights={weights}"

    cmd.extend([
        "-filter_complex", filter_str,
        "-ar", "24000",
        "-ac", "1",
        str(output_path),
    ])

    _run(cmd, description="mix_audio")

    return output_path


def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Merge video with new audio track, copying video stream.
    
    Args:
        video_path: Path to the original video (used for video stream).
        audio_path: Path to the new audio track.
        output_path: Path for the final output video.
    
    Returns:
        Path to the merged video file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ],
        description="merge_video_audio",
    )

    return output_path
