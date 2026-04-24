"""
Analysis Agent — Scene Analysis via Gemini Vision
LangGraph node name: "analysis"

Extracts frames from video, sends to Gemini Vision for scene context,
speaker count, and speaker profile analysis.
"""

import json
import logging
from pathlib import Path

import google.generativeai as genai

from backend.state import DubbingState
from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.utils.ffmpeg import extract_frames

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

ANALYSIS_PROMPT = """You are analyzing a video for dubbing purposes. Look at these frames and provide:

1. Scene context: What kind of content is this? (movie, news, documentary, comedy, drama, action etc.) What is the overall mood and tone?
2. Estimated number of speakers visible or implied
3. For each speaker (label them SPEAKER_00, SPEAKER_01 etc.): their apparent gender, emotional state, and speaking style (formal, casual, intense, calm etc.)

Respond in this exact JSON format and nothing else:
{
    "scene_context": "string describing tone mood and setting",
    "speaker_count": number,
    "speaker_profiles": {
        "SPEAKER_00": {"gender": "male or female", "emotion": "string", "style": "string"},
        "SPEAKER_01": {"gender": "male or female", "emotion": "string", "style": "string"}
    }
}"""


def _load_frame_as_part(frame_path: str) -> dict:
    """Load a frame image file for Gemini API upload.
    
    Args:
        frame_path: Path to the JPEG frame file.
    
    Returns:
        A dict suitable for Gemini's multimodal content API.
    """
    with open(frame_path, "rb") as f:
        image_bytes = f.read()
    return {
        "mime_type": "image/jpeg",
        "data": image_bytes,
    }


def analysis_agent(state: DubbingState) -> dict:
    """Analyze video frames with Gemini Vision to extract scene context
    and speaker profiles.
    
    This agent is NON-CRITICAL: if Gemini vision fails entirely, it
    returns safe defaults and the pipeline continues.
    
    Args:
        state: Current pipeline state.
    
    Returns:
        Dict of state updates: scene_context, speaker_count, speaker_profiles.
    """
    logger.info("=== ANALYSIS AGENT START ===")
    video_path = state["video_path"]
    temp_dir = state["temp_dir"]
    frames_dir = str(Path(temp_dir) / "frames")

    # Default fallback values
    defaults = {
        "scene_context": "general content",
        "speaker_count": 1,
        "speaker_profiles": {
            "SPEAKER_00": {
                "gender": "male",
                "emotion": "neutral",
                "style": "conversational",
            }
        },
        "current_step": "analysis_complete",
    }

    try:
        # Step 1: Extract frames every 30 seconds
        logger.info("Extracting frames from video...")
        frame_paths = extract_frames(video_path, frames_dir, interval_seconds=30)

        if not frame_paths:
            logger.warning("No frames extracted — using defaults")
            state_updates = dict(defaults)
            state_updates["warnings"] = state.get("warnings", []) + [
                "No frames extracted from video"
            ]
            return state_updates

        # Limit to 5 representative frames
        if len(frame_paths) > 5:
            step = len(frame_paths) // 5
            frame_paths = frame_paths[::step][:5]

        logger.info(f"Using {len(frame_paths)} frames for analysis")

        # Step 2: Send frames to Gemini Vision
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Build multimodal content
        content_parts = []
        for fp in frame_paths:
            content_parts.append(_load_frame_as_part(fp))
        content_parts.append(ANALYSIS_PROMPT)

        logger.info("Sending frames to Gemini Vision...")
        response = model.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

        response_text = response.text.strip()
        logger.info(f"Gemini Vision response: {response_text[:200]}...")

        # Step 3: Parse JSON response
        # Handle markdown code block wrapping
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        # Replace single quotes with double quotes for valid JSON
        response_text = response_text.replace("'", '"')

        parsed = json.loads(response_text)

        scene_context = parsed.get("scene_context", defaults["scene_context"])
        speaker_count = int(parsed.get("speaker_count", defaults["speaker_count"]))
        speaker_profiles = parsed.get("speaker_profiles", defaults["speaker_profiles"])

        # Validate speaker_profiles structure
        if not isinstance(speaker_profiles, dict) or len(speaker_profiles) == 0:
            speaker_profiles = defaults["speaker_profiles"]

        logger.info(
            f"Analysis complete: {speaker_count} speakers, "
            f"context='{scene_context[:60]}...'"
        )

        return {
            "scene_context": scene_context,
            "speaker_count": speaker_count,
            "speaker_profiles": speaker_profiles,
            "current_step": "analysis_complete",
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response as JSON: {e}")
        result = dict(defaults)
        result["warnings"] = state.get("warnings", []) + [
            f"Analysis JSON parse failed: {e}"
        ]
        return result

    except Exception as e:
        logger.warning(f"Analysis agent failed: {e}")
        result = dict(defaults)
        result["warnings"] = state.get("warnings", []) + [
            f"Analysis agent error: {e}"
        ]
        return result
