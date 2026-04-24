"""
DubGraph — FastAPI Backend
Single-file entry point for the dubbing API.

Endpoints:
  POST /dub    — Upload video + direction, run pipeline, return dubbed video
  GET  /health — Health check with GPU info
  
Serves React frontend as static files.
"""

import logging
import mimetypes
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    TEMP_DIR,
    OUTPUT_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    ACCEPTED_VIDEO_MIMES,
    ACCEPTED_VIDEO_EXTENSIONS,
)
from backend.graph import dubbing_graph
from backend.utils.gpu import is_cuda_available, get_gpu_name
from backend.utils.cleanup import cleanup_job_temp

# ── Logging Setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dubgraph")

# ── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="DubGraph",
    description="Multi-agent video dubbing pipeline powered by LangGraph",
    version="1.0.0",
)

# CORS — allow React dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper Functions ────────────────────────────────────────────────────

def _validate_video_file(file: UploadFile) -> None:
    """Validate that the uploaded file is a video within size limits.
    
    Args:
        file: The uploaded file.
    
    Raises:
        HTTPException: If validation fails.
    """
    # Check extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ACCEPTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. "
                       f"Accepted: {', '.join(ACCEPTED_VIDEO_EXTENSIONS)}",
            )

    # Check MIME type
    content_type = file.content_type or ""
    if content_type and content_type not in ACCEPTED_VIDEO_MIMES:
        # Some browsers send application/octet-stream — allow it if extension is valid
        if content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported MIME type: {content_type}",
            )


def _parse_direction(direction: str) -> tuple[str, str]:
    """Parse the dubbing direction string into source/target language codes.
    
    Args:
        direction: Either 'hi_to_en' or 'en_to_hi'.
    
    Returns:
        Tuple of (source_language, target_language).
    
    Raises:
        HTTPException: If direction is invalid.
    """
    if direction == "hi_to_en":
        return "hi", "en"
    elif direction == "en_to_hi":
        return "en", "hi"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid direction: {direction}. Must be 'hi_to_en' or 'en_to_hi'.",
        )


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU info."""
    cuda = is_cuda_available()
    return {
        "status": "ok",
        "gpu": cuda,
        "gpu_name": get_gpu_name() if cuda else "N/A",
    }


@app.post("/dub")
async def dub_video(
    video: UploadFile = File(...),
    direction: str = Form(...),
):
    """Upload a video and run the dubbing pipeline.
    
    Args:
        video: The video file to dub.
        direction: 'hi_to_en' or 'en_to_hi'.
    
    Returns:
        FileResponse with the dubbed video on success.
        JSONResponse with error details on failure.
    """
    # ── Validate inputs ─────────────────────────────────────────────
    _validate_video_file(video)
    source_language, target_language = _parse_direction(direction)

    # ── Generate job ID and paths ───────────────────────────────────
    job_id = str(uuid.uuid4())
    job_temp_dir = str(TEMP_DIR / job_id)
    Path(job_temp_dir).mkdir(parents=True, exist_ok=True)

    input_video_path = str(Path(job_temp_dir) / "input.mp4")

    logger.info(
        f"New dubbing job: {job_id} | "
        f"direction={direction} | "
        f"file={video.filename}"
    )

    # ── Save uploaded video ─────────────────────────────────────────
    try:
        total_bytes = 0
        with open(input_video_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    # Clean up partial file
                    Path(input_video_path).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: "
                               f"{MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB",
                    )
                f.write(chunk)

        logger.info(
            f"Video saved: {input_video_path} "
            f"({total_bytes / (1024*1024):.1f} MB)"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # ── Initialize DubbingState ─────────────────────────────────────
    initial_state = {
        "video_path": input_video_path,
        "source_language": source_language,
        "target_language": target_language,
        "job_id": job_id,
        "scene_context": "",
        "speaker_count": 0,
        "speaker_profiles": {},
        "segments": [],
        "translated_segments": [],
        "audio_segments": [],
        "final_video_path": "",
        "current_step": "initialized",
        "errors": [],
        "warnings": [],
        "retry_count": 0,
        "temp_dir": job_temp_dir,
        "vocals_path": None,
        "background_path": None,
    }

    # ── Run LangGraph Pipeline (synchronous) ────────────────────────
    try:
        logger.info(f"Starting DubGraph pipeline for job {job_id}...")
        final_state = dubbing_graph.invoke(initial_state)

        # Check for errors
        errors = final_state.get("errors", [])
        if errors:
            logger.error(f"Pipeline completed with errors: {errors}")
            # Clean up on failure
            cleanup_job_temp(job_temp_dir)
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "step": final_state.get("current_step", "unknown"),
                    "message": "; ".join(errors),
                    "warnings": final_state.get("warnings", []),
                },
            )

        final_video_path = final_state.get("final_video_path", "")
        if not final_video_path or not Path(final_video_path).exists():
            logger.error("Pipeline completed but no output video found")
            cleanup_job_temp(job_temp_dir)
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "step": final_state.get("current_step", "unknown"),
                    "message": "Pipeline completed but no output video was produced",
                    "warnings": final_state.get("warnings", []),
                },
            )

        logger.info(f"Job {job_id} complete! Output: {final_video_path}")

        # Log any warnings
        warnings = final_state.get("warnings", [])
        if warnings:
            logger.warning(f"Job {job_id} warnings: {warnings}")

        return FileResponse(
            path=final_video_path,
            media_type="video/mp4",
            filename=f"dubbed_{video.filename or 'video.mp4'}",
        )

    except Exception as e:
        logger.error(f"Pipeline exception for job {job_id}: {e}", exc_info=True)
        cleanup_job_temp(job_temp_dir)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "step": "pipeline_error",
                "message": str(e),
                "warnings": [],
            },
        )


@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    """Serve a dubbed video from the outputs directory."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), media_type="video/mp4")


# ── Serve React Frontend ───────────────────────────────────────────────
# Mount static files LAST so API routes take priority
frontend_build = Path(__file__).resolve().parent.parent / "frontend" / "build"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")
