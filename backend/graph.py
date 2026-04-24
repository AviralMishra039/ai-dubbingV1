"""
DubGraph — LangGraph Graph Definition
Defines the full multi-agent pipeline as a LangGraph StateGraph.

Nodes: analysis → transcription → translation → tts → assembly
Director: Conditional edges validate each node's output before proceeding.
"""

import logging

from langgraph.graph import StateGraph, END

from backend.state import DubbingState
from backend.agents.analysis import analysis_agent
from backend.agents.transcription import transcription_agent
from backend.agents.translation import translation_agent
from backend.agents.tts import tts_agent
from backend.agents.assembly import assembly_agent
from backend.agents.director import (
    route_after_analysis,
    route_after_transcription,
    route_after_translation,
    route_after_tts,
    increment_retry,
)

logger = logging.getLogger(__name__)


# ── Error-Wrapped Node Functions ────────────────────────────────────────
# Each wrapper catches exceptions and routes them through state.errors
# so the director can decide how to handle them.

def _safe_analysis(state: DubbingState) -> dict:
    """Run analysis agent with error handling (NON-CRITICAL)."""
    try:
        state_update = {"current_step": "analysis"}
        result = analysis_agent(state)
        state_update.update(result)
        return state_update
    except Exception as e:
        logger.error(f"Analysis agent exception: {e}")
        # Analysis is non-critical — return defaults and continue
        return {
            "scene_context": "general content",
            "speaker_count": 1,
            "speaker_profiles": {
                "SPEAKER_00": {
                    "gender": "male",
                    "emotion": "neutral",
                    "style": "conversational",
                }
            },
            "current_step": "analysis_failed",
            "warnings": state.get("warnings", []) + [f"Analysis failed: {e}"],
        }


def _safe_transcription(state: DubbingState) -> dict:
    """Run transcription agent with error handling (CRITICAL)."""
    try:
        state_update = {"current_step": "transcription"}
        result = transcription_agent(state)
        state_update.update(result)
        return state_update
    except Exception as e:
        logger.error(f"Transcription agent exception: {e}")
        return {
            "segments": [],
            "current_step": "transcription_failed",
            "errors": state.get("errors", []) + [f"Transcription failed: {e}"],
        }


def _safe_translation(state: DubbingState) -> dict:
    """Run translation agent with error handling (CRITICAL)."""
    try:
        state_update = {"current_step": "translation"}
        result = translation_agent(state)
        state_update.update(result)
        return state_update
    except Exception as e:
        logger.error(f"Translation agent exception: {e}")
        return {
            "translated_segments": [],
            "current_step": "translation_failed",
            "errors": state.get("errors", []) + [f"Translation failed: {e}"],
        }


def _safe_tts(state: DubbingState) -> dict:
    """Run TTS agent with error handling (CRITICAL)."""
    try:
        state_update = {"current_step": "tts"}
        result = tts_agent(state)
        state_update.update(result)
        return state_update
    except Exception as e:
        logger.error(f"TTS agent exception: {e}")
        return {
            "audio_segments": [],
            "current_step": "tts_failed",
            "errors": state.get("errors", []) + [f"TTS failed: {e}"],
        }


def _safe_assembly(state: DubbingState) -> dict:
    """Run assembly agent with error handling (CRITICAL)."""
    try:
        state_update = {"current_step": "assembly"}
        result = assembly_agent(state)
        state_update.update(result)
        return state_update
    except Exception as e:
        logger.error(f"Assembly agent exception: {e}")
        return {
            "final_video_path": "",
            "current_step": "assembly_failed",
            "errors": state.get("errors", []) + [f"Assembly failed: {e}"],
        }


def _tts_retry_node(state: DubbingState) -> dict:
    """Increment retry counter before re-entering TTS.
    
    This node runs when the director decides to retry TTS
    for segments that are too long.
    """
    return increment_retry(state)


# ── Build the Graph ─────────────────────────────────────────────────────

def build_dubbing_graph() -> StateGraph:
    """Build and compile the DubGraph LangGraph pipeline.
    
    Graph topology:
        START → analysis → [director] → transcription → [director] →
        translation → [director] → tts → [director] →
        (assembly | tts_retry → tts) → END
    
    Returns:
        Compiled LangGraph StateGraph ready for .invoke().
    """
    graph = StateGraph(DubbingState)

    # ── Add nodes ───────────────────────────────────────────────────
    graph.add_node("analysis", _safe_analysis)
    graph.add_node("transcription", _safe_transcription)
    graph.add_node("translation", _safe_translation)
    graph.add_node("tts", _safe_tts)
    graph.add_node("tts_retry", _tts_retry_node)
    graph.add_node("assembly", _safe_assembly)

    # ── Add edges ───────────────────────────────────────────────────

    # START → analysis
    graph.set_entry_point("analysis")

    # analysis → [director validates] → transcription or end
    graph.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "transcription": "transcription",
        },
    )

    # transcription → [director validates] → translation or end
    graph.add_conditional_edges(
        "transcription",
        route_after_transcription,
        {
            "translation": "translation",
            "end": END,
        },
    )

    # translation → [director validates] → tts or end
    graph.add_conditional_edges(
        "translation",
        route_after_translation,
        {
            "tts": "tts",
            "end": END,
        },
    )

    # tts → [director validates] → assembly, tts_retry, or end
    graph.add_conditional_edges(
        "tts",
        route_after_tts,
        {
            "assembly": "assembly",
            "tts": "tts_retry",
            "end": END,
        },
    )

    # tts_retry → tts (re-enter the TTS node after incrementing counter)
    graph.add_edge("tts_retry", "tts")

    # assembly → END
    graph.add_edge("assembly", END)

    # ── Compile ─────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("DubGraph compiled successfully")

    return compiled


# Module-level compiled graph (singleton)
dubbing_graph = build_dubbing_graph()
