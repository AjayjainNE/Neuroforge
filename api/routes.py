"""
NeuroForge API Routes
REST endpoints for architecture design, dataset analysis, comparison, and session management.
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse

from config import settings
from models.schemas import (
    DesignRequest, DesignResponse,
    CompareRequest, CompareResponse,
    DatasetAnalysisRequest, HealthResponse,
)
from core.memory import memory_graph
from core.pipeline import APEXPipeline

router = APIRouter()
_pipeline = APEXPipeline()


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check and system information."""
    return HealthResponse(
        status="healthy",
        provider=settings.LLM_PROVIDER.value,
        model=settings.primary_model,
        agents_available=[
            "orchestrator", "analyzer", "architect",
            "coder", "optimizer", "validator", "explainer"
        ],
    )


# ─── Architecture Design ─────────────────────────────────────────────────────

@router.post("/design", response_model=DesignResponse, tags=["Architecture"])
async def design_architecture(request: DesignRequest):
    """
    Main endpoint: design an ML/DL/RL architecture from a problem statement.
    
    Runs the full APEX multi-agent pipeline:
    Orchestrator → Analyzer → Architect → Optimizer → Coder → Validator → Explainer
    
    Returns complete architecture DNA, implementation code, design document, and training guide.
    """
    try:
        response = await _pipeline.run(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ─── WebSocket streaming endpoint ────────────────────────────────────────────

@router.websocket("/design/stream")
async def design_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming real-time agent progress.
    Send a DesignRequest JSON, receive agent-by-agent progress events.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        request = DesignRequest(**data)

        progress_log = []

        async def send_progress(agent: str, msg: str):
            event = {"event": "progress", "agent": agent, "message": msg, "ts": time.time()}
            progress_log.append(event)
            try:
                await websocket.send_json(event)
            except Exception:
                pass

        response = await _pipeline.run(request, progress_callback=send_progress)

        await websocket.send_json({
            "event": "complete",
            "session_id": response.session_id,
            "domain": response.domain.value,
            "complexity": response.complexity.value,
            "recommended_architecture": response.recommended.name,
            "total_tokens": response.total_tokens_used,
            "processing_time_sec": response.processing_time_sec,
            "full_response": response.model_dump(mode="json"),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Dataset Analysis ─────────────────────────────────────────────────────────

@router.post("/analyze-dataset", tags=["Data"])
async def analyze_dataset(request: DatasetAnalysisRequest):
    """
    Profile a local dataset file.
    Returns column statistics, inferred task type, quality score, and recommendations.
    """
    inspector = _pipeline.inspector
    try:
        profile = await inspector.profile(
            request.dataset_path,
            target_column=request.target_column,
            task_hint=request.task_hint,
        )
        return profile.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Architecture Comparison ──────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse, tags=["Architecture"])
async def compare_architectures(request: CompareRequest):
    """
    Compare multiple candidate architectures from a session.
    Generates a markdown comparison table and selects a winner.
    """
    session = memory_graph.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Run /design first.")

    candidates = {a.dna_id: a for a in session.architecture_candidates}
    selected = [candidates[aid] for aid in request.architecture_ids if aid in candidates]

    if len(selected) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 valid architecture IDs. Available: {list(candidates.keys())}"
        )

    # Build comparison table
    table_lines = ["| Criterion | " + " | ".join(a.name for a in selected) + " |"]
    table_lines.append("|" + "---|" * (len(selected) + 1))

    criteria_scores: Dict[str, Dict[str, float]] = {c: {} for c in request.criteria}

    score_map = {
        "accuracy": lambda a: a.novelty_score * 0.3 + a.feasibility_score * 0.7,
        "speed": lambda a: 1.0 - min(a.compute.parameters_millions / 1000, 1.0) if a.compute else 0.5,
        "memory": lambda a: 1.0 - min((a.compute.gpu_memory_gb or 8) / 80, 1.0) if a.compute else 0.5,
        "simplicity": lambda a: 1.0 - a.novelty_score * 0.5,
        "scalability": lambda a: a.feasibility_score,
    }

    for criterion in request.criteria:
        scorer = score_map.get(criterion, lambda a: 0.5)
        row = [f"**{criterion}**"]
        for arch in selected:
            score = round(scorer(arch), 2)
            criteria_scores[criterion][arch.dna_id] = score
            row.append(f"{score:.2f}")
        table_lines.append("| " + " | ".join(row) + " |")

    comparison_table = "\n".join(table_lines)

    # Select winner by total score
    totals = {
        a.dna_id: sum(criteria_scores[c].get(a.dna_id, 0) for c in request.criteria)
        for a in selected
    }
    winner_id = max(totals, key=lambda k: totals[k])
    winner = candidates[winner_id]

    rationale = (
        f"**{winner.name}** (DNA-{winner_id}) wins with a total score of {totals[winner_id]:.2f}. "
        f"It excels in feasibility ({winner.feasibility_score:.2f}) "
        f"with {winner.compute.parameters_millions if winner.compute else '?'}M parameters. "
        f"Key differentiator: {winner.rationale[:200]}..."
    )

    return CompareResponse(
        session_id=request.session_id,
        comparison_table=comparison_table,
        winner_id=winner_id,
        rationale=rationale,
        criteria_scores=criteria_scores,
    )


# ─── Session Management ────────────────────────────────────────────────────────

@router.get("/session/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Retrieve full session state including all agent traces and architectures."""
    session = memory_graph.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump(mode="json")


@router.get("/session/{session_id}/architectures", tags=["Sessions"])
async def get_session_architectures(session_id: str):
    """List all candidate architectures in a session."""
    session = memory_graph.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "count": len(session.architecture_candidates),
        "architectures": [a.model_dump() for a in session.architecture_candidates],
    }


@router.delete("/session/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a session and free its memory."""
    session = memory_graph.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    memory_graph._sessions.pop(session_id, None)
    memory_graph._graphs.pop(session_id, None)
    return {"deleted": session_id}


@router.post("/session/cleanup", tags=["Sessions"])
async def cleanup_stale_sessions():
    """Evict sessions older than SESSION_TTL_HOURS."""
    evicted = memory_graph.evict_stale_sessions()
    return {"evicted_sessions": evicted}
