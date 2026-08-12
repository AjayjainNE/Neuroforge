"""
NeuroForge Cognitive Memory Graph
Shared graph-based memory that all agents read and write to.
Agents leave semantic traces that other agents can query.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import networkx as nx

from models.schemas import AgentTrace, AgentRole, SessionMemory, ArchitectureDNA
from config import settings


class CognitiveMemoryGraph:
    """
    A directed graph where:
    - Nodes are agent traces (thoughts, outputs, critiques)
    - Edges represent information flow between agents
    - Supports semantic querying and context building
    """

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._last_access: Dict[str, float] = {}

    # ─── Session Management ───────────────────────────────────────────────────

    def create_session(self, session_id: Optional[str] = None) -> SessionMemory:
        session = SessionMemory()
        if session_id:
            session.session_id = session_id
        self._sessions[session.session_id] = session
        self._graphs[session.session_id] = nx.DiGraph()
        self._last_access[session.session_id] = time.time()
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        self._last_access[session_id] = time.time()
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> SessionMemory:
        if session_id and session_id in self._sessions:
            return self.get_session(session_id)
        return self.create_session(session_id)

    # ─── Trace Management ────────────────────────────────────────────────────

    def record_trace(self, session_id: str, trace: AgentTrace) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        session.add_trace(trace)
        graph = self._graphs[session_id]
        graph.add_node(
            trace.trace_id,
            agent=trace.agent.value,
            summary=trace.output_summary,
            confidence=trace.confidence,
            timestamp=trace.timestamp.isoformat(),
        )
        # Connect to previous trace from the same or dependent agent
        if len(session.traces) > 1:
            prev = session.traces[-2]
            graph.add_edge(prev.trace_id, trace.trace_id, relation="follows")

    def record_architecture(self, session_id: str, arch: ArchitectureDNA) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        # Deduplicate by dna_id
        existing_ids = {a.dna_id for a in session.architecture_candidates}
        if arch.dna_id not in existing_ids:
            session.architecture_candidates.append(arch)

    # ─── Context Building ────────────────────────────────────────────────────

    def build_agent_context(self, session_id: str, requesting_agent: AgentRole) -> str:
        """
        Builds a rich context string for an agent by summarising
        what all prior agents have produced in this session.
        """
        session = self._sessions.get(session_id)
        if not session or not session.traces:
            return "No prior agent context available."

        lines = ["=== COGNITIVE MEMORY GRAPH — AGENT CONTEXT ==="]
        for trace in session.traces[-8:]:  # Last 8 traces
            lines.append(
                f"[{trace.agent.value.upper()} | conf={trace.confidence:.2f}]"
                f"\n  → {trace.output_summary}"
            )
            if trace.critique:
                lines.append(f"  ✗ Self-critique: {trace.critique}")

        if session.dataset_profile:
            dp = session.dataset_profile
            lines.append(
                f"\n[DATASET PROFILE]\n"
                f"  Rows={dp.rows}, Cols={dp.columns}, Format={dp.format}\n"
                f"  Inferred task: {dp.inferred_task}, Quality: {dp.data_quality_score:.2f}"
            )

        if session.architecture_candidates:
            lines.append(f"\n[CANDIDATE ARCHITECTURES: {len(session.architecture_candidates)}]")
            for a in session.architecture_candidates:
                lines.append(
                    f"  DNA-{a.dna_id}: {a.name} | "
                    f"novelty={a.novelty_score:.2f} | feasibility={a.feasibility_score:.2f}"
                )

        return "\n".join(lines)

    def build_conversation_messages(
        self,
        session_id: str,
        system_prompt: str,
        new_user_message: str,
        max_history: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Builds the full messages list for an LLM call including
        rolling conversation history from the session.
        """
        session = self._sessions.get(session_id)
        messages = [{"role": "system", "content": system_prompt}]

        if session and session.conversation_history:
            history = session.conversation_history[-max_history * 2:]
            messages.extend(history)

        messages.append({"role": "user", "content": new_user_message})
        return messages

    def append_conversation(
        self, session_id: str, role: str, content: str
    ) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.conversation_history.append({"role": role, "content": content})

    # ─── Graph Analytics ─────────────────────────────────────────────────────

    def get_agent_chain(self, session_id: str) -> List[str]:
        graph = self._graphs.get(session_id, nx.DiGraph())
        return [
            graph.nodes[n].get("agent", "?")
            for n in nx.topological_sort(graph)
        ]

    # ─── Cleanup ─────────────────────────────────────────────────────────────

    def evict_stale_sessions(self) -> int:
        cutoff = time.time() - settings.SESSION_TTL_HOURS * 3600
        stale = [
            sid for sid, t in self._last_access.items() if t < cutoff
        ]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._graphs.pop(sid, None)
            self._last_access.pop(sid, None)
        return len(stale)


# Singleton memory graph shared by all agents
memory_graph = CognitiveMemoryGraph()
