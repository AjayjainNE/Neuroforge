"""
NeuroForge Orchestrator Agent
Master controller: parses problems, plans agent pipelines, and synthesizes results.
"""
from __future__ import annotations
from typing import Any, Dict

from agents.base import BaseAgent
from models.schemas import AgentRole, TaskDomain, ComplexityLevel, OutputDepth
from core.memory import memory_graph
from core.prompts import ORCHESTRATOR_SYSTEM


class OrchestratorAgent(BaseAgent):
    role = AgentRole.ORCHESTRATOR

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse the problem statement and produce a structured task plan.
        Returns: domain, complexity, depth, agent_plan, and key metadata.
        """
        problem = input_data.get("problem_statement", "")
        constraints = input_data.get("constraints", "")
        depth = input_data.get("depth", "complete")
        preferred_framework = input_data.get("preferred_framework", "")

        user_msg = f"""Problem Statement: {problem}

Additional Constraints: {constraints or "None specified"}
Preferred Framework: {preferred_framework or "PyTorch (default)"}
Requested Output Depth: {depth}

Analyse this and produce the structured task plan JSON."""

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.1, max_tokens=1024)
        parsed = self._extract_json(content)

        # Normalise fields with safe defaults
        domain_str = parsed.get("domain", "deep_learning")
        complexity_str = parsed.get("complexity", "intermediate")

        try:
            domain = TaskDomain(domain_str)
        except ValueError:
            domain = TaskDomain.DEEP_LEARNING

        try:
            complexity = ComplexityLevel(complexity_str)
        except ValueError:
            complexity = ComplexityLevel.INTERMEDIATE

        result = {
            "domain": domain,
            "complexity": complexity,
            "depth": OutputDepth(depth),
            "core_challenges": parsed.get("core_challenges", []),
            "data_characteristics": parsed.get("data_characteristics", ""),
            "success_metrics": parsed.get("success_metrics", []),
            "agent_plan": parsed.get("agent_plan", ["analyzer", "architect", "coder", "optimizer", "validator"]),
            "special_considerations": parsed.get("special_considerations", ""),
            "estimated_architecture_family": parsed.get("estimated_architecture_family", "transformer"),
            "raw": parsed,
        }

        critique = await self._self_critique(content)
        self._record_trace(
            session_id=session_id,
            input_summary=problem[:120],
            output_summary=(
                f"domain={domain.value}, complexity={complexity.value}, "
                f"family={result['estimated_architecture_family']}"
            ),
            full_output=content,
            tokens_used=tokens,
            confidence=0.95,
            critique=critique,
        )

        return result
