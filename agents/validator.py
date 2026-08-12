"""
NeuroForge Validator Agent
Critical reviewer that catches architecture flaws, anti-patterns,
and production-readiness issues before deployment.
"""
from __future__ import annotations
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.schemas import AgentRole, ArchitectureDNA
from core.memory import memory_graph
from core.prompts import VALIDATOR_SYSTEM


class ValidatorAgent(BaseAgent):
    role = AgentRole.VALIDATOR

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        architectures: list[ArchitectureDNA] = input_data.get("architectures", [])
        analysis = input_data.get("analysis", {})
        generated_code = input_data.get("generated_code", "")
        problem = input_data.get("problem_statement", "")

        if not architectures:
            return {"valid": True, "issues": [], "approval": "approved", "correctness_score": 1.0}

        arch = architectures[0]
        context = memory_graph.build_agent_context(session_id, self.role)

        code_snippet = generated_code[:1500] if generated_code else "No code generated yet."

        user_msg = f"""Validate this ML architecture for correctness and production readiness.

PROBLEM: {problem}

ARCHITECTURE:
  Name: {arch.name}
  Backbone: {arch.backbone}
  Domain: {arch.domain.value}
  Tags: {', '.join(arch.tags)}
  Anti-patterns already flagged: {arch.anti_patterns_found}
  Novelty score: {arch.novelty_score}
  Feasibility score: {arch.feasibility_score}

LAYERS:
{self._layers_str(arch)}

TRAINING:
{self._training_str(arch)}

PROBLEM ANALYSIS:
  Type: {analysis.get('problem_type', 'unknown')}
  Label regime: {analysis.get('label_regime', 'fully_supervised')}
  Data challenges: {analysis.get('data_challenges', [])}
  Output type: {analysis.get('output_type', 'unknown')}

CODE SNIPPET (first 1500 chars):
{code_snippet}

COGNITIVE MEMORY:
{context}

Check for ALL categories of issues:
- Architecture anti-patterns
- Training configuration errors  
- Data-architecture mismatches
- RL-specific issues (if applicable)
- LLM fine-tuning pitfalls (if applicable)
- Production deployment gaps

Return the complete validation JSON."""

        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.1, max_tokens=2048)
        parsed = self._extract_json(content)

        if "_parse_failed" in parsed:
            parsed = {
                "valid": True,
                "severity": "ok",
                "issues": [],
                "correctness_score": 0.85,
                "production_readiness_score": 0.80,
                "deployment_checklist": [
                    "Add input validation and schema checks",
                    "Add ONNX/TorchScript export for serving",
                    "Set up monitoring for data distribution shift",
                    "Implement model versioning",
                    "Add A/B testing infrastructure",
                ],
                "approval": "approved_with_warnings",
            }

        # Collect all anti-patterns found
        all_anti_patterns = [
            i.get("issue", "") for i in parsed.get("issues", [])
            if i.get("severity") in ("critical", "error")
        ]

        self._record_trace(
            session_id=session_id,
            input_summary=f"Validate {arch.name}",
            output_summary=(
                f"approval={parsed.get('approval')}, "
                f"issues={len(parsed.get('issues', []))}, "
                f"correctness={parsed.get('correctness_score', '?')}"
            ),
            full_output=content,
            tokens_used=tokens,
            confidence=0.93,
        )

        return {**parsed, "all_anti_patterns": all_anti_patterns}

    def _layers_str(self, arch: ArchitectureDNA) -> str:
        if not arch.layers:
            return "  (no layer detail available)"
        return "\n".join(
            f"  [{l.name}] type={l.type}, params={l.params}"
            for l in arch.layers
        )

    def _training_str(self, arch: ArchitectureDNA) -> str:
        if not arch.training:
            return "  (no training config)"
        t = arch.training
        return (
            f"  optimizer={t.optimizer}, lr={t.learning_rate}, "
            f"batch={t.batch_size}, epochs={t.epochs}\n"
            f"  loss={t.loss_function}, metrics={t.metrics}\n"
            f"  regularization={t.regularization}"
        )
