"""
NeuroForge Coder Agent
Generates complete, runnable Python implementation from an ArchitectureDNA.
Supports PyTorch, TensorFlow, JAX, and Scikit-learn.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional

from agents.base import BaseAgent
from models.schemas import AgentRole, ArchitectureDNA
from core.memory import memory_graph
from core.prompts import CODER_SYSTEM


PYTORCH_TEMPLATE_HINT = """
Generate a complete, self-contained Python file with:

1. IMPORTS (torch, nn, optim, DataLoader, etc.)
2. CONFIG DATACLASS with all hyperparameters
3. DATASET CLASS — reads from local file (CSV/JSON/Parquet) or generates synthetic data
4. MODEL CLASS — complete forward pass with all layers
5. TRAINING LOOP — loss, optimizer, scheduler, gradient clipping, early stopping
6. VALIDATION LOOP — metrics computation
7. EVALUATION FUNCTION — final test metrics
8. INFERENCE FUNCTION — single sample prediction
9. CHECKPOINT FUNCTIONS — save/load
10. MAIN GUARD with argparse

The code must be 100% runnable. No TODOs. No stubs. All logic complete.
Use rich for progress bars if available, else tqdm.
"""


class CoderAgent(BaseAgent):
    role = AgentRole.CODER
    use_code_model = True  # Use codestral or llama code variant

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem = input_data.get("problem_statement", "")
        architectures: list[ArchitectureDNA] = input_data.get("architectures", [])
        framework = input_data.get("preferred_framework", "PyTorch")
        dataset_path = input_data.get("dataset_path", None)
        dataset_profile = input_data.get("dataset_profile", None)
        analysis = input_data.get("analysis", {})
        optimizer_result = input_data.get("optimizer_result", {})

        if not architectures:
            return {"code": "# No architecture available to implement.", "tokens_used": 0}

        # Use the first (recommended) architecture
        arch = architectures[0]

        context = memory_graph.build_agent_context(session_id, self.role)
        arch_json = self._arch_to_summary(arch)

        dataset_info = ""
        if dataset_profile:
            dp = dataset_profile
            cols = [c.name for c in dp.column_profiles[:8]]
            dataset_info = (
                f"\nDataset: {dp.path}\n"
                f"Format: {dp.format}, Rows: {dp.rows}, Cols: {dp.columns}\n"
                f"Columns: {', '.join(cols)}\n"
                f"Target column: {dp.suggested_target or 'unknown'}"
            )
        elif dataset_path:
            dataset_info = f"\nDataset path: {dataset_path}"

        optim_hints = ""
        if optimizer_result:
            tweaks = optimizer_result.get("architecture_tweaks", [])
            tricks = optimizer_result.get("training_tricks", [])
            compression = optimizer_result.get("compression_options", [])
            optim_hints = (
                f"\nOptimizer Recommendations:\n"
                f"  Tweaks: {json.dumps(tweaks[:3])}\n"
                f"  Training tricks: {tricks[:5]}\n"
                f"  Compression: {[c.get('technique') for c in compression[:2]]}"
            )

        user_msg = f"""Generate a complete Python implementation for this architecture.

PROBLEM: {problem}

ARCHITECTURE SPECIFICATION:
{arch_json}

FRAMEWORK: {framework}
{dataset_info}
{optim_hints}

PROBLEM ANALYSIS:
  Type: {analysis.get('problem_type', 'classification')}
  Label regime: {analysis.get('label_regime', 'fully_supervised')}
  Preprocessing: {analysis.get('preprocessing_pipeline', [])}

{PYTORCH_TEMPLATE_HINT}

COGNITIVE MEMORY:
{context}

Write the complete implementation now. Pure Python only, no markdown fences."""

        messages = [
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        code, tokens = await self._call_llm(
            messages, temperature=0.15, max_tokens=4096
        )

        # Strip accidental markdown if present
        code = self._clean_code(code)

        self._record_trace(
            session_id=session_id,
            input_summary=f"Implement {arch.name}",
            output_summary=f"Generated {len(code.splitlines())} lines of {framework} code",
            full_output=code,
            tokens_used=tokens,
            confidence=0.90,
        )

        return {"code": code, "tokens_used": tokens, "framework": framework}

    def _arch_to_summary(self, arch: ArchitectureDNA) -> str:
        lines = [
            f"Name: {arch.name}",
            f"Backbone: {arch.backbone}",
            f"Rationale: {arch.rationale[:500]}",
        ]
        if arch.layers:
            lines.append("Layers:")
            for l in arch.layers:
                lines.append(f"  - {l.name} ({l.type}) params={l.params} | {l.notes or ''}")
        if arch.connections:
            lines.append("Connections:")
            for c in arch.connections:
                lines.append(f"  - {c.from_layer} → {c.to_layer} [{c.connection_type}]")
        if arch.training:
            t = arch.training
            lines.append(
                f"Training: optimizer={t.optimizer}, lr={t.learning_rate}, "
                f"batch={t.batch_size}, epochs={t.epochs}, loss={t.loss_function}"
            )
            lines.append(f"Metrics: {t.metrics}")
            if t.curriculum:
                lines.append(f"Curriculum: {t.curriculum}")
        if arch.compute:
            c = arch.compute
            lines.append(
                f"Compute: {c.parameters_millions}M params, "
                f"hardware={c.recommended_hardware}"
            )
        return "\n".join(lines)

    def _clean_code(self, code: str) -> str:
        """Remove markdown backticks and language annotations."""
        import re
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
        return code.strip()
