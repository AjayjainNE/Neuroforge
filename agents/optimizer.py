"""
NeuroForge Optimizer Agent
Produces hyperparameter ranges, architecture tweaks, compression options,
and training tricks to maximize architecture performance.
"""
from __future__ import annotations
from typing import Any, Dict

from agents.base import BaseAgent
from models.schemas import AgentRole, ArchitectureDNA
from core.memory import memory_graph
from core.prompts import OPTIMIZER_SYSTEM


class OptimizerAgent(BaseAgent):
    role = AgentRole.OPTIMIZER

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem = input_data.get("problem_statement", "")
        architectures: list[ArchitectureDNA] = input_data.get("architectures", [])
        constraints = input_data.get("constraints", "")
        analysis = input_data.get("analysis", {})
        orchestrator = input_data.get("orchestrator_result", {})

        if not architectures:
            return self._default_optimization()

        arch = architectures[0]
        context = memory_graph.build_agent_context(session_id, self.role)

        # Build rich architecture description for optimizer
        arch_desc = self._describe_arch(arch)
        data_challenges = analysis.get("data_challenges", [])
        label_regime = analysis.get("label_regime", "fully_supervised")
        complexity = orchestrator.get("complexity", "intermediate")

        user_msg = f"""Optimize this ML architecture.

PROBLEM: {problem}

ARCHITECTURE:
{arch_desc}

CONTEXT:
  Complexity Level: {complexity}
  Label Regime: {label_regime}
  Data Challenges: {data_challenges}
  Constraints: {constraints or "None"}

COGNITIVE MEMORY:
{context}

Provide:
1. Hyperparameter search ranges (for Optuna or Ray Tune)
2. Architecture micro-optimizations (specific component tweaks)
3. Compression / efficiency options (quantization, pruning, distillation)
4. Training tricks and regularization techniques
5. Estimated performance improvements

Return the structured JSON optimization report."""

        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.2, max_tokens=2048)
        parsed = self._extract_json(content)

        if "_parse_failed" in parsed:
            parsed = self._default_optimization()

        critique = await self._self_critique(content)
        self._record_trace(
            session_id=session_id,
            input_summary=f"Optimize {arch.name}",
            output_summary=(
                f"estimated improvement: "
                f"acc +{parsed.get('estimated_improvement', {}).get('accuracy_gain_pct', '?')}%, "
                f"speedup {parsed.get('estimated_improvement', {}).get('training_speedup', '?')}"
            ),
            full_output=content,
            tokens_used=tokens,
            confidence=0.85,
            critique=critique,
        )

        return parsed

    def _describe_arch(self, arch: ArchitectureDNA) -> str:
        parts = [
            f"Name: {arch.name}",
            f"Backbone: {arch.backbone}",
            f"Tags: {', '.join(arch.tags)}",
        ]
        if arch.training:
            t = arch.training
            parts.append(
                f"Current training: optimizer={t.optimizer}, lr={t.learning_rate}, "
                f"batch={t.batch_size}, epochs={t.epochs}"
            )
            parts.append(f"Regularization: {t.regularization}")
        if arch.compute:
            parts.append(f"Parameters: {arch.compute.parameters_millions}M")
            parts.append(f"Hardware target: {arch.compute.recommended_hardware}")
        if arch.layers:
            parts.append(f"Layer count: {len(arch.layers)}")
            for l in arch.layers[:5]:
                parts.append(f"  {l.name}: {l.type} {l.params}")
        return "\n".join(parts)

    def _default_optimization(self) -> Dict[str, Any]:
        return {
            "hyperparameter_ranges": {
                "learning_rate": {"min": 1e-5, "max": 1e-2, "log_scale": True},
                "batch_size": {"options": [16, 32, 64, 128]},
                "dropout": {"min": 0.0, "max": 0.5},
                "weight_decay": {"min": 1e-6, "max": 1e-1, "log_scale": True},
            },
            "architecture_tweaks": [
                {
                    "component": "activation",
                    "current": "relu",
                    "suggested": "gelu",
                    "reason": "GELU typically improves performance in transformer-based models"
                }
            ],
            "compression_options": [
                {
                    "technique": "fp16_mixed_precision",
                    "speedup": "1.5-2x",
                    "memory_saving": "40-50%",
                    "accuracy_drop": "<0.1%"
                }
            ],
            "training_tricks": [
                "gradient_clipping_max_norm_1.0",
                "warmup_steps_500",
                "ema_decay_0.9999",
                "label_smoothing_0.1",
            ],
            "estimated_improvement": {
                "accuracy_gain_pct": 1.5,
                "training_speedup": "1.5x",
                "memory_reduction_pct": 30
            },
            "search_strategy": "bayesian_optuna",
            "optuna_config": {
                "n_trials": 50,
                "timeout_hours": 4,
                "pruner": "HyperbandPruner"
            }
        }
