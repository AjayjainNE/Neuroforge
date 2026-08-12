"""
NeuroForge Explainer Agent
Synthesises all agent outputs into comprehensive Markdown documentation:
  - Full architecture design document
  - Training guide
  - Evaluation strategy
  - Novelty analysis
"""
from __future__ import annotations
import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.schemas import AgentRole, ArchitectureDNA, ComputeEstimate
from core.memory import memory_graph
from core.prompts import EXPLAINER_SYSTEM, NOVELTY_SYSTEM, TRAINING_GUIDE_SYSTEM


class ExplainerAgent(BaseAgent):
    role = AgentRole.EXPLAINER

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem = input_data.get("problem_statement", "")
        architectures: list[ArchitectureDNA] = input_data.get("architectures", [])
        analysis = input_data.get("analysis", {})
        optimizer_result = input_data.get("optimizer_result", {})
        validator_result = input_data.get("validator_result", {})
        generated_code = input_data.get("generated_code", "")
        orchestrator = input_data.get("orchestrator_result", {})

        if not architectures:
            return {
                "full_design": "# No architecture to document.",
                "novelty_analysis": "",
                "training_guide": "",
                "evaluation_strategy": "",
                "tokens_used": 0,
            }

        arch = architectures[0]
        context = memory_graph.build_agent_context(session_id, self.role)

        # ── 1. Full Design Document ─────────────────────────────────────────
        design_doc, design_tokens = await self._generate_design_doc(
            problem, arch, analysis, optimizer_result, validator_result, orchestrator, context
        )

        # ── 2. Novelty Analysis ─────────────────────────────────────────────
        novelty_analysis, novelty_tokens = await self._novelty_analysis(arch)

        # ── 3. Training Guide ───────────────────────────────────────────────
        training_guide, training_tokens = await self._training_guide(arch, analysis)

        # ── 4. Evaluation Strategy ──────────────────────────────────────────
        evaluation_strategy = self._build_evaluation_strategy(arch, analysis)

        total_tokens = design_tokens + novelty_tokens + training_tokens

        self._record_trace(
            session_id=session_id,
            input_summary=f"Document {arch.name}",
            output_summary=f"Generated {len(design_doc.splitlines())} line design doc",
            full_output=design_doc[:500],
            tokens_used=total_tokens,
            confidence=0.92,
        )

        return {
            "full_design": design_doc,
            "novelty_analysis": novelty_analysis,
            "training_guide": training_guide,
            "evaluation_strategy": evaluation_strategy,
            "tokens_used": total_tokens,
        }

    async def _generate_design_doc(
        self, problem, arch, analysis, optimizer, validator, orchestrator, context
    ) -> tuple[str, int]:
        issues_summary = ""
        if validator:
            issues = validator.get("issues", [])
            checklist = validator.get("deployment_checklist", [])
            issues_summary = (
                f"Validation: {validator.get('approval')} | "
                f"Correctness: {validator.get('correctness_score', '?')} | "
                f"Issues: {len(issues)}\n"
                f"Deployment checklist: {checklist}"
            )

        optim_summary = ""
        if optimizer:
            tricks = optimizer.get("training_tricks", [])
            compression = optimizer.get("compression_options", [])
            improvement = optimizer.get("estimated_improvement", {})
            optim_summary = (
                f"Training tricks: {tricks}\n"
                f"Compression: {[c.get('technique') for c in compression]}\n"
                f"Est. improvement: {improvement}"
            )

        user_msg = f"""Create the complete technical design document for this architecture.

PROBLEM: {problem}

DOMAIN: {orchestrator.get('domain', 'deep_learning')}
COMPLEXITY: {orchestrator.get('complexity', 'advanced')}

ARCHITECTURE:
  Name: {arch.name}
  Backbone: {arch.backbone}
  Domain: {arch.domain.value}
  Tags: {', '.join(arch.tags)}
  Novelty Score: {arch.novelty_score:.2f}
  Feasibility Score: {arch.feasibility_score:.2f}
  Rationale: {arch.rationale}
  
  Layers ({len(arch.layers)}):
{self._layers_md(arch)}

  Connections:
{self._connections_md(arch)}

  Training:
{self._training_md(arch)}

  Compute:
{self._compute_md(arch)}

ANALYSIS:
  Type: {analysis.get('problem_type')}
  Regime: {analysis.get('label_regime')}
  Challenges: {analysis.get('data_challenges')}
  Paradigm: {analysis.get('recommended_paradigm')}

OPTIMIZATION:
{optim_summary}

VALIDATION:
{issues_summary}

COGNITIVE MEMORY:
{context}

Write the complete, rigorous Markdown design document. Include LaTeX math where appropriate."""

        messages = [
            {"role": "system", "content": EXPLAINER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.3, max_tokens=4096)
        return content, tokens

    async def _novelty_analysis(self, arch: ArchitectureDNA) -> tuple[str, int]:
        user_msg = f"""Analyze the novelty of this architecture:

Name: {arch.name}
Backbone: {arch.backbone}
Tags: {', '.join(arch.tags)}
Novel elements claimed: {arch.rationale[:400]}
Layers: {[l.name + ':' + l.type for l in arch.layers[:8]]}
Connections: {[f"{c.from_layer}->{c.to_layer}[{c.connection_type}]" for c in arch.connections]}
Novelty score (self-reported): {arch.novelty_score}

Return the JSON novelty analysis."""

        messages = [
            {"role": "system", "content": NOVELTY_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.2, max_tokens=1024)
        parsed = self._extract_json(content)

        novelty_md = f"""## Novelty Analysis

**Novelty Score**: {parsed.get('novelty_score', arch.novelty_score):.2f} / 1.0

**Similar to**: {', '.join(parsed.get('similar_to', []))}

**Novel Combinations**:
{chr(10).join('- ' + n for n in parsed.get('novel_combinations', []))}

**Originality Analysis**: {parsed.get('originality_analysis', '')}

**Research Angle**: {parsed.get('potential_publication_angle', 'N/A')}
"""
        return novelty_md, tokens

    async def _training_guide(self, arch: ArchitectureDNA, analysis: Dict) -> tuple[str, int]:
        user_msg = f"""Write a complete training guide for this architecture.

Architecture: {arch.name} ({arch.backbone})
Domain: {arch.domain.value}
Problem type: {analysis.get('problem_type', 'classification')}
Label regime: {analysis.get('label_regime', 'fully_supervised')}
Data challenges: {analysis.get('data_challenges', [])}
Preprocessing: {analysis.get('preprocessing_pipeline', [])}

Training config:
{self._training_md(arch)}

Compute:
{self._compute_md(arch)}

Write the complete phased training guide in Markdown."""

        messages = [
            {"role": "system", "content": TRAINING_GUIDE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.3, max_tokens=2048)
        return content, tokens

    def _build_evaluation_strategy(self, arch: ArchitectureDNA, analysis: Dict) -> str:
        problem_type = analysis.get("problem_type", "classification")
        label_regime = analysis.get("label_regime", "fully_supervised")

        metric_map = {
            "classification": ["Accuracy", "F1-macro", "ROC-AUC", "Confusion Matrix", "Calibration (ECE)"],
            "regression": ["MAE", "RMSE", "R²", "MAPE", "Residual plots"],
            "generation": ["BLEU", "ROUGE", "BERTScore", "Perplexity", "Human evaluation"],
            "detection": ["mAP@50", "mAP@50:95", "Precision-Recall curve", "IoU"],
            "segmentation": ["mIoU", "Dice coefficient", "Pixel accuracy", "Boundary F1"],
            "control": ["Episode return", "Sample efficiency", "Policy entropy", "Stability"],
        }

        metrics = metric_map.get(problem_type, ["Accuracy", "F1", "AUC"])

        return f"""## Evaluation Strategy

### Primary Metrics
{chr(10).join(f'- **{m}**' for m in metrics)}

### Experimental Protocol
- **Test set**: Held-out, never touched during training or tuning
- **Cross-validation**: 5-fold stratified CV for final model selection
- **Baseline comparisons**: Compare against 3+ established baselines
- **Ablation studies**: Verify contribution of each novel component

### Statistical Significance
- Report mean ± std over 3–5 random seeds
- Use McNemar's test or Wilcoxon signed-rank test for significance
- Report p < 0.05 threshold

### Out-of-Distribution Evaluation
- Test on data from different distributions
- Report performance degradation under domain shift

### Efficiency Metrics
- Inference time (ms/sample)
- Peak GPU memory during inference
- Model size (MB)

### Production Monitoring
- Track metric drift over time
- Alert on distribution shift (KL divergence threshold)
- Set up shadow mode testing for new model versions
"""

    def _layers_md(self, arch: ArchitectureDNA) -> str:
        if not arch.layers:
            return "  (none)"
        return "\n".join(f"    {l.name}: {l.type} {l.params}" for l in arch.layers)

    def _connections_md(self, arch: ArchitectureDNA) -> str:
        if not arch.connections:
            return "  (sequential)"
        return "\n".join(
            f"    {c.from_layer} → {c.to_layer} [{c.connection_type}]"
            for c in arch.connections
        )

    def _training_md(self, arch: ArchitectureDNA) -> str:
        if not arch.training:
            return "  (default training config)"
        t = arch.training
        return (
            f"  optimizer={t.optimizer}, lr={t.learning_rate}, "
            f"scheduler={t.scheduler}, batch={t.batch_size}, "
            f"epochs={t.epochs}, loss={t.loss_function}"
        )

    def _compute_md(self, arch: ArchitectureDNA) -> str:
        if not arch.compute:
            return "  (unknown)"
        c = arch.compute
        return (
            f"  {c.parameters_millions}M params | "
            f"GPU: {c.gpu_memory_gb}GB | "
            f"Time: {c.training_time_estimate} | "
            f"HW: {c.recommended_hardware}"
        )
