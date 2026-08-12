"""
NeuroForge Chief Architect Agent
Designs one or more candidate ArchitectureDNA objects.
This is the heart of NeuroForge — the deepest ML expertise lives here.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from models.schemas import (
    AgentRole, ArchitectureDNA, LayerSpec, ConnectionSpec,
    TrainingSpec, ComputeEstimate, TaskDomain, ComplexityLevel
)
from core.memory import memory_graph
from core.prompts import ARCHITECT_SYSTEM


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    async def design_architecture(
        self,
        session_id: str,
        problem: str,
        analysis: Dict[str, Any],
        orchestrator: Dict[str, Any],
        constraints: str = "",
        framework: str = "PyTorch",
        candidate_index: int = 0,
        count: int = 1,
    ) -> ArchitectureDNA:
        """Design one candidate architecture."""
        context = memory_graph.build_agent_context(session_id, self.role)

        diversity_hint = ""
        if count > 1:
            approaches = ["efficient and lightweight", "high-accuracy research-grade", "novel hybrid approach"]
            diversity_hint = f"\nCANDIDATE APPROACH: Design the {approaches[candidate_index % 3]} variant."

        candidate_families = analysis.get("candidate_families", [])
        families_str = json.dumps(candidate_families[:3], indent=2) if candidate_families else "[]"

        user_msg = f"""Design a complete ML architecture for this problem.

PROBLEM: {problem}

ANALYSIS RESULTS:
  Problem Type: {analysis.get('problem_type', 'classification')}
  Input Modalities: {analysis.get('input_modalities', [])}
  Output Type: {analysis.get('output_type', 'scalar')}
  Label Regime: {analysis.get('label_regime', 'fully_supervised')}
  Data Challenges: {analysis.get('data_challenges', [])}
  Recommended Paradigm: {analysis.get('recommended_paradigm', 'supervised')}
  Preprocessing: {analysis.get('preprocessing_pipeline', [])}

CANDIDATE FAMILIES IDENTIFIED:
{families_str}

DOMAIN: {orchestrator.get('domain', 'deep_learning')}
COMPLEXITY: {orchestrator.get('complexity', 'advanced')}
CORE CHALLENGES: {orchestrator.get('core_challenges', [])}
FRAMEWORK: {framework}
CONSTRAINTS: {constraints or "None"}
{diversity_hint}

COGNITIVE MEMORY:
{context}

Design the complete architecture. Be innovative yet practical. 
Justify every choice. Return ONLY the JSON architecture specification."""

        messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.4, max_tokens=3000)
        parsed = self._extract_json(content)

        # Build ArchitectureDNA from parsed JSON
        arch = self._build_dna(parsed)

        # Record to memory graph
        memory_graph.record_architecture(session_id, arch)

        critique = await self._self_critique(content)
        self._record_trace(
            session_id=session_id,
            input_summary=problem[:120],
            output_summary=(
                f"Designed: {arch.name} | "
                f"params={arch.compute.parameters_millions if arch.compute else '?'}M | "
                f"novelty={arch.novelty_score:.2f}"
            ),
            full_output=content,
            tokens_used=tokens,
            confidence=0.88,
            critique=critique,
        )

        return arch

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Design multiple candidate architectures."""
        count = input_data.get("compare_count", 1)
        architectures = []

        for i in range(min(count, 3)):
            arch = await self.design_architecture(
                session_id=session_id,
                problem=input_data.get("problem_statement", ""),
                analysis=input_data.get("analysis", {}),
                orchestrator=input_data.get("orchestrator_result", {}),
                constraints=input_data.get("constraints", ""),
                framework=input_data.get("preferred_framework", "PyTorch"),
                candidate_index=i,
                count=count,
            )
            architectures.append(arch)

        return {"architectures": architectures}

    # ─── DNA Construction ─────────────────────────────────────────────────────

    def _build_dna(self, parsed: Dict[str, Any]) -> ArchitectureDNA:
        """Convert raw JSON from the LLM into a typed ArchitectureDNA object."""
        layers = []
        for ldata in parsed.get("layers", []):
            layers.append(LayerSpec(
                name=ldata.get("name", "layer"),
                type=ldata.get("type", "linear"),
                params=ldata.get("params", {}),
                notes=ldata.get("notes", ""),
            ))

        connections = []
        for cdata in parsed.get("connections", []):
            connections.append(ConnectionSpec(
                from_layer=cdata.get("from_layer", ""),
                to_layer=cdata.get("to_layer", ""),
                connection_type=cdata.get("connection_type", "sequential"),
            ))

        training = None
        td = parsed.get("training", {})
        if td:
            loss = td.get("loss_function", "cross_entropy")
            if not isinstance(loss, str):
                loss = "cross_entropy"
            sched = td.get("scheduler")
            if not isinstance(sched, str):
                sched = None
            curr = td.get("curriculum")
            if curr and not all(isinstance(c, str) for c in curr):
                curr = [str(c) for c in curr]
            training = TrainingSpec(
                optimizer=str(td.get("optimizer", "AdamW")),
                learning_rate=float(td.get("learning_rate", 3e-4)),
                scheduler=sched,
                batch_size=int(td.get("batch_size", 32)),
                epochs=int(td.get("epochs", 100)),
                loss_function=loss,
                metrics=[str(m) for m in td.get("metrics", ["accuracy"])],
                regularization={k: v for k, v in td.get("regularization", {}).items() if isinstance(v, (int, float, str))},
                curriculum=curr,
            )

        compute = None
        cd = parsed.get("compute", {})
        if cd:
            compute = ComputeEstimate(
                parameters_millions=float(cd.get("parameters_millions", 10.0)),
                flops_billions=cd.get("flops_billions"),
                gpu_memory_gb=cd.get("gpu_memory_gb"),
                training_time_estimate=cd.get("training_time_estimate", "Unknown"),
                recommended_hardware=cd.get("recommended_hardware", "GPU with 8GB+ VRAM"),
            )

        # Infer domain from tags/backbone
        backbone = parsed.get("backbone", "neural_network")
        domain = self._infer_domain(backbone, parsed.get("tags", []))

        return ArchitectureDNA(
            name=parsed.get("name", "NeuroForge Architecture"),
            domain=domain,
            complexity=ComplexityLevel.ADVANCED,
            backbone=backbone,
            layers=layers,
            connections=connections,
            training=training,
            compute=compute,
            novelty_score=float(parsed.get("novelty_score", 0.5)),
            feasibility_score=float(parsed.get("feasibility_score", 0.9)),
            anti_patterns_found=parsed.get("anti_patterns_found", []),
            tags=parsed.get("tags", []),
            rationale=parsed.get("rationale", ""),
        )

    def _infer_domain(self, backbone: str, tags: List[str]) -> TaskDomain:
        all_text = (backbone + " " + " ".join(tags)).lower()
        if any(k in all_text for k in ["rl", "policy", "reward", "actor", "critic", "q-network", "dqn", "ppo", "sac"]):
            return TaskDomain.REINFORCEMENT_LEARNING
        if any(k in all_text for k in ["lora", "qlora", "finetune", "llm", "instruction", "rlhf", "dpo"]):
            return TaskDomain.LLM_FINETUNING
        if any(k in all_text for k in ["gnn", "graph", "gcn", "gat", "sage"]):
            return TaskDomain.GRAPH_NEURAL
        if any(k in all_text for k in ["timeseries", "time_series", "lstm", "gru", "temporal"]):
            return TaskDomain.TIME_SERIES
        if any(k in all_text for k in ["cnn", "vit", "resnet", "convnet", "image", "vision"]):
            return TaskDomain.COMPUTER_VISION
        if any(k in all_text for k in ["bert", "gpt", "t5", "nlp", "text", "transformer"]):
            return TaskDomain.NLP
        if any(k in all_text for k in ["multimodal", "clip", "flamingo", "multi-modal"]):
            return TaskDomain.MULTIMODAL
        if any(k in all_text for k in ["xgboost", "random forest", "svm", "logistic", "classical"]):
            return TaskDomain.CLASSICAL_ML
        return TaskDomain.DEEP_LEARNING
