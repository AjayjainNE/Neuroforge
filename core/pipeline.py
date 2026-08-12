"""
NeuroForge APEX Pipeline
Adaptive Progressive EXpert pipeline executor.
Coordinates all agents in the correct order, passing outputs as inputs.
Levels:
  L1 sketch    → orchestrator only
  L2 detailed  → orchestrator + analyzer + architect
  L3 complete  → full pipeline (all agents)
  L4 research  → full pipeline + novelty + critique loops
"""
from __future__ import annotations
import inspect
import time
from typing import Any, Callable, Dict, List, Optional


from models.schemas import (
    DesignRequest, DesignResponse, AgentStep, ArchitectureDNA,
    ComputeEstimate, TaskDomain, ComplexityLevel, OutputDepth,
)
OD = OutputDepth
from core.memory import memory_graph
from agents import (
    OrchestratorAgent,
    AnalyzerAgent,
    ArchitectAgent,
    CoderAgent,
    OptimizerAgent,
    ValidatorAgent,
    ExplainerAgent,
)
from utils.dataset_inspector import DatasetInspector



class APEXPipeline:
    """
    The APEX Pipeline orchestrates all agents in sequence.
    Each agent's output enriches the shared memory graph and
    is passed as context to downstream agents.
    """

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.analyzer = AnalyzerAgent()
        self.architect = ArchitectAgent()
        self.coder = CoderAgent()
        self.optimizer = OptimizerAgent()
        self.validator = ValidatorAgent()
        self.explainer = ExplainerAgent()
        self.inspector = DatasetInspector()

    async def _progress(
        self,
        agent: str,
        msg: str,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Fire progress_callback, awaiting it if it is a coroutine function."""
        if progress_callback:
            result = progress_callback(agent, msg)
            if inspect.isawaitable(result):
                await result

    async def run(
        self,
        request: DesignRequest,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> DesignResponse:
        t0 = time.time()
        total_tokens = 0
        steps: List[AgentStep] = []

        # ── Session setup ─────────────────────────────────────────────────────
        session = memory_graph.get_or_create(request.session_id)
        sid = session.session_id

        # ── Dataset profiling (optional) ──────────────────────────────────────
        dataset_profile = None
        if request.dataset_path:
            await self._progress("inspector", f"Profiling dataset: {request.dataset_path}", progress_callback)
            try:
                dataset_profile = await self.inspector.profile(
                    request.dataset_path, target_column=None
                )
                session.dataset_profile = dataset_profile
            except Exception as e:
                await self._progress("inspector", f"Dataset profiling failed: {e}", progress_callback)

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: Orchestrator (all depths)
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("orchestrator", "Analysing problem...", progress_callback)
        orch_result = await self.orchestrator.run(sid, {
            "problem_statement": request.problem_statement,
            "constraints": request.constraints or "",
            "depth": request.depth.value,
            "preferred_framework": request.preferred_framework or "PyTorch",
        })
        total_tokens += session.traces[-1].tokens_used if session.traces else 0
        steps.append(AgentStep(
            agent="orchestrator",
            status="complete",
            summary=f"Domain: {orch_result['domain'].value}, Complexity: {orch_result['complexity'].value}",
            tokens_used=session.traces[-1].tokens_used if session.traces else 0,
        ))

        if request.depth == OD.SKETCH:
            return self._sketch_response(sid, session, orch_result, steps, total_tokens, t0)

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 2+: Analyzer
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("analyzer", "Decomposing problem structure...", progress_callback)
        analysis = await self.analyzer.run(sid, {
            "problem_statement": request.problem_statement,
            "orchestrator_result": orch_result,
            "dataset_profile": dataset_profile,
        })
        total_tokens += session.traces[-1].tokens_used if session.traces else 0
        steps.append(AgentStep(
            agent="analyzer",
            status="complete",
            summary=f"Type: {analysis.get('problem_type')}, Regime: {analysis.get('label_regime')}",
            tokens_used=session.traces[-1].tokens_used if session.traces else 0,
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 2+: Architect — design candidates
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("architect", f"Designing {request.compare_count} architecture candidate(s)...", progress_callback)
        arch_result = await self.architect.run(sid, {
            "problem_statement": request.problem_statement,
            "orchestrator_result": orch_result,
            "analysis": analysis,
            "constraints": request.constraints or "",
            "preferred_framework": request.preferred_framework or "PyTorch",
            "compare_count": request.compare_count,
        })
        architectures: List[ArchitectureDNA] = arch_result.get("architectures", [])
        total_tokens += session.traces[-1].tokens_used if session.traces else 0
        steps.append(AgentStep(
            agent="architect",
            status="complete",
            summary=f"Designed {len(architectures)} architecture(s): " +
                    ", ".join(a.name for a in architectures),
            tokens_used=session.traces[-1].tokens_used if session.traces else 0,
        ))

        if request.depth == OD.DETAILED:
            return await self._detailed_response(
                sid, session, request, orch_result, analysis, architectures, steps, total_tokens, t0
            )

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 3+: Optimizer
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("optimizer", "Optimising hyperparameters & architecture...", progress_callback)
        optimizer_result = await self.optimizer.run(sid, {
            "problem_statement": request.problem_statement,
            "orchestrator_result": orch_result,
            "architectures": architectures,
            "analysis": analysis,
            "constraints": request.constraints or "",
        })
        total_tokens += session.traces[-1].tokens_used if session.traces else 0
        steps.append(AgentStep(
            agent="optimizer",
            status="complete",
            summary=f"Est. accuracy gain: {optimizer_result.get('estimated_improvement', {}).get('accuracy_gain_pct', '?')}%",
            tokens_used=session.traces[-1].tokens_used if session.traces else 0,
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 3+: Coder
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("coder", "Generating production code...", progress_callback)
        coder_result = await self.coder.run(sid, {
            "problem_statement": request.problem_statement,
            "architectures": architectures,
            "preferred_framework": request.preferred_framework or "PyTorch",
            "dataset_path": request.dataset_path,
            "dataset_profile": dataset_profile,
            "analysis": analysis,
            "optimizer_result": optimizer_result,
        })
        generated_code = coder_result.get("code", "")
        total_tokens += coder_result.get("tokens_used", 0)
        steps.append(AgentStep(
            agent="coder",
            status="complete",
            summary=f"Generated {len(generated_code.splitlines())} lines of {coder_result.get('framework')} code",
            tokens_used=coder_result.get("tokens_used", 0),
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 3+: Validator
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("validator", "Validating architecture...", progress_callback)
        validator_result = await self.validator.run(sid, {
            "problem_statement": request.problem_statement,
            "architectures": architectures,
            "analysis": analysis,
            "generated_code": generated_code,
        })
        total_tokens += session.traces[-1].tokens_used if session.traces else 0
        steps.append(AgentStep(
            agent="validator",
            status="complete",
            summary=f"Approval: {validator_result.get('approval')} | Issues: {len(validator_result.get('issues', []))}",
            tokens_used=session.traces[-1].tokens_used if session.traces else 0,
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 3+: Explainer
        # ─────────────────────────────────────────────────────────────────────
        await self._progress("explainer", "Generating documentation...", progress_callback)
        explainer_result = await self.explainer.run(sid, {
            "problem_statement": request.problem_statement,
            "architectures": architectures,
            "analysis": analysis,
            "optimizer_result": optimizer_result,
            "validator_result": validator_result,
            "generated_code": generated_code,
            "orchestrator_result": orch_result,
        })
        total_tokens += explainer_result.get("tokens_used", 0)
        steps.append(AgentStep(
            agent="explainer",
            status="complete",
            summary="Full design document generated",
            tokens_used=explainer_result.get("tokens_used", 0),
        ))

        # ── Build final response ──────────────────────────────────────────────
        recommended = architectures[0] if architectures else self._fallback_arch()
        session.final_architecture = recommended

        compute = recommended.compute or ComputeEstimate(
            parameters_millions=0,
            recommended_hardware="GPU",
        )

        anti_patterns = (
            validator_result.get("all_anti_patterns", []) +
            recommended.anti_patterns_found
        )

        return DesignResponse(
            session_id=sid,
            request_summary=request.problem_statement[:200],
            domain=orch_result["domain"],
            complexity=orch_result["complexity"],
            depth=request.depth,
            agent_pipeline=steps,
            architectures=architectures,
            recommended=recommended,
            full_design=explainer_result.get("full_design", ""),
            generated_code=generated_code,
            anti_patterns=list(set(anti_patterns)),
            compute_estimate=compute,
            novelty_analysis=explainer_result.get("novelty_analysis", ""),
            training_guide=explainer_result.get("training_guide", ""),
            evaluation_strategy=explainer_result.get("evaluation_strategy", ""),
            total_tokens_used=total_tokens,
            processing_time_sec=round(time.time() - t0, 2),
        )

    # ─── Partial responses for lower depth levels ────────────────────────────

    def _sketch_response(self, sid, session, orch_result, steps, tokens, t0) -> DesignResponse:
        arch = self._sketch_arch(orch_result)
        return DesignResponse(
            session_id=sid,
            request_summary="",
            domain=orch_result["domain"],
            complexity=orch_result["complexity"],
            depth=OD.SKETCH,
            agent_pipeline=steps,
            architectures=[arch],
            recommended=arch,
            full_design=(
                f"# Architecture Sketch\n\n"
                f"Domain: {orch_result['domain'].value}\n"
                f"Family: {orch_result['estimated_architecture_family']}\n"
                f"Challenges: {orch_result['core_challenges']}"
            ),
            generated_code="# Run at depth='complete' for full code generation",
            anti_patterns=[],
            compute_estimate=ComputeEstimate(parameters_millions=0, recommended_hardware="TBD"),
            novelty_analysis="",
            training_guide="",
            evaluation_strategy="",
            total_tokens_used=tokens,
            processing_time_sec=round(time.time() - t0, 2),
        )

    async def _detailed_response(
        self, sid, session, request, orch_result, analysis,
        architectures, steps, tokens, t0
    ) -> DesignResponse:
        recommended = architectures[0] if architectures else self._fallback_arch()
        compute = recommended.compute or ComputeEstimate(
            parameters_millions=0, recommended_hardware="GPU"
        )

        guide = await self.explainer._training_guide(recommended, analysis)
        tokens += guide[1]

        return DesignResponse(
            session_id=sid,
            request_summary=request.problem_statement[:200],
            domain=orch_result["domain"],
            complexity=orch_result["complexity"],
            depth=OD.DETAILED,
            agent_pipeline=steps,
            architectures=architectures,
            recommended=recommended,
            full_design=f"# {recommended.name}\n\n{recommended.rationale}",
            generated_code="# Run at depth='complete' for full code generation",
            anti_patterns=recommended.anti_patterns_found,
            compute_estimate=compute,
            novelty_analysis="",
            training_guide=guide[0],
            evaluation_strategy=self.explainer._build_evaluation_strategy(recommended, analysis),
            total_tokens_used=tokens,
            processing_time_sec=round(time.time() - t0, 2),
        )

    def _sketch_arch(self, orch_result: Dict) -> ArchitectureDNA:
        return ArchitectureDNA(
            name=f"NeuroForge-{orch_result.get('estimated_architecture_family', 'Net').title()}",
            domain=orch_result["domain"],
            complexity=orch_result["complexity"],
            backbone=orch_result.get("estimated_architecture_family", "neural_network"),
            rationale="Sketch — run at depth='complete' for full design.",
        )

    def _fallback_arch(self) -> ArchitectureDNA:
        return ArchitectureDNA(
            name="NeuroForge-Baseline",
            domain=TaskDomain.DEEP_LEARNING,
            complexity=ComplexityLevel.INTERMEDIATE,
            backbone="transformer",
            rationale="Fallback baseline architecture.",
        )
