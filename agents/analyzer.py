"""
NeuroForge Analyzer Agent
Deep ML problem decomposition: identifies paradigms, candidate families, and data challenges.
"""
from __future__ import annotations
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.schemas import AgentRole, TaskDomain
from core.memory import memory_graph
from core.prompts import ANALYZER_SYSTEM


class AnalyzerAgent(BaseAgent):
    role = AgentRole.ANALYZER

    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem = input_data.get("problem_statement", "")
        orchestrator_result = input_data.get("orchestrator_result", {})
        dataset_profile = input_data.get("dataset_profile", None)

        # Build rich context from memory graph
        context = memory_graph.build_agent_context(session_id, self.role)

        dataset_info = ""
        if dataset_profile:
            dp = dataset_profile
            dataset_info = (
                f"\nDataset Profile:\n"
                f"  Format: {dp.format}, Rows: {dp.rows}, Columns: {dp.columns}\n"
                f"  Size: {dp.size_mb:.1f}MB\n"
                f"  Inferred Task: {dp.inferred_task}\n"
                f"  Quality Score: {dp.data_quality_score:.2f}\n"
                f"  Recommendations: {', '.join(dp.recommendations[:3])}"
            )

        user_msg = f"""Problem Statement:
{problem}

Orchestrator Analysis:
- Domain: {orchestrator_result.get('domain', 'unknown')}
- Complexity: {orchestrator_result.get('complexity', 'intermediate')}
- Core Challenges: {', '.join(orchestrator_result.get('core_challenges', []))}
- Estimated Family: {orchestrator_result.get('estimated_architecture_family', 'unknown')}
- Special Considerations: {orchestrator_result.get('special_considerations', 'None')}
{dataset_info}

Cognitive Memory:
{context}

Perform a deep problem analysis and return the structured JSON."""

        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        content, tokens = await self._call_llm(messages, temperature=0.2, max_tokens=2048)
        parsed = self._extract_json(content)

        if "_parse_failed" in parsed:
            # Provide sensible defaults on parse failure
            parsed = {
                "problem_type": "classification",
                "input_modalities": ["tabular"],
                "output_type": "scalar",
                "label_regime": "fully_supervised",
                "data_challenges": [],
                "candidate_families": [
                    {
                        "family": "transformer",
                        "suitability": 0.8,
                        "pros": ["strong performance", "flexible"],
                        "cons": ["compute heavy"],
                        "variants": []
                    }
                ],
                "recommended_paradigm": "supervised",
                "preprocessing_pipeline": ["normalize"],
                "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
            }

        critique = await self._self_critique(content)
        self._record_trace(
            session_id=session_id,
            input_summary=problem[:120],
            output_summary=(
                f"type={parsed.get('problem_type')}, "
                f"regime={parsed.get('label_regime')}, "
                f"paradigm={parsed.get('recommended_paradigm')}"
            ),
            full_output=content,
            tokens_used=tokens,
            confidence=0.92,
            critique=critique,
        )

        return parsed
