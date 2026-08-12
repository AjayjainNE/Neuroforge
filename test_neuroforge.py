"""
NeuroForge Production Test Suite
Run: python test_neuroforge.py
Tests all agents with real API calls using the configured provider.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from typing import List

# ── ensure src is on path ─────────────────────────────────────
sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ─── Test Cases ───────────────────────────────────────────────────────────────

TESTS = [
    {
        "id": "T01",
        "name": "Computer Vision — Few-shot Classification",
        "depth": "complete",
        "request": {
            "problem_statement": (
                "Design a deep learning architecture to classify 15 species of rare birds "
                "from field photographs. We have only 200 labeled images per class but access "
                "to ImageNet pretrained weights. The model must run inference in under 50ms "
                "on a single RTX 3080. Accuracy target >90%."
            ),
            "constraints": "RTX 3080 10GB, latency <50ms, PyTorch",
            "preferred_framework": "PyTorch",
            "compare_count": 1,
        },
    },
    {
        "id": "T02",
        "name": "NLP — Multilingual Sentiment with Distillation",
        "depth": "detailed",
        "request": {
            "problem_statement": (
                "Build a sentiment classifier that handles 12 languages including Arabic and "
                "Chinese. We have a CSV dataset with 500K rows: columns are [text, lang, sentiment]. "
                "The model needs to be distilled to under 50M parameters for edge deployment."
            ),
            "constraints": "Max 50M parameters, CPU inference, <100ms",
            "preferred_framework": "PyTorch",
            "compare_count": 1,
        },
    },
    {
        "id": "T03",
        "name": "Reinforcement Learning — Multi-Agent Navigation",
        "depth": "complete",
        "request": {
            "problem_statement": (
                "Design a reinforcement learning system for coordinating 8 autonomous warehouse "
                "robots. Each robot must navigate to pick locations while avoiding collisions. "
                "The environment is partially observable. Use MARL with centralized training "
                "and decentralized execution (CTDE). Reward: throughput + collision penalty."
            ),
            "constraints": "Must support variable number of agents at inference",
            "preferred_framework": "PyTorch",
            "compare_count": 1,
        },
    },
    {
        "id": "T04",
        "name": "LLM Fine-tuning — Domain Adaptation",
        "depth": "detailed",
        "request": {
            "problem_statement": (
                "Fine-tune a 7B parameter LLM on a proprietary medical Q&A dataset of 50K "
                "samples. The dataset is in JSON format with fields: question, context, answer, specialty. "
                "Budget: 2x A100 40GB GPUs. Must use PEFT techniques to fit in memory. "
                "Evaluate with ROUGE-L and domain expert scoring."
            ),
            "constraints": "2x A100 40GB, QLoRA preferred, 4-bit quantization",
            "preferred_framework": "PyTorch",
            "compare_count": 1,
        },
    },
    {
        "id": "T05",
        "name": "Time Series — Multivariate Anomaly Detection",
        "depth": "complete",
        "request": (
            "Design an architecture for real-time anomaly detection across 200 sensor channels "
            "from industrial machinery. Data arrives as 1Hz time series. We need to detect "
            "anomalies within 2 seconds of occurrence with <1% false positive rate. "
            "Local CSV dataset with 18 months of historical readings."
        ),
        "request": {
            "problem_statement": (
                "Design an architecture for real-time anomaly detection across 200 sensor channels "
                "from industrial machinery. Data arrives at 1Hz. Detect anomalies within 2 seconds "
                "with <1% false positive rate. Local parquet dataset with 18 months of history."
            ),
            "constraints": "Edge GPU with 4GB VRAM, latency <2s, streaming inference",
            "preferred_framework": "PyTorch",
            "compare_count": 1,
        },
    },
]


# ─── Test Runner ─────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, test_id, name):
        self.test_id = test_id
        self.name = name
        self.passed = False
        self.error = None
        self.latency_sec = 0.0
        self.tokens = 0
        self.arch_name = ""
        self.domain = ""
        self.novelty = 0.0
        self.has_code = False
        self.has_design = False
        self.approval = ""


async def run_test(test: dict) -> TestResult:
    from models.schemas import DesignRequest, OutputDepth
    from core.pipeline import APEXPipeline

    result = TestResult(test["id"], test["name"])
    pipeline = APEXPipeline()

    try:
        req_data = test["request"]
        depth_map = {"sketch": OutputDepth.SKETCH, "detailed": OutputDepth.DETAILED,
                     "complete": OutputDepth.COMPLETE, "research": OutputDepth.RESEARCH}
        depth = depth_map.get(test.get("depth", "complete"), OutputDepth.COMPLETE)
        request = DesignRequest(**req_data, depth=depth)

        t0 = time.time()
        response = await pipeline.run(request)
        result.latency_sec = round(time.time() - t0, 1)
        result.tokens = response.total_tokens_used
        result.arch_name = response.recommended.name
        result.domain = response.domain.value
        result.novelty = response.recommended.novelty_score
        result.has_code = len(response.generated_code) > 100
        result.has_design = len(response.full_design) > 100
        result.approval = getattr(response, "_validator_approval", "n/a")
        result.passed = True

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


async def run_all_tests(test_ids: List[str] = None):
    console.print(Panel.fit(
        "[bold cyan]NeuroForge APEX — Production Test Suite[/bold cyan]",
        border_style="cyan",
    ))

    tests = TESTS
    if test_ids:
        tests = [t for t in TESTS if t["id"] in test_ids]

    results = []
    for i, test in enumerate(tests, 1):
        console.print(f"\n[cyan][{test['id']}] {test['name']}[/cyan] ({test.get('depth','complete')})")
        result = await run_test(test)
        results.append(result)

        if result.passed:
            console.print(
                f"  [green]✓ PASS[/green] | {result.latency_sec}s | {result.tokens:,} tokens | "
                f"arch={result.arch_name} | domain={result.domain} | novelty={result.novelty:.2f} | "
                f"code={'✓' if result.has_code else '✗'} | design={'✓' if result.has_design else '✗'}"
            )
        else:
            console.print(f"  [red]✗ FAIL[/red] | {result.error}")

    # Summary table
    table = Table(title="\nTest Results Summary", show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Test")
    table.add_column("Status")
    table.add_column("Time")
    table.add_column("Tokens", justify="right")
    table.add_column("Architecture")
    table.add_column("Code")
    table.add_column("Doc")

    passed = sum(1 for r in results if r.passed)
    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            r.test_id, r.name[:35], status,
            f"{r.latency_sec}s", f"{r.tokens:,}",
            r.arch_name[:30] if r.arch_name else r.error[:30] if r.error else "-",
            "✓" if r.has_code else "✗",
            "✓" if r.has_design else "✗",
        )

    console.print(table)
    console.print(
        f"\n[bold]Results: {passed}/{len(results)} passed[/bold]"
        + (" [green]ALL PASS[/green]" if passed == len(results) else " [yellow]SOME FAILED[/yellow]")
    )

    return passed == len(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroForge test suite")
    parser.add_argument("--tests", nargs="*", default=None, help="Test IDs to run (e.g. T01 T03)")
    parser.add_argument("--quick", action="store_true", help="Run only T01 (fastest)")
    args = parser.parse_args()

    if args.quick:
        ids = ["T01"]
    else:
        ids = args.tests

    ok = asyncio.run(run_all_tests(ids))
    sys.exit(0 if ok else 1)
