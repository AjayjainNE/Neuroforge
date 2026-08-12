"""
NeuroForge CLI
Interactive terminal interface for testing the APEX agent pipeline
without running the full FastAPI server.

Usage:
    python cli.py design "Design a transformer for protein folding prediction"
    python cli.py design "..." --depth complete --framework PyTorch --compare 2
    python cli.py analyze ./data/train.csv
    python cli.py interactive
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.rule import Rule

from config import settings
from models.schemas import OutputDepth
from models.schemas import DesignRequest, OutputDepth as OD, DatasetAnalysisRequest
from core.pipeline import APEXPipeline
from utils.dataset_inspector import DatasetInspector

app = typer.Typer(
    name="neuroforge",
    help="NeuroForge APEX — ML Architecture Design AI Agent",
    add_completion=False,
)
console = Console()
pipeline = APEXPipeline()


def _banner():
    console.print(Panel.fit(
        "[bold cyan]NeuroForge APEX[/bold cyan]\n"
        "[dim]Adaptive Progressive EXpert ML Architecture Design Agent[/dim]\n"
        f"[dim]Provider: {settings.LLM_PROVIDER.value.upper()} | Model: {settings.primary_model}[/dim]",
        border_style="cyan",
    ))


@app.command()
def design(
    problem: str = typer.Argument(..., help="Natural language problem statement"),
    depth: str = typer.Option("complete", "--depth", "-d", help="sketch|detailed|complete|research"),
    framework: str = typer.Option("PyTorch", "--framework", "-f", help="PyTorch|TensorFlow|JAX"),
    constraints: Optional[str] = typer.Option(None, "--constraints", "-c", help="Hardware/latency/memory constraints"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Path to local dataset file"),
    compare: int = typer.Option(1, "--compare", help="Number of candidate architectures (1-3)"),
    save_code: Optional[str] = typer.Option(None, "--save-code", help="Save generated code to file"),
    save_doc: Optional[str] = typer.Option(None, "--save-doc", help="Save design document to file"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Design an ML/DL/RL architecture from a problem statement."""
    _banner()

    try:
        depth_enum = OD(depth)
    except ValueError:
        console.print(f"[red]Invalid depth '{depth}'. Choose: sketch|detailed|complete|research[/red]")
        raise typer.Exit(1)

    request = DesignRequest(
        problem_statement=problem,
        depth=depth_enum,
        constraints=constraints,
        dataset_path=dataset,
        preferred_framework=framework,
        compare_count=min(max(compare, 1), 3),
    )

    progress_msgs = []

    def progress_cb(agent: str, msg: str):
        progress_msgs.append((agent, msg))

    async def _run():
        return await pipeline.run(request, progress_callback=progress_cb)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("Running APEX pipeline...", total=None)

        async def _run_with_progress():
            orig_cb = progress_cb

            async def cb(agent: str, msg: str):
                prog.update(task, description=f"[cyan]{agent}[/cyan]: {msg}")
                orig_cb(agent, msg)

            return await pipeline.run(request, progress_callback=cb)

        response = asyncio.run(_run_with_progress())

    if json_out:
        console.print_json(json.dumps(response.model_dump(mode="json"), indent=2))
        return

    # ── Display results ───────────────────────────────────────────────────────

    console.print()
    console.print(Rule("[bold green]APEX Pipeline Complete[/bold green]"))

    # Agent pipeline table
    agent_table = Table(title="Agent Pipeline", show_header=True, header_style="bold cyan")
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Status", style="green")
    agent_table.add_column("Summary")
    agent_table.add_column("Tokens", justify="right")
    for step in response.agent_pipeline:
        agent_table.add_row(
            step.agent, f"✓ {step.status}", step.summary, str(step.tokens_used)
        )
    console.print(agent_table)

    # Architecture summary
    rec = response.recommended
    arch_table = Table(title=f"Recommended: {rec.name}", show_header=False, box=None)
    arch_table.add_column("Field", style="bold")
    arch_table.add_column("Value")
    arch_table.add_row("Backbone", rec.backbone)
    arch_table.add_row("Domain", rec.domain.value)
    arch_table.add_row("Tags", ", ".join(rec.tags))
    arch_table.add_row("Novelty", f"{rec.novelty_score:.2f}/1.0")
    arch_table.add_row("Feasibility", f"{rec.feasibility_score:.2f}/1.0")
    if rec.compute:
        arch_table.add_row("Parameters", f"{rec.compute.parameters_millions}M")
        arch_table.add_row("Hardware", rec.compute.recommended_hardware)
        arch_table.add_row("Training time", rec.compute.training_time_estimate or "Unknown")
    if rec.training:
        arch_table.add_row("Optimizer", rec.training.optimizer)
        arch_table.add_row("Learning rate", str(rec.training.learning_rate))
        arch_table.add_row("Loss", rec.training.loss_function)
    console.print()
    console.print(arch_table)

    # Anti-patterns
    if response.anti_patterns:
        console.print()
        console.print(Panel(
            "\n".join(f"⚠  {ap}" for ap in response.anti_patterns),
            title="[yellow]Anti-patterns Detected[/yellow]",
            border_style="yellow",
        ))

    # Compute estimate
    ce = response.compute_estimate
    console.print()
    console.print(Panel(
        f"Parameters: {ce.parameters_millions}M\n"
        f"GPU Memory: {ce.gpu_memory_gb or 'N/A'}GB\n"
        f"Training: {ce.training_time_estimate or 'N/A'}\n"
        f"Hardware: {ce.recommended_hardware}",
        title="[blue]Compute Estimate[/blue]",
        border_style="blue",
    ))

    # Stats
    console.print()
    console.print(
        f"[dim]Session: {response.session_id} | "
        f"Tokens: {response.total_tokens_used:,} | "
        f"Time: {response.processing_time_sec}s[/dim]"
    )

    # Design document
    if response.full_design:
        console.print()
        console.print(Rule("[bold]Design Document[/bold]"))
        console.print(Markdown(response.full_design[:4000]))
        if len(response.full_design) > 4000:
            console.print("[dim]... (truncated — save to file for full document)[/dim]")

    # Generated code
    if response.generated_code and len(response.generated_code) > 50:
        console.print()
        console.print(Rule("[bold]Generated Code[/bold]"))
        preview = "\n".join(response.generated_code.splitlines()[:60])
        console.print(Syntax(preview, "python", theme="monokai", line_numbers=True))
        if len(response.generated_code.splitlines()) > 60:
            console.print("[dim]... (truncated — use --save-code to save full file)[/dim]")

    # Save outputs
    if save_code and response.generated_code:
        Path(save_code).write_text(response.generated_code, encoding="utf-8")
        console.print(f"[green]✓ Code saved: {save_code}[/green]")

    if save_doc and response.full_design:
        Path(save_doc).write_text(response.full_design, encoding="utf-8")
        console.print(f"[green]✓ Design doc saved: {save_doc}[/green]")


@app.command()
def analyze(
    dataset_path: str = typer.Argument(..., help="Path to dataset file (CSV/Parquet/JSON)"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    hint: Optional[str] = typer.Option(None, "--hint", "-h", help="Task type hint"),
):
    """Profile a local dataset file."""
    _banner()
    inspector = DatasetInspector()

    async def _run():
        return await inspector.profile(dataset_path, target_column=target, task_hint=hint)

    with Progress(SpinnerColumn(), TextColumn("Profiling dataset..."), console=console, transient=True):
        profile = asyncio.run(_run())

    table = Table(title=f"Dataset Profile: {Path(dataset_path).name}", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Format", profile.format)
    table.add_row("Rows", f"{profile.rows:,}")
    table.add_row("Columns", str(profile.columns))
    table.add_row("Size", f"{profile.size_mb:.1f} MB")
    table.add_row("Inferred Task", str(profile.inferred_task))
    table.add_row("Target Column", profile.suggested_target or "Unknown")
    table.add_row("Quality Score", f"{profile.data_quality_score:.3f}/1.000")
    console.print(table)

    col_table = Table(title="Column Profiles", show_header=True, header_style="cyan")
    col_table.add_column("Column")
    col_table.add_column("Type")
    col_table.add_column("Nulls %")
    col_table.add_column("Unique")
    col_table.add_column("Sample")
    for cp in profile.column_profiles[:20]:
        col_table.add_row(
            cp.name, cp.dtype, f"{cp.null_pct:.1f}%",
            str(cp.unique_count),
            str(cp.sample_values[:2])[:40],
        )
    console.print(col_table)

    if profile.recommendations:
        console.print(Panel(
            "\n".join(f"• {r}" for r in profile.recommendations),
            title="[yellow]Recommendations[/yellow]",
            border_style="yellow",
        ))


@app.command()
def interactive():
    """Start an interactive REPL for iterative architecture design."""
    _banner()
    console.print("[dim]Type your ML problem. Commands: /depth, /framework, /compare, /quit[/dim]\n")

    session_id = None
    depth = "complete"
    framework = "PyTorch"
    compare = 1

    while True:
        try:
            user_input = console.input("[bold cyan]neuroforge>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.startswith("/depth "):
            depth = user_input.split(" ", 1)[1].strip()
            console.print(f"[green]Depth set to: {depth}[/green]")
            continue
        if user_input.startswith("/framework "):
            framework = user_input.split(" ", 1)[1].strip()
            console.print(f"[green]Framework set to: {framework}[/green]")
            continue
        if user_input.startswith("/compare "):
            compare = int(user_input.split(" ", 1)[1].strip())
            console.print(f"[green]Compare count set to: {compare}[/green]")
            continue
        if user_input.startswith("/session"):
            console.print(f"[cyan]Current session: {session_id or 'none'}[/cyan]")
            continue

        try:
            depth_enum = OD(depth)
        except ValueError:
            depth_enum = OD.COMPLETE

        request = DesignRequest(
            problem_statement=user_input,
            depth=depth_enum,
            preferred_framework=framework,
            compare_count=compare,
            session_id=session_id,
        )

        async def _run():
            return await pipeline.run(request)

        with Progress(SpinnerColumn(), TextColumn("APEX thinking..."), console=console, transient=True):
            try:
                response = asyncio.run(_run())
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

        session_id = response.session_id
        rec = response.recommended

        console.print(Panel(
            f"[bold]{rec.name}[/bold] | {rec.backbone} | "
            f"novelty={rec.novelty_score:.2f} | feasibility={rec.feasibility_score:.2f}\n\n"
            f"{rec.rationale[:300]}...",
            title=f"[green]Architecture (session={session_id[:8]}...)[/green]",
            border_style="green",
        ))

        if response.generated_code and len(response.generated_code) > 100:
            preview = "\n".join(response.generated_code.splitlines()[:30])
            console.print(Syntax(preview, "python", theme="monokai"))


if __name__ == "__main__":
    app()
