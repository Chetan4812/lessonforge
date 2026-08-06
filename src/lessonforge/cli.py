"""LessonForge CLI — all commands as Typer stubs (M0).

Commands implemented as stubs here will be fleshed out in later milestones.
Every command is already wired to the correct future module so the CLI surface
is frozen from day 1.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="lessonforge",
    help="Self-evaluating agentic lesson content generator.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# ── run ───────────────────────────────────────────────────────────────────────


@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Topic to generate a lesson for."),
    provider: str = typer.Option("openai", "--provider", help="LLM provider (openai | mock)."),
    inject_error: str | None = typer.Option(
        None,
        "--inject-error",
        help="Demo mode: inject a known defect (factual | jargon | remove-example | ...).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream node events."),
    seed: int | None = typer.Option(None, "--seed", help="Override random seed."),
) -> None:
    """[bold green]Generate, evaluate, and ship[/bold green] a beginner lesson for TOPIC."""
    console.print(
        Panel(
            f"[bold]lessonforge run[/bold]\n"
            f"topic=[cyan]{topic}[/cyan]  provider=[cyan]{provider}[/cyan]"
            + (f"  inject-error=[red]{inject_error}[/red]" if inject_error else ""),
            title="[bold magenta]LessonForge[/bold magenta]",
        )
    )
    console.print("[yellow]⚙  M0 stub — graph execution wired in M3.[/yellow]")


# ── evaluate ──────────────────────────────────────────────────────────────────


@app.command()
def evaluate(
    file: str = typer.Option(..., "--file", "-f", help="Path to a lesson markdown file."),
    topic: str = typer.Option(..., "--topic", "-t", help="Topic the lesson covers."),
    provider: str = typer.Option("openai", "--provider", help="LLM provider."),
) -> None:
    """[bold]Evaluate[/bold] an existing lesson file without generating."""
    console.print(f"[yellow]⚙  M0 stub — evaluate wired in M5. file={file} topic={topic}[/yellow]")


# ── ground ────────────────────────────────────────────────────────────────────


@app.command()
def ground(
    topic: str = typer.Option(..., "--topic", "-t", help="Topic to retrieve grounding for."),
    k: int = typer.Option(8, "--k", help="Number of chunks to retrieve."),
) -> None:
    """[bold]Retrieve[/bold] grounding chunks from the corpus (M2)."""
    from rich.table import Table

    from lessonforge.config import AppConfig
    from lessonforge.grounding.retriever import retrieve

    cfg = AppConfig()

    console.print(f"\n[bold magenta]LessonForge Ground[/bold magenta] · topic=[cyan]{topic}[/cyan] k=[cyan]{k}[/cyan]\n")

    with console.status("[bold green]Building / loading index and retrieving chunks…"):
        pack = retrieve(query=topic, config=cfg)

    table = Table(title=f"Retrieved {len(pack.chunks)} chunks · corpus v{pack.corpus_version}", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Heading", style="yellow")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Preview", max_width=60)

    for chunk in pack.chunks:
        preview = chunk.text[:120].replace("\n", " ") + ("…" if len(chunk.text) > 120 else "")
        table.add_row(chunk.id, chunk.title[:40], str(len(chunk.text.split())), preview)

    console.print(table)
    console.print(f"\n[green]✓[/green] Corpus version: [dim]{pack.corpus_version}[/dim]")



# ── batch ─────────────────────────────────────────────────────────────────────


@app.command()
def batch(
    topics_file: str = typer.Option(
        ..., "--topics-file", help="Path to a file with one topic per line."
    ),
    provider: str = typer.Option("openai", "--provider", help="LLM provider."),
) -> None:
    """[bold]Run[/bold] the pipeline over multiple topics from a file."""
    console.print(f"[yellow]⚙  M0 stub — batch wired in M7. topics_file={topics_file}[/yellow]")


# ── evolve ────────────────────────────────────────────────────────────────────


@app.command()
def evolve(
    since: str = typer.Option("7d", "--since", help="Mine failures from the last N days/runs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show proposed changes without promoting."),
) -> None:
    """[bold]Self-evolution[/bold]: mine failures, propose prompt/rubric patches, validate, gate."""
    console.print(f"[yellow]⚙  M0 stub — evolve wired in M8. since={since} dry_run={dry_run}[/yellow]")


# ── report ────────────────────────────────────────────────────────────────────


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to render."),
    open_browser: bool = typer.Option(False, "--open", help="Open report.html in browser."),
) -> None:
    """[bold]Render[/bold] an HTML report for a completed run."""
    console.print(f"[yellow]⚙  M0 stub — report wired in M9. run_id={run_id}[/yellow]")


# ── memory ────────────────────────────────────────────────────────────────────

memory_app = typer.Typer(help="Inspect the memory layer.")
app.add_typer(memory_app, name="memory")


@memory_app.command("guardrails")
def memory_guardrails() -> None:
    """List active guardrails."""
    console.print("[yellow]⚙  M0 stub — memory wired in M7.[/yellow]")


@memory_app.command("failures")
def memory_failures(
    top: int = typer.Option(10, "--top", help="Show top N failure clusters."),
) -> None:
    """List top failure mode clusters."""
    console.print(f"[yellow]⚙  M0 stub — memory wired in M7. top={top}[/yellow]")


# ── replay ────────────────────────────────────────────────────────────────────


@app.command()
def replay(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to replay from trace."),
) -> None:
    """[bold]Replay[/bold] a past run using cached LLM responses."""
    console.print(f"[yellow]⚙  M0 stub — replay wired in M9. run_id={run_id}[/yellow]")


# ── export ────────────────────────────────────────────────────────────────────


@app.command()
def export(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to export."),
    to: str = typer.Option("notion", "--to", help="Export target (notion)."),
) -> None:
    """[bold]Export[/bold] the final lesson to an external platform."""
    console.print(f"[yellow]⚙  M0 stub — export wired in M9. run_id={run_id} to={to}[/yellow]")
