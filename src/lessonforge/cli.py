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
    provider: str = typer.Option("groq", "--provider", help="LLM provider (groq | openai | mock)."),
    inject_error: str | None = typer.Option(
        None,
        "--inject-error",
        help="Demo mode: inject a known defect (factual | jargon | remove-example | ...).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream node events."),
    seed: int | None = typer.Option(None, "--seed", help="Override random seed."),
) -> None:
    """[bold green]Generate, evaluate, and ship[/bold green] a beginner lesson for TOPIC."""
    import logging

    from rich.table import Table

    from lessonforge import pipeline

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    console.print(
        Panel(
            f"[bold]lessonforge run[/bold]\n"
            f"topic=[cyan]{topic}[/cyan]  provider=[cyan]{provider}[/cyan]"
            + (f"  inject-error=[red]{inject_error}[/red]" if inject_error else ""),
            title="[bold magenta]LessonForge[/bold magenta]",
        )
    )

    with console.status("[bold green]Running pipeline…"):
        state = pipeline.run(
            topic=topic,
            provider=provider,
            inject_error=inject_error,
            seed=seed,
        )

    verdict = state.verdict
    ship = verdict.ship_decision if verdict else "UNKNOWN"
    colour = {"SHIP": "green", "RETRY": "yellow", "ESCALATE": "red"}.get(ship, "white")

    console.print(f"\n[bold {colour}]▶ Verdict: {ship}[/bold {colour}]  (attempt {state.attempt})\n")

    if state.structural_report:
        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Dimension", style="dim")
        table.add_column("Result", justify="center")
        table.add_column("Reason", max_width=55)

        for r in state.structural_report.results:
            icon = "[green]✓ PASS[/green]" if r.verdict == "PASS" else "[red]✗ FAIL[/red]"
            table.add_row(r.check_id, r.dimension, icon, r.reason[:120])

        console.print(table)

    if ship == "SHIP" and state.lesson:
        console.print(f"\n[green]✓[/green] Lesson shipped: [bold]{state.lesson.title}[/bold]")
        console.print(f"  Run ID: [dim]{state.run_id}[/dim]")
        console.print(f"  Output: [dim]out/{state.run_id}/lesson.md[/dim]")
    elif ship == "ESCALATE":
        console.print(
            "[red]✗[/red] Lesson could not be fixed within the attempt budget. "
            "See [dim]out/{state.run_id}/report.json[/dim] for details."
        )




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
    since: str = typer.Option("30d", "--since", help="Mine failures from the last N days (e.g. 7d, 30d)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show proposed changes without promoting."),
    auto: bool = typer.Option(False, "--auto", help="Skip human approval gate (CI mode)."),
    provider: str = typer.Option("groq", "--provider", help="LLM provider for the analyst call."),
    top_n: int = typer.Option(10, "--top-n", help="Max failure clusters to analyse."),
) -> None:
    """[bold]Self-evolution[/bold]: mine failures, diagnose root causes, propose and gate prompt fixes."""
    from rich.table import Table

    from lessonforge.config import AppConfig
    from lessonforge.evolve import analyst as analyst_mod
    from lessonforge.evolve import miner as miner_mod
    from lessonforge.evolve import promoter as promoter_mod
    from lessonforge.llm.gateway import Gateway
    from lessonforge.memory import db

    cfg = AppConfig()
    require_human = cfg._raw.get("evolve", {}).get("require_human_approval", True)

    # Parse since string (e.g. "30d", "7d")
    since_days = int(since.rstrip("d"))

    console.print(
        Panel(
            f"[bold]lessonforge evolve[/bold]\n"
            f"since=[cyan]{since}[/cyan]  top-n=[cyan]{top_n}[/cyan]"
            + ("  [yellow]DRY RUN[/yellow]" if dry_run else ""),
            title="[bold magenta]LessonForge Evolve[/bold magenta]",
        )
    )

    # ── Step 1: Mine ──────────────────────────────────────────────────────────
    db.init_db()
    with console.status("[green]Mining failure clusters…"):
        clusters = miner_mod.mine(since_days=since_days, top_n=top_n)
        pass_rate = miner_mod.first_attempt_pass_rate(since_days=since_days)

    console.print(f"\n[bold]Found {len(clusters)} failure cluster(s)[/bold] — first-attempt pass rate: [cyan]{pass_rate * 100:.1f}%[/cyan]\n")

    if not clusters:
        console.print("[green]✓[/green] No recurring failures found. Nothing to evolve.")
        return

    # ── Step 2: Diagnose ──────────────────────────────────────────────────────
    with console.status("[green]Calling evolve analyst…"):
        gw = Gateway(provider=provider, run_id="evolve")
        diagnoses = analyst_mod.diagnose(clusters, pass_rate, gw, cfg)

    if not diagnoses:
        console.print("[yellow]⚠[/yellow] Analyst returned no diagnoses.")
        return

    # ── Step 3: Display proposals ─────────────────────────────────────────────
    table = Table(title="Proposed Fixes", show_lines=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Root Cause", style="dim")
    table.add_column("Fix Type", style="yellow")
    table.add_column("Proposed Fix", max_width=60)

    for d in diagnoses:
        fix_text = (
            d.proposed_fix.guardrail_text or d.proposed_fix.description
        )
        table.add_row(d.check_id, d.root_cause, d.proposed_fix.fix_type, fix_text[:100])
    console.print(table)

    # ── Step 4: Gate ──────────────────────────────────────────────────────────
    proceed = False
    if dry_run:
        console.print("\n[yellow]DRY RUN — no changes written.[/yellow]")
        proceed = False
    elif auto or not require_human:
        console.print("\n[green]Auto-promoting (--auto or require_human_approval=false).[/green]")
        proceed = True
    else:
        confirm = typer.confirm(f"\nPromote {len(diagnoses)} fix(es) to active guardrails?")
        proceed = confirm

    if not proceed:
        console.print("[dim]Aborted.[/dim]")
        return

    # ── Step 5: Promote ───────────────────────────────────────────────────────
    with console.status("[green]Promoting approved fixes…"):
        changes = promoter_mod.promote(diagnoses, dry_run=False, config=cfg)

    for ch in changes:
        console.print(f"[green]✓[/green] {ch}")

    console.print(f"\n[bold green]Evolution complete — {len(changes)} change(s) promoted.[/bold green]")
    console.print("  Next run: [dim]lessonforge run[/dim] will pick up the new guardrails automatically.\n")




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
