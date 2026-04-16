"""MEDLEY-BENCH admin CLI for generation and orchestration."""
from __future__ import annotations

from pathlib import Path

import click

from src.core.db import init_db, get_db
from src.admin.db.models import list_cases, get_known_answer_cases

DEFAULT_DB = Path("data") / "database.db"
SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"


MAIN_HELP = """
medley-bench: Behavioral metacognition benchmark for LLMs.

\b
Quick start:
  pip install medley-bench
  medley-bench benchmark --models "ollama/gemma3:12b"
\b
With a live judge (Gemini Flash recommended):
  export GOOGLE_API_KEY="AI..."
  medley-bench benchmark \\
    --models "ollama/gemma3:12b" \\
    --judge-model gemini-2.5-flash

\b
Supported model ID patterns:
  claude-*             Anthropic (ANTHROPIC_API_KEY)
  gpt-*, o1-*, o3-*    OpenAI (OPENAI_API_KEY)
  gemini-*             Google (GOOGLE_API_KEY)
  ollama/model         Ollama local or cloud (no key needed)
  org/model            OpenRouter (OPENROUTER_API_KEY)

\b
Run 'medley-bench examples' for more usage examples.
Run 'medley-bench COMMAND --help' for command-specific options.
"""


@click.group(help=MAIN_HELP)
def cli():
    pass


@cli.command()
@click.option("--db", default=str(DEFAULT_DB), help="Database path")
def init(db: str):
    """Initialize the SQLite database and register all prompt versions."""
    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path, SCHEMA_PATH)
    # Register prompts
    from src.admin.generation.prompts.registry import register_all_prompts
    with get_db(db_path) as conn:
        n = register_all_prompts(conn)
    click.echo(f"Database initialized at {db_path} ({n} prompts registered)")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB), help="Database path")
def status(db: str):
    """Show generation progress."""
    db_path = Path(db)
    if not db_path.exists():
        click.echo("Database not found. Run 'medley-bench init' first.")
        return

    with get_db(db_path) as conn:
        cases = list_cases(conn)
        ka_cases = get_known_answer_cases(conn)

        by_domain = {}
        probes = {"trap": 0, "dose_response": 0, "minimal": 0, "counterfactual": 0, "error_detection": 0}
        for c in cases:
            by_domain.setdefault(c.domain, 0)
            by_domain[c.domain] += 1
            if c.is_trap: probes["trap"] += 1
            if c.is_dose_response: probes["dose_response"] += 1
            if c.is_minimal_instruction: probes["minimal"] += 1
            if c.is_counterfactual: probes["counterfactual"] += 1
            if c.is_error_detection: probes["error_detection"] += 1

        # Analyst + consensus counts
        analyst_count = conn.execute("SELECT COUNT(DISTINCT case_id) FROM analyst_responses").fetchone()[0]
        consensus_count = conn.execute("SELECT COUNT(*) FROM consensus").fetchone()[0]
        claim_count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

        click.echo(f"Total cases: {len(cases)}")
        for domain, count in sorted(by_domain.items()):
            click.echo(f"  {domain}: {count}")
        click.echo(f"Known-answer instances: {len(ka_cases)} (target: 20)")
        click.echo(f"Probes: trap={probes['trap']}, dose_response={probes['dose_response']}, "
                    f"minimal={probes['minimal']}, counterfactual={probes['counterfactual']}, "
                    f"error_detection={probes['error_detection']}")
        click.echo(f"Analyst coverage: {analyst_count}/{len(cases)} cases")
        click.echo(f"Consensus built: {consensus_count}/{len(cases)} cases")
        click.echo(f"Claims extracted: {claim_count}")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def known_answers(db: str):
    """Create known-answer instances from seeds."""
    from src.admin.generation.known_answer import create_known_answer_cases
    db_path = Path(db)
    count = create_known_answer_cases(db_path)
    click.echo(f"Created {count} known-answer cases")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--domain", type=click.Choice(["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]),
              help="Generate for a single domain")
def load_seeds(db: str, domain: str | None):
    """Load standard case seeds into the database."""
    from src.admin.generation.seeds import get_seeds
    from src.admin.db.models import Case, insert_case, get_case
    db_path = Path(db)
    count = 0
    with get_db(db_path) as conn:
        for d, seeds in get_seeds(domain).items():
            for seed in seeds:
                if get_case(conn, seed["seed_id"]) is None:
                    insert_case(conn, Case(
                        case_id=seed["seed_id"], domain=d, seed_data=seed,
                    ))
                    count += 1
    click.echo(f"Loaded {count} case seeds")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--model", default="anthropic/claude-sonnet-4.6", help="Premium model for expansion")
@click.option("--domain", type=click.Choice(["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]))
def expand(db: str, model: str, domain: str | None):
    """Expand seeds into full vignettes using a premium model."""
    import asyncio
    from src.admin.generation.expand import expand_all_seeds

    db_path = Path(db)
    click.echo(f"Expanding seeds with {model}...")
    result = asyncio.run(expand_all_seeds(db_path, model, domain))
    click.echo(f"  expanded={result['expanded']}, skipped={result['skipped']}, failed={result['failed']}")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--domain", type=click.Choice(["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]),
              required=True)
@click.option("--max-concurrent", default=5, help="Max concurrent API calls")
def generate_cases(db: str, domain: str, max_concurrent: int):
    """Run Step 1 designers on all cases in a domain (async)."""
    import asyncio
    from src.admin.generation.step1_designers.pipeline import process_case
    from src.admin.db.models import get_case

    db_path = Path(db)
    with get_db(db_path) as conn:
        cases = conn.execute(
            "SELECT case_id, seed_data FROM cases WHERE domain = ? AND vignette IS NULL",
            (domain,),
        ).fetchall()

    if not cases:
        click.echo(f"No unprocessed cases in {domain}")
        return

    click.echo(f"Running Step 1 on {len(cases)} cases in {domain}...")

    async def _run():
        for row in cases:
            import json
            seed = json.loads(row["seed_data"])
            click.echo(f"  Processing {row['case_id']}...")
            await process_case(row["case_id"], seed, domain, db_path)

    asyncio.run(_run())
    click.echo("Step 1 complete.")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--domain", type=click.Choice(["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]),
              required=True)
def generate_analysts(db: str, domain: str):
    """Run Step 2 analysts on all cases in a domain (async)."""
    import asyncio
    from src.admin.generation.step2_analysts.pipeline import process_case

    db_path = Path(db)
    with get_db(db_path) as conn:
        cases = conn.execute(
            "SELECT case_id FROM cases WHERE domain = ? AND vignette IS NOT NULL",
            (domain,),
        ).fetchall()

    if not cases:
        click.echo(f"No cases ready for analysts in {domain} (run generate-cases first)")
        return

    click.echo(f"Running Step 2 on {len(cases)} cases in {domain}...")

    async def _run():
        for row in cases:
            click.echo(f"  Processing {row['case_id']}...")
            await process_case(row["case_id"], db_path)

    asyncio.run(_run())
    click.echo("Step 2 complete.")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def build_consensus(db: str):
    """Build jackknife consensus for all cases with analyst responses."""
    import asyncio
    from src.admin.generation.consensus.jackknife import build_and_save_consensus

    db_path = Path(db)
    with get_db(db_path) as conn:
        cases = conn.execute("""
            SELECT DISTINCT ar.case_id FROM analyst_responses ar
            LEFT JOIN consensus c ON ar.case_id = c.case_id
            WHERE c.case_id IS NULL
        """).fetchall()

    if not cases:
        click.echo("No cases need consensus building.")
        return

    click.echo(f"Building consensus for {len(cases)} cases...")

    async def _run():
        for row in cases:
            await build_and_save_consensus(row["case_id"], db_path)

    asyncio.run(_run())
    click.echo("Consensus building complete.")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def designate_probes(db: str):
    """Designate dose-response, minimal-instruction, trap, and counterfactual probes."""
    from src.admin.generation.probes import run_all_designations
    results = run_all_designations(Path(db))
    for probe_type, count in results.items():
        click.echo(f"  {probe_type}: {count}")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def validate(db: str):
    """Run all quality gates (G1-G15)."""
    from src.admin.validation import run_quality_gates
    db_path = Path(db)
    results = run_quality_gates(db_path)
    any_failed = False
    for gate in results:
        status_icon = "✓" if gate["passed"] else ("⚠" if gate["severity"] == "WARNING" else "✗")
        if not gate["passed"] and gate["severity"] == "BLOCKING":
            any_failed = True
        click.echo(f"  {status_icon} {gate['gate']}: {gate['message']} [{gate['severity']}]")
    if any_failed:
        click.echo("\nBLOCKING gates failed. Fix before export.")
    else:
        click.echo("\nAll blocking gates passed.")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--output", default="data/export", help="Export directory")
def export(db: str, output: str):
    """Export benchmark data for Part 2 deployment."""
    from src.admin.db.export import export_for_benchmark
    export_for_benchmark(Path(db), Path(output))
    click.echo(f"Exported to {output}")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def prompts(db: str):
    """List all prompt versions and their hashes."""
    db_path = Path(db)
    if not db_path.exists():
        click.echo("Database not found.")
        return

    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT prompt_name, version, content_hash, created_at FROM prompt_versions ORDER BY prompt_name, version"
        ).fetchall()
        if not rows:
            click.echo("No prompt versions recorded yet.")
            return
        for row in rows:
            click.echo(f"  {row['prompt_name']} v{row['version']} [{row['content_hash']}] ({row['created_at']})")


# ── Feature 2: Incremental Collection ──────────────────────────

@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--model", required=True, help="Model ID (e.g. ollama/llama3.2, deepseek/deepseek-v3.2)")
@click.option("--role", type=click.Choice(["designer", "analyst"]), required=True)
@click.option("--domain", type=click.Choice(["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]))
@click.option("--concurrent", default=0, help="Max concurrent API calls (0=auto: 4 for small, 2 for large, 1 for cloud)")
@click.option("--max-tokens", default=10000, help="Max output tokens (use 20000+ for thinking models)")
@click.option("--force-recollect", is_flag=True, default=False, help="Delete existing responses for this model before collecting (for re-runs)")
def collect(db: str, model: str, role: str, domain: str | None, concurrent: int, max_tokens: int, force_recollect: bool):
    """Collect responses for a single model incrementally (skips already done)."""
    import asyncio
    from src.admin.generation.collect import collect_designer_responses, collect_analyst_responses

    db_path = Path(db)

    # Auto-detect concurrency based on model size
    # Host: Mac Studio 512GB RAM — all models under 120B fit comfortably
    if concurrent == 0:
        m = model.lower()
        if any(x in m for x in ["675b", "cloud"]):
            concurrent = 1  # Cloud/API models — respect rate limits
        elif any(x in m for x in ["120b", "scout"]):
            concurrent = 2  # Very large (>100B) — some GPU contention
        else:
            concurrent = 4  # Everything else fits in 512GB RAM — parallel OK
        click.echo(f"  Auto concurrency: {concurrent} (based on model size)")

    click.echo(f"  Max tokens: {max_tokens}")

    if force_recollect:
        table = "analyst_responses" if role == "analyst" else "designer_responses"
        with get_db(db_path) as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE model_id = ?", (model,)).fetchone()[0]
            if count > 0:
                click.echo(f"  FORCE RECOLLECT: Deleting {count} existing {role} responses for {model}")
                conn.execute(f"DELETE FROM {table} WHERE model_id = ?", (model,))
                conn.commit()
            # Also clear failures
            conn.execute("DELETE FROM collection_failures WHERE model_id = ? AND role = ?", (model, role))
            conn.commit()

    async def _run():
        if role == "designer":
            return await collect_designer_responses(db_path, model, domain, concurrent)
        else:
            return await collect_analyst_responses(db_path, model, domain, concurrent, max_tokens=max_tokens)

    result = asyncio.run(_run())
    click.echo(f"  {role} {model}: completed={result['completed']}, "
               f"skipped={result['skipped']}, failed={result['failed']}, "
               f"total={result['total']}")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def collect_failures(db: str):
    """Show collection failures and their retry status."""
    from src.admin.generation.collect import get_failure_summary, get_unresolved_failures

    db_path = Path(db)
    summary = get_failure_summary(db_path)
    if not summary:
        click.echo("No collection failures recorded.")
        return

    click.echo(f"\n  {'Model':<45} {'Role':>10} {'Fails':>6} {'Fixed':>6}")
    click.echo(f"  {'─' * 70}")
    for s in summary:
        click.echo(f"  {s['model_id']:<45} {s['role']:>10} {s['failures']:>6} {s['resolved']:>6}")

    unresolved = get_unresolved_failures(db_path)
    if unresolved:
        click.echo(f"\n  {len(unresolved)} unresolved failures. Run 'medley-bench collect' again to retry.")


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def analyze_analysts(db: str):
    """Analyze quality of analyst model responses — instruction following, patterns, agreement."""
    from src.benchmark.analysis.analyst_quality import print_full_report
    print_full_report(Path(db))


@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
def collect_status(db: str):
    """Show which models have responded to which cases."""
    from src.admin.generation.collect import get_collection_status

    status = get_collection_status(Path(db))
    click.echo(f"Total cases: {status['total_cases']}")
    click.echo(f"\nModel response counts:")
    click.echo(f"  {'Model':<45} {'Designer':>10} {'Analyst':>10}")
    click.echo(f"  {'─' * 65}")
    for model_id, counts in sorted(status["summary"].items()):
        click.echo(f"  {model_id:<45} {counts['designer']:>10} {counts['analyst']:>10}")


# ── Feature 1: Standalone Benchmark Runner ─────────────────────

@cli.command()
@click.option("--models", required=True, help="Comma-separated model IDs")
@click.option("--data", default=None, help="Data directory. Defaults to bundled dataset (data/metacognition/v1.0/).")
@click.option("--output", default="results", help="Output directory for results")
@click.option("--domain", multiple=True, help="Filter to specific domains")
@click.option("--db", default=None, help="DB path to reuse analyst responses as Step A (saves 130 API calls/model)")
@click.option("--judge-model", default=None, help="Live judge model id (e.g. 'gemini-2.5-flash', 'qwen3-coder:480b-cloud'). If unset, only precomputed judges are used.")
@click.option("--judge-base-url", default=None, help="OpenAI-compatible endpoint for the judge. Defaults to Google's Gemini endpoint. Use 'http://localhost:11434/v1' for Ollama.")
@click.option("--judge-api-key", default=None, help="API key for the judge endpoint. Defaults to $GOOGLE_API_KEY, or 'ollama' for local endpoints.")
@click.option("--n-instances", type=int, default=None, help="Smoke-test limit: only run first N instances per domain.")
def benchmark(
    models: str, data: str, output: str, domain: tuple, db: str,
    judge_model: str, judge_base_url: str, judge_api_key: str, n_instances: int,
):
    """Run standalone benchmark on given models.

\b
Examples:
  # Cloud model via OpenRouter
  medley-bench benchmark --models "anthropic/claude-haiku-4.5"
\b
  # Local Ollama model
  medley-bench benchmark --models "ollama/gemma3:12b"
\b
  # With live judge (Gemini Flash)
  medley-bench benchmark --models "ollama/gemma3:12b" \\
    --judge-model gemini-2.5-flash

\b
  # Ollama cloud as judge (no external API key needed)
  medley-bench benchmark --models "ollama/gemma3:12b" \\
    --judge-model gemma4:31b-cloud --judge-base-url http://localhost:11434/v1

\b
  # Smoke test: first 3 instances per domain
  medley-bench benchmark --models "ollama/gemma3:12b" \\
    --n-instances 3

\b
  # Multiple models in one run
  medley-bench benchmark --models "ollama/gemma3:12b,ollama/qwen3:32b"
\b
Notes:
  Full 130-instance run = 390 target calls + 130 judge calls = 520 API calls.
  Expect ~1 hour on fast APIs, several hours on slower providers.
  Results are saved incrementally and runs are resumable.
    """
    import asyncio
    import os
    from src.benchmark.runner import run_benchmark

    # Resolve data directory: user-provided, or bundled package data.
    if data is None:
        from data import get_default_data_dir
        data = str(get_default_data_dir())

    model_list = [m.strip() for m in models.split(",")]
    domains = list(domain) if domain else None

    # Resolve judge API key from env if not supplied explicitly.
    if judge_model and not judge_api_key:
        if judge_base_url and "localhost" in judge_base_url:
            judge_api_key = "ollama"
        else:
            judge_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

    if db:
        click.echo(f"Running benchmark: {len(model_list)} models on {data} (reusing Step A from {db})")
    else:
        click.echo(f"Running benchmark: {len(model_list)} models on {data}")
    if judge_model:
        click.echo(f"Live judge: {judge_model} @ {judge_base_url or '(default Gemini endpoint)'}")
    if n_instances:
        click.echo(f"Smoke-test limit: first {n_instances} instances per domain")

    asyncio.run(run_benchmark(
        model_list, data, output, domains, db_path=db,
        judge_model=judge_model, judge_base_url=judge_base_url,
        judge_api_key=judge_api_key, n_instances=n_instances,
    ))
    click.echo(f"Results saved to {output}/")


@cli.command()
def about():
    """Show project information, version, and links."""
    from importlib.metadata import version as pkg_version
    try:
        ver = pkg_version("medley-bench")
    except Exception:
        ver = "0.5.x (dev)"
    click.echo(f"""
MEDLEY-BENCH v{ver}
Behavioral Metacognition Under Social Pressure

Measures how LLMs monitor, evaluate, and control their own reasoning
under escalating social-epistemic pressure. Unlike accuracy-focused
benchmarks, MEDLEY-BENCH measures *how models behave when challenged*,
not whether they know the answer.

Three-step protocol:
  Step A (Solo)       Independent analysis + confidence
  Step B-Private      Self-revision (no ensemble)
  Step B-Social       Updating after seeing 8 analyst opinions + consensus

Scoring (MMS = Medley Metacognition Score):
  T1 Reflective Updating   33%   Deterministic
  T2 Social Robustness     33%   Mixed (mostly deterministic)
  T3 Epistemic Articulation 33%  Mixed (judge-dependent)
  75% of total weight is rule-based; 25% uses an LLM judge.

Dataset: 130 instances across 5 domains (medical, troubleshooting,
  code review, architecture, statistical reasoning).

Links:
  PyPI       https://pypi.org/project/medley-bench/
  GitHub     https://github.com/ki-smile/medley-bench
  Dataset    https://www.kaggle.com/datasets/farhadabtahi/medley-bench-data

Citation:
  Abtahi, F., Karbalaie, A., Illueca-Fernandez, E., & Seoane, F. (2026).
  MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition.
  Preprint.

License: Apache 2.0
""")


@cli.command()
def examples():
    """Show usage examples for common tasks."""
    click.echo("""
medley-bench usage examples
============================

1. BENCHMARK A CLOUD MODEL (via OpenRouter)
   export OPENROUTER_API_KEY="sk-or-..."
   medley-bench benchmark --models "anthropic/claude-haiku-4.5"

2. BENCHMARK A LOCAL OLLAMA MODEL
   ollama pull gemma3:12b
   medley-bench benchmark --models "ollama/gemma3:12b"

3. ADD A LIVE JUDGE (recommended: Gemini 2.5 Flash)
   export GOOGLE_API_KEY="AI..."
   medley-bench benchmark \\
     --models "ollama/gemma3:12b" \\
     --judge-model gemini-2.5-flash

4. FULLY OFFLINE (Ollama cloud model as judge)
   medley-bench benchmark \\
     --models "ollama/gemma3:12b" \\
     --judge-model gemma4:31b-cloud \\
     --judge-base-url http://localhost:11434/v1

5. SMOKE TEST (first 3 instances per domain)
   medley-bench benchmark \\
     --models "ollama/gemma3:12b" \\
     --judge-model gemini-2.5-flash \\
     --n-instances 3

6. MULTIPLE MODELS IN ONE RUN
   medley-bench benchmark --models "ollama/gemma3:12b,ollama/qwen3:32b"

7. VIEW LEADERBOARD
   medley-bench leaderboard --results results/

Supported providers:
  claude-*             Anthropic   (ANTHROPIC_API_KEY)
  gpt-*, o1-*, o3-*    OpenAI      (OPENAI_API_KEY)
  gemini-*             Google      (GOOGLE_API_KEY)
  ollama/model         Ollama      (no key needed)
  org/model            OpenRouter  (OPENROUTER_API_KEY)

Notes:
  - Full 130-instance run = 390 target + 130 judge = 520 API calls
  - Expect ~1 hr on fast APIs, several hours on slower providers
  - Results are saved incrementally; runs are resumable
  - Reasoning-model judges (gpt-oss, glm-4.6, etc.) work automatically
""")


@cli.command()
@click.option("--results", default="results", help="Results directory")
def leaderboard(results: str):
    """Display leaderboard from benchmark results."""
    from src.benchmark.runner import build_leaderboard_from_results

    entries = build_leaderboard_from_results(Path(results))
    if not entries:
        click.echo("No results found.")
        return

    click.echo(f"\n  {'#':>2} {'Model':<40} {'Total':>6} {'PvSD':>6} {'T1':>6} {'T2':>6} {'T3':>6} {'N':>4}")
    click.echo(f"  {'─' * 78}")
    for i, e in enumerate(entries, 1):
        t = e["tiers"]
        click.echo(f"  {i:>2} {e['model']:<40} {e['total']:>6.3f} {e['pvsd']:>6.3f} "
                    f"{t.get('reflective_updating',0.5):>6.3f} "
                    f"{t.get('social_robustness',0.5):>6.3f} "
                    f"{t.get('epistemic_articulation',0.5):>6.3f} "
                    f"{e['n_instances']:>4}")


# ── Feature 3: Admin Web Panel ─────────────────────────────────

@cli.command()
@click.option("--db", default=str(DEFAULT_DB))
@click.option("--port", default=8000, help="Port to serve on")
@click.option("--host", default="127.0.0.1")
def web(db: str, port: int, host: str):
    """Start the admin web panel."""
    import uvicorn
    from src.admin.web.app import create_app

    app = create_app(Path(db))
    click.echo(f"Starting admin panel at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
