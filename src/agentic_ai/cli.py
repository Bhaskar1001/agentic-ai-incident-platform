"""Interactive CLI demo for the AI Engineering & Incident Resolution Platform.

Provides a single ``demo`` command that runs a simulated incident through the
full LangGraph workflow and prints a human-readable diagnostic report.

Usage::

    uv run agentic-ai-demo
    uv run agentic-ai-demo --scenario connection_pool_exhaustion
    uv run agentic-ai-demo --scenario healthy
    uv run agentic-ai-demo --description "Database latency spike on payment-svc" \\
        --scenario connection_pool_exhaustion \\
        --post-remediation-scenario healthy

Notes
-----
- No credentials are ever printed.  The description, scenario, and all output
  are free of environment variables and secrets.
- All observability data comes from ``handle_incident()``.  This module does
  not duplicate workflow logic.
- Typer is already installed as a transitive dependency; no new dependency is
  added.
"""

from __future__ import annotations

from typing import Optional

import typer

from agentic_ai.tools.mock_tools import Scenario
from agentic_ai.workflows.incident import handle_incident

app = typer.Typer(
    name="agentic-ai-demo",
    help="Interactive CLI showcase for the AI Engineering & Incident Resolution Platform.",
    add_completion=False,
)

_SCENARIO_CHOICES = [s.value for s in Scenario]

_DEFAULT_DESCRIPTIONS: dict[Scenario, str] = {
    Scenario.CONNECTION_POOL_EXHAUSTION: (
        "High error rate and database connection timeouts on payment-service. "
        "P95 latency has exceeded 5 000 ms for the last 10 minutes."
    ),
    Scenario.UPSTREAM_DEPENDENCY_FAILURE: (
        "Elevated 503 errors and payment processing timeouts reported on payment-service."
    ),
    Scenario.HEALTHY: ("Routine health check for payment service; minor retry warning logged."),
}

_OUTCOME_LABELS: dict[str, str] = {
    "resolve": "RESOLVED",
    "escalate": "ESCALATED",
    "monitor": "MONITOR",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_outcome(outcome: str | None) -> str:
    if not outcome:
        return "UNKNOWN"
    return _OUTCOME_LABELS.get(outcome, outcome.upper())


def _safe_str(value: object) -> str:
    """Convert a value to its string representation, never raising."""
    try:
        return str(value) if value is not None else "-"
    except Exception:  # noqa: BLE001
        return "<unprintable>"


def _print_section(title: str) -> None:
    typer.echo(f"\n{'-' * 60}")
    typer.echo(f"  {title}")
    typer.echo("-" * 60)


def _print_kv(label: str, value: object, indent: int = 2) -> None:
    pad = " " * indent
    typer.echo(f"{pad}{label:<28}{_safe_str(value)}")


# ---------------------------------------------------------------------------
# CLI command -- single default command (no subcommand prefix needed)
# ---------------------------------------------------------------------------


@app.command()
def demo(
    description: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--description",
        "-d",
        help=(
            "Incident description or alert text to submit. "
            "If omitted, a scenario-appropriate default description is used."
        ),
    ),
    scenario: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--scenario",
        "-s",
        help=(
            f"Simulation scenario to use. "
            f"Choices: {', '.join(_SCENARIO_CHOICES)}. "
            f"Default: {Scenario.CONNECTION_POOL_EXHAUSTION.value}."
        ),
    ),
    post_remediation_scenario: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--post-remediation-scenario",
        help=(f"Post-remediation scenario override. Choices: {', '.join(_SCENARIO_CHOICES)}."),
    ),
) -> None:
    """Run a simulated incident through the full workflow and print a diagnostic report."""
    # -----------------------------------------------------------------------
    # Validate scenario choices before touching the workflow
    # -----------------------------------------------------------------------
    resolved_scenario: Scenario
    if scenario is not None:
        if scenario not in _SCENARIO_CHOICES:
            typer.echo(
                f"Error: invalid scenario '{scenario}'. "
                f"Choose from: {', '.join(_SCENARIO_CHOICES)}.",
                err=True,
            )
            raise typer.Exit(code=1)
        resolved_scenario = Scenario(scenario)
    else:
        resolved_scenario = Scenario.CONNECTION_POOL_EXHAUSTION

    resolved_post: Scenario | None = None
    if post_remediation_scenario is not None:
        if post_remediation_scenario not in _SCENARIO_CHOICES:
            typer.echo(
                f"Error: invalid post-remediation-scenario '{post_remediation_scenario}'. "
                f"Choose from: {', '.join(_SCENARIO_CHOICES)}.",
                err=True,
            )
            raise typer.Exit(code=1)
        resolved_post = Scenario(post_remediation_scenario)

    effective_description = (
        description if description is not None else _DEFAULT_DESCRIPTIONS[resolved_scenario]
    )

    # -----------------------------------------------------------------------
    # Print preamble
    # -----------------------------------------------------------------------
    typer.echo()
    typer.echo("=" * 62)
    typer.echo("   AI Engineering & Incident Resolution Platform Demo")
    typer.echo("=" * 62)
    _print_section("Input")
    _print_kv("Scenario:", resolved_scenario.value)
    _print_kv(
        "Description:",
        effective_description[:80] + ("..." if len(effective_description) > 80 else ""),
    )

    typer.echo()
    typer.echo("  Running incident workflow ...  (this may take a moment)")

    # -----------------------------------------------------------------------
    # Run workflow
    # -----------------------------------------------------------------------
    try:
        result = handle_incident(
            incident_description=effective_description,
            scenario=resolved_scenario,
            post_remediation_scenario=resolved_post,
        )
    except Exception as exc:  # noqa: BLE001
        # Never expose stack traces or raw exception messages that might
        # contain secrets or internal paths.
        typer.echo(f"\nError: workflow execution failed - {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from exc

    # -----------------------------------------------------------------------
    # Extract fields safely
    # -----------------------------------------------------------------------
    outcome: str | None = result.get("next_step")
    severity_assessment = result.get("severity_assessment")
    verification = result.get("verification")
    trace = result.get("trace")
    incident_id: str = result.get("incident_id", "-")
    correlation_id: str = result.get("correlation_id", "-")

    # -----------------------------------------------------------------------
    # Severity
    # -----------------------------------------------------------------------
    severity_label = "-"
    if severity_assessment is not None:
        try:
            if hasattr(severity_assessment, "severity"):
                severity_label = str(severity_assessment.severity.value)
            elif isinstance(severity_assessment, dict):
                severity_label = str(severity_assessment.get("severity", "-"))
        except Exception:  # noqa: BLE001
            severity_label = "-"

    # -----------------------------------------------------------------------
    # Diagnostic
    # -----------------------------------------------------------------------
    diagnostic_decision = "-"
    diagnostic_trigger = "-"
    diagnostic_reason = "-"
    if trace is not None and trace.diagnostic is not None:
        diag = trace.diagnostic
        diagnostic_decision = _safe_str(getattr(diag, "decision", "-"))
        diagnostic_trigger = _safe_str(getattr(diag, "trigger_type", "-"))
        diagnostic_reason = _safe_str(getattr(diag, "primary_reason", "-"))

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------
    verification_status = "-"
    verification_attempts = "-"
    if verification is not None:
        try:
            if hasattr(verification, "status"):
                verification_status = _safe_str(verification.status.value)
            elif isinstance(verification, dict):
                verification_status = str(verification.get("status", "-"))
            if hasattr(verification, "attempts"):
                verification_attempts = str(verification.attempts)
            elif isinstance(verification, dict):
                verification_attempts = str(verification.get("attempts", "-"))
        except Exception:  # noqa: BLE001
            pass

    # -----------------------------------------------------------------------
    # Trace summary
    # -----------------------------------------------------------------------
    stage_count = "-"
    tool_count = "-"
    total_duration = "-"
    if trace is not None:
        try:
            stage_count = str(len(trace.stages)) if trace.stages is not None else "-"
            tool_count = str(len(trace.tools)) if trace.tools is not None else "-"
            if trace.total_duration_ms is not None:
                total_duration = f"{trace.total_duration_ms:.1f} ms"
        except Exception:  # noqa: BLE001
            pass

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    _print_section("Outcome")
    _print_kv("Incident ID:", incident_id)
    _print_kv("Correlation ID:", correlation_id)
    _print_kv("Result:", _format_outcome(outcome))
    _print_kv("Severity:", severity_label)

    _print_section("Diagnostic")
    _print_kv("Decision:", diagnostic_decision)
    _print_kv("Trigger type:", diagnostic_trigger)
    _print_kv("Primary reason:", diagnostic_reason)

    if verification is not None:
        _print_section("Verification")
        _print_kv("Status:", verification_status)
        _print_kv("Attempts:", verification_attempts)

    _print_section("Trace Summary")
    _print_kv("Stages recorded:", stage_count)
    _print_kv("Tool invocations:", tool_count)
    _print_kv("Total duration:", total_duration)

    typer.echo()
    typer.echo("-" * 60)
    typer.echo("  Demo complete.")
    typer.echo("-" * 60)
    typer.echo()


def main() -> None:
    """Entry point for the ``agentic-ai-demo`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
