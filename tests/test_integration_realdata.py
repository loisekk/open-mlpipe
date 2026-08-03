"""Integration tests on a REAL heavy-tail dataset.

Why this file exists:
    The unit tests in tests/ use 200-row synthetic numpy noise with a perfectly
    linear target (conftest.py:sample_dataframe_regression). That data has no
    skew, no outliers, no zeros -- so the heavy-tail code paths in clean.py,
    split.py and tune.py NEVER EXECUTE during the test suite. Bugs like the
    regression split being stratify=None (test_r2=0.05 on House_Price) slip
    through because the toy target is Gaussian and the gate never fires.

    This file runs the FULL PipelineRunner on skewed_target_demo.csv (real-ish
    heavy-tail data, skew~10.8) and asserts accuracy thresholds. If any of the
    three fixes regresses (stratified split, target outlier cleanup, tree-aware
    transform gate), the assertion here will fail loudly.

Run:
    pytest tests/test_integration_realdata.py -v -s
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from open_mlpipe.config.resolver import build_level1_config
from open_mlpipe.core.pipeline import PipelineRunner
from open_mlpipe.utils.typing import TaskType

# ── Real dataset fixtures ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
SKEWED_DEMO_CSV = PROJECT_ROOT / "skewed_target_demo.csv"


def _has_skewed_demo() -> bool:
    return SKEWED_DEMO_CSV.exists() and SKEWED_DEMO_CSV.stat().st_size > 1000


pytestmark = pytest.mark.skipif(
    not _has_skewed_demo(),
    reason="skewed_target_demo.csv not found in repo root -- cannot run real-data integration test",
)


@pytest.fixture(scope="module")
def skewed_demo_pipeline_result():
    """Run the FULL real pipeline on skewed_target_demo.csv and return ctx.

    module-scoped so we only pay the runtime once per test session. We cap
    tuning at 10 trials / 120s so the whole test finishes in ~2 min instead
    of the full 5-min production run. The accuracy assertions still hold at
    this budget because the 3 heavy-tail fixes fire BEFORE tuning.

    We also capture stdout into a StringIO so test 4 can verify the
    stratified-split message printed by split.py -- the message goes to
    stdout (print), not to the pipeline_run_*.log file (which is only
    written by the CLI's logging handler, not when PipelineRunner is
    called directly here).
    """
    csv_path = str(SKEWED_DEMO_CSV)
    config = build_level1_config(csv_path, target="revenue")
    # Cap tuning so the test doesn't take 5 minutes. 10 trials is enough for
    # TPE to find a decent config; the test isn't asserting tuning IMPROVED,
    # just that the pipeline as a whole reaches usable accuracy (which is
    # dominated by the stratified split + outlier cleanup, not the tuning).
    config.tuning.n_trials = 10
    config.tuning.timeout = 120
    # Skip the slow optional stages: explainability (SHAP), MLflow tracking,
    # deployment packaging. We only need the train/eval path for the
    # accuracy assertions.
    config.evaluation.explainability = False
    config.artifacts.mlflow_tracking = False
    config.deployment.enabled = False
    runner = PipelineRunner(config)

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        ctx = runner.run()
    # Stash on the module-level variable instead of an undeclared attribute
    # on PipelineContext (which trips Pylance's "no attribute" check).
    global REAL_DATA_RUN_STDOUT
    REAL_DATA_RUN_STDOUT = stdout_buf.getvalue()

    # Sanity: the pipeline must have produced something
    assert ctx is not None, "PipelineRunner.run() returned None"
    assert ctx.task_type == TaskType.REGRESSION, \
        f"expected regression, got {ctx.task_type!r}"
    return ctx


# ── Test 1: the pipeline actually reaches a usable accuracy ───────────────

def test_pipeline_reaches_usable_r2_on_skewed_target(skewed_demo_pipeline_result):
    """The single most important regression test in the repo.

    Before the 3 fixes (stratified split, outlier cleanup, tree transform gate),
    the pipeline on a heavy-tail target produced test_r2 ~ 0.05. After the
    fixes it produces test_r2 ~ 0.75+.

    We assert test_r2 > 0.60 -- a comfortable floor below the observed 0.75
    that still catches the "0.05 weak model" regression decisively. A flaky
    run might land at 0.68; a genuinely broken regression will land at 0.05
    or below. The threshold separates these worlds with margin to spare.
    """
    ctx = skewed_demo_pipeline_result
    test_r2 = ctx.metrics.get("test_r2")
    assert test_r2 is not None, "test_r2 metric missing from pipeline output"
    assert test_r2 > 0.60, (
        f"REGRESSION: test_r2={test_r2:.4f} dropped below 0.60. The 3 heavy-tail "
        f"fixes (stratified split / outlier cleanup / tree transform gate) likely "
        f"regressed. Expected > 0.60 on skewed_target_demo.csv."
    )


# ── Test 2: MAPE is finite (no division-by-zero from price=0 rows) ────────

def test_mape_is_finite_on_skewed_target(skewed_demo_pipeline_result):
    """Before the outlier cleanup, zero-encoded missing data in the target
    caused MAPE = 9.9e15 (division by zero). After the fix MAPE drops to
    a sensible 17-50%. We assert MAPE < 1e6 -- well above the normal range
    but decisively finite, so a regression that reintroduces zero targets
    trips this test.
    """
    ctx = skewed_demo_pipeline_result
    test_mape = ctx.metrics.get("test_mape")
    assert test_mape is not None, "test_mape metric missing from pipeline output"
    assert test_mape < 1e6, (
        f"REGRESSION: test_mape={test_mape:.4e} is not finite -- target outlier "
        f"cleanup in clean.py likely regressed (zero-encoded missing data is "
        f"reaching the evaluate stage and dividing by zero)."
    )


# ── Test 3: tuning improved over baseline (or at least didn't regress) ────

def test_tuning_match_or_beats_baseline(skewed_demo_pipeline_result):
    """Tuning should match or improve the baseline score. We don't assert
    strict improvement because Optuna runs are stochastic and on a 3-minute
    budget TPE sometimes lands marginally below the sklearn-default baseline.
    A test_r2 drop > 0.15 would indicate a real tuning regression.
    """
    ctx = skewed_demo_pipeline_result
    baseline = ctx.metrics.get("tune_baseline_score")
    tuned = ctx.metrics.get("tuned_best_value")
    assert baseline is not None and tuned is not None, \
        "baseline/tuned metrics missing"
    drop = baseline - tuned
    assert drop < 0.15, (
        f"REGRESSION: tuning hurt accuracy by {drop:.4f} "
        f"(baseline={baseline:.4f}, tuned={tuned:.4f}). The MedianPruner or "
        f"per-fold CV reporting in tune.py likely regressed."
    )


# ── Test 4: heavy-tail code paths actually executed ──────────────────────

def test_stratified_split_and_outlier_cleanup_fired(skewed_demo_pipeline_result):
    """The unit tests never exercise the heavy-tail code paths because the
    toy target is Gaussian (skew ~ 0.5 < 1.0). This test confirms the REAL
    pipeline on a skewed target actually triggered:
      - the stratified regression split (split.py)
      - the target outlier cleanup (clean.py)

    If either message is missing, the gates may have regressed (e.g. someone
    changed the skew threshold from 1.0 to 100, or removed the print).

    Fixture reference is what forces the pipeline to run (pytest runs the
    fixture on first param reference); the actual stdout is read from the
    module-level REAL_DATA_RUN_STDOUT because split.py prints to stdout,
    not a logger -- pipeline_run_*.log files are only written by the CLI's
    logging handler, inactive when we call PipelineRunner directly here.
    """
    _ = skewed_demo_pipeline_result  # force fixture to run
    stdout = REAL_DATA_RUN_STDOUT
    assert "Stratified split on target" in stdout, (
        "REGRESSION: split.py did not stratify on the skewed target. "
        "Check that the skew gate (split.py) still triggers for skew > 1.0. "
        f"Captured stdout tail:\n{stdout[-800:]}"
    )
    assert "dropped" in stdout.lower(), (
        "REGRESSION: clean.py outlier cleanup did not fire. Check that the "
        "skew gate in clean.py still triggers for skew > 1.0 and prints a "
        "dropped-rows message. "
        f"Captured stdout tail:\n{stdout[-800:]}"
    )
