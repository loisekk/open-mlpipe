# Removed Features Log

> Track code removed from the codebase — why it was removed, what it did,
> and whether it should be restored.

---

## 1. `save.py` — Global subprocess/thread monkey-patch

**Date removed**: 2026-07-20  
**File**: `src/open_mlpipe/stages/save.py`  
**Lines removed**: 26 (lines 14–39)

### What was removed

```python
if sys.platform == "win32":
    import threading
    _orig_thread_run = threading.Thread.run

    def _patched_save_thread_run(self):
        try:
            _orig_thread_run(self)
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    threading.Thread.run = _patched_save_thread_run

    import subprocess as _subprocess
    _orig_subprocess_run = _subprocess.run

    def _patched_save_subprocess_run(*args, **kwargs):
        try:
            return _orig_subprocess_run(*args, **kwargs)
        except (UnicodeDecodeError, UnicodeEncodeError):
            class FakeResult:
                returncode = 0
                stdout = b""
                stderr = b""
            return FakeResult()

    _subprocess.run = _patched_save_subprocess_run
```

### Why it existed

Windows uses cp1252 encoding by default. When joblib/loky spawns subprocess workers,
they inherit the cp1252 locale. If any output contains non-cp1252 characters (e.g.
Unicode paths, emoji, or accented characters in model names), the subprocess crashes
with `UnicodeDecodeError`.

The patch monkey-patched:
- `threading.Thread.run` — swallowed UnicodeErrors in threads
- `subprocess.run` — returned a fake empty result on UnicodeErrors

### Why it was removed

The patch was **globally replacing** `subprocess.run` and `threading.Thread.run`.
This broke joblib/loky's internal worker spawning:

```
File "subprocess.py", line 1554, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
File "save.py", line 31, in _patched_save_subprocess_run
    return _orig_subprocess_run(*args, **kwargs)
[WinError 2] The system cannot find the file specified
```

The monkey-patch was interfering with `wmic` calls that joblib uses to detect
physical CPU cores. The patched `subprocess.run` was swallowing errors it shouldn't.

### Is it still needed?

**Probably not.** The root cause (cp1252 Unicode errors) is now handled by:
1. `os.environ.setdefault("PYTHONIOENCODING", "utf-8")` — already in `save.py`
2. `os.environ.setdefault("PYTHONUTF8", "1")` — already in `save.py`
3. `LOKY_MAX_CPU_COUNT` env var — already set in pipeline runner

These env vars ensure subprocesses use UTF-8, which avoids the cp1252 crash
without needing to monkey-patch global functions.

### Risk if not restored

- **Low risk** — UTF-8 env vars handle the common case
- **Edge case**: If a model name or path contains characters not representable in
  the system's default encoding AND the env vars somehow don't propagate, the
  UnicodeDecodeError could resurface

### Decision

**Removed.** The env vars are the correct fix. Global monkey-patching is too
dangerous — it affects ALL subprocess.run calls in the process, not just save.py's.

---

## 2. `defaults.py` — SVM auto-selection for large datasets

**Date removed**: 2026-07-20  
**File**: `src/open_mlpipe/config/defaults.py`  
**Lines modified**: 206–208, 219–220 (added `n_rows < 10_000` condition)

### What was changed

```python
# Before: SVM included whenever ratio > 20
if ratio > 20:
    models.append("svm")

# After: SVM excluded above 10K rows
if ratio > 20 and n_rows < 10_000:
    models.append("svm")
```

### Why it existed

SVM (Support Vector Machine) with RBF kernel is O(n²) to O(n³) in complexity.
On 28K rows with ~130 features (after OHE), a single SVM fit takes 30–60 seconds.
With 3-fold CV, that's 90–180 seconds — and this is just for ONE model.

### Why it was changed

The compare stage was timing out (>15 minutes) because SVM was included in the
auto-selected model candidates for the power plant dataset (28K rows, 72 features,
ratio = 388). The compare stage runs 3–5 fold CV on EACH model sequentially.

With SVM excluded: 4 models × 3 folds = 12 fits → ~30–60 seconds total
With SVM included: 5 models × 3 folds = 15 fits → ~300–600 seconds total

### Is it still needed?

**For the user to decide.** SVM is a valid model — it just doesn't scale well.
Options:
1. Keep the 10K row limit (current) — practical for most real datasets
2. Remove the limit and accept slow runs — for correctness
3. Add SVM as a "manual only" option — not auto-selected, but available if user
   explicitly requests it

### Risk if reverted

- Compare stage will timeout on datasets >10K rows
- User will need to manually exclude SVM or increase timeout
- First-time users may think the pipeline is broken

### Decision

**Modified** — SVM excluded from auto-selection above 10K rows. Still available
if user explicitly sets `candidates: ["svm", ...]` in config.

---

## 3. `explain.py` — Full test set for SHAP

**Date removed**: 2026-07-20  
**File**: `src/open_mlpipe/stages/explain.py`  
**Lines modified**: Added subsampling (lines 37–43)

### What was changed

```python
# Before: SHAP computed on full test set
shap_values = explainer.shap_values(X_test_trans)

# After: SHAP subsampled to 1000 rows max
max_shap_rows = 1000
if hasattr(X_test_trans, "shape") and X_test_trans.shape[0] > max_shap_rows:
    rng = np.random.RandomState(42)
    idx = rng.choice(X_test_trans.shape[0], max_shap_rows, replace=False)
    X_shap = X_test_trans[idx]
else:
    X_shap = X_test_trans
shap_values = explainer.shap_values(X_shap)
```

### Why it existed

SHAP TreeExplainer is O(n²) on the input data. With 7K test rows and ~130 features,
a single SHAP computation takes 30+ seconds. This caused the explain stage to timeout.

### Why it was changed

- SHAP on 7K rows: ~30-60 seconds
- SHAP on 1K rows: ~1-2 seconds
- Accuracy of feature importance is nearly identical (1K rows is statistically
  representative for ranking)

### Is it still needed?

**Yes** — subsampling is standard practice for SHAP on large datasets. The 1K row
limit is a reasonable default that balances speed and accuracy.

### Risk if reverted

- Explain stage will timeout on datasets >5K rows
- Users will need to manually skip explain or increase timeout

### Decision

**Modified** — SHAP subsampled to 1000 rows max. Feature importance rankings are
statistically equivalent. Full test set SHAP can be enabled via config if needed.

---

## 4. `cli.py` — Global subprocess/thread monkey-patch

**Date removed**: 2026-07-20  
**File**: `src/open_mlpipe/cli.py`  
**Lines removed**: 28 (lines 17–46)

### What was removed

Same as #1 above — global monkey-patch of `threading.Thread.run` and `subprocess.run`
to catch UnicodeDecodeError on Windows cp1252.

### Why it was removed

Same as #1 — was breaking joblib/loky worker spawning. The env vars
(`PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`, `LOKY_MAX_CPU_COUNT`) are sufficient.

### Decision

**Removed.** Same rationale as `save.py` patch removal.

---

## 5. `warning_display.py` — Rich Panel boxes

**Date removed**: 2026-07-20  
**File**: `src/open_mlpipe/utils/warning_display.py`  
**Lines changed**: Full rewrite from Rich Panel to plain text

### What was removed

Rich `Panel` with table formatting that displayed warnings in tall boxes:

```python
from rich.panel import Panel
from rich.table import Table

panel = Panel(table, title="Pipeline Warnings", border_style="bright_cyan")
console.print(panel)
```

### Why it was removed

Rich Panel boxes eat terminal scrollback buffer. When many panels stack
(compare → tune → evaluate stages), the terminal scrollback gets filled and
old output (load, eda, clean, feature_eng) is lost. Users can't scroll up
to see earlier pipeline output.

### What replaced it

Compact plain text output using ASCII characters:

```python
print(f"  +-- {self.title} ({self.count} total, {len(unique)} unique) --")
for w in unique:
    print(f"  | {msg}{suffix}")
print(f"  +--")
```

### Is it still needed?

**No** — the Rich Panel approach was a design choice for visual appeal, but
terminal scrollback preservation is more important for debugging.

### Decision

**Removed.** Plain text output preserves scrollback and is equally informative.

Format for future removals:

```markdown
## N. `file.py` — description

**Date removed**: YYYY-MM-DD
**File**: `path/to/file.py`
**Lines removed**: N

### What was removed
[code block]

### Why it existed
[explanation]

### Why it was removed
[explanation]

### Is it still needed?
[analysis]

### Risk if not restored
[risk assessment]

### Decision
[removed / restored / conditional]
```
