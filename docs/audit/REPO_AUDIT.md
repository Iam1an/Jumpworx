# Repository Audit Report: Jumpworx

**Date:** 2026-02-05
**Auditor:** Claude Code (autonomous)
**Repo:** `/Users/davidpetersen/jumpworx-assessment`
**Commit:** `e45e66e` (HEAD of main at audit start)

---

## 1. Overview

Jumpworx is a computer vision + ML pipeline for analyzing trampoline-based aerial ski trick videos. It extracts pose keypoints (MediaPipe), segments jump phases, classifies tricks (sklearn), compares amateur vs. pro performance via DTW alignment, and optionally generates LLM-powered coaching feedback.

**Tech stack:** Python 3.10+, NumPy, scikit-learn, joblib, MediaPipe (optional), OpenCV (optional), OpenAI SDK (optional).

---

## 2. Repository Structure

| Directory | Purpose | File count |
|-----------|---------|------------|
| `jwcore/` | Core library: pure math, feature extraction, classification, coaching | 14 .py |
| `scripts/` | CLI entrypoints: demo, runner, keypoint extraction, training, viz | 7 .py |
| `tests/` | Pytest test suite | 7 .py (incl. conftest) |
| `experiments/` | Exploratory / legacy scripts | 13 .py |
| `tools/` | Utility scripts for model eval, inspection, reporting | 12 .py |
| `models/` | Trained model artifacts (.joblib, .pkl) | 2 files |

---

## 3. Entrypoints and Public APIs

### Core Library (`jwcore/`)

| Module | Key Exports | External Deps | Side Effects |
|--------|-------------|---------------|--------------|
| `pose_utils.py` | `FEATURE_KEYS`, `features_from_posetrack()`, `compute_adaptive_pitch()`, `extract_trick_features()`, `rotation_sign()`, `features_to_vector()` | numpy | None (pure) |
| `phase_segmentation.py` | `Phases`, `PhaseResult`, `segment_phases_with_airtime_v2()`, `phase_result_from_pose()` | numpy | None (pure) |
| `trick_classifier.py` | `TrickClassifier` (predict, predict_with_proba, predict_proba_vector) | joblib, sklearn, numpy | File I/O (model load) |
| `compare_metrics.py` | `compare_metrics_from_xyz()` | numpy, jwcore.joints, jwcore.coaching_thresholds | None (pure) |
| `coach.py` | `coach()` | numpy, jwcore.coaching_thresholds | Optional LLM I/O |
| `coaching_thresholds.py` | `TAU`, `UNITS`, `PHASE_TAG`, `material_delta()`, `select_top_metrics()` | None | None (pure constants) |
| `normalize.py` | `normalize_prerot()`, `height_series_up()` | numpy | None (pure) |
| `posetrack_io.py` | `load_posetrack()`, `resolve_posetrack_path()` | numpy, json | File I/O (NPZ read) |
| `io_cache.py` | `load_pose()` | cv2, mediapipe, numpy | File I/O (video capture, cache write) |
| `joints.py` | Landmark index constants, `JOINT_NAME_BY_INDEX` | None | None (pure constants) |
| `feature_deltas.py` | `stance_hand_deltas()` | None | None (pure) |
| `pose_extract.py` | `process_file()` | jwcore.pose_utils, json | File I/O (read NPZ, write JSON) |
| `pro_utils.py` | `compute_pro_features_from_xyz()`, `get_or_build_pro_features()` | numpy, json | File I/O (cache JSON) |
| `pro_index.py` | `load_pro_index()`, `pick_pro_for_label()` | json, subprocess | File I/O, subprocess |

### Script Entrypoints (`scripts/`)

| Script | Main Function | Purpose |
|--------|--------------|---------|
| `demo_cli.py` | `main(argv)` | User-facing CLI wrapper; calls runner via subprocess |
| `runner.py` | `main()` | Core orchestrator: extract -> classify -> compare -> coach -> viz |
| `extract_keypoints.py` | `main(argv)` | MediaPipe pose extraction -> canonical NPZ |
| `train_trick_model.py` | `main()` | Train RF/LR trick classifiers |
| `viz_compare_side_by_side.py` | `main()` | Render side-by-side comparison video |
| `generate_coaching.py` | `main(argv)` | LLM coaching generation |
| `measure_compare.py` | `main()` | Batch comparison utility |

---

## 4. Safety Harness Strategy

### Chosen Strategy: **A (pytest-based tests)**

### Why Strategy B (golden harness) was rejected:
- The pipeline's heavy I/O path (video capture via MediaPipe/OpenCV) is **non-deterministic** across runs and platforms. MediaPipe inference produces slightly different landmark coordinates depending on hardware, library version, and even run order. Capturing golden outputs would produce flaky comparisons.
- The optional LLM coaching path is inherently non-deterministic.
- Pro reference file selection depends on available video files (not present in the repo).
- **Strategy A is strongly preferred** because:
  - 6 existing test files already exist with synthetic data fixtures.
  - The core computation modules (`pose_utils`, `phase_segmentation`, `compare_metrics`, `coach`, `coaching_thresholds`, `normalize`) are **pure math on numpy arrays** with deterministic outputs for given inputs.
  - The classifier has a testable train/predict cycle using synthetic features.
  - I/O contracts (NPZ schema, feature JSON schema) are well-defined and testable.

### Critical Blocker Found:
**All 6 existing tests fail at collection time** due to `jwcore/__init__.py` line 2:
```python
from .io_cache import load_pose
```
This triggers `import cv2` at package import time. Since `cv2` (OpenCV) is a heavy optional dependency not declared in `pyproject.toml` core deps, any test importing from `jwcore` fails without it. The harness fix will make this import lazy/conditional.

Additionally, `test_feedback_generator.py` imports `jwcore.feedback_generator` which does not exist in `jwcore/` (it exists in `experiments/`). This test cannot pass as-is.

### Behaviors/entrypoints the harness will lock down:

1. **Feature extraction** (`jwcore/pose_utils.py`): `features_from_posetrack()`, `compute_adaptive_pitch()`, `rotation_sign()`, `features_to_vector()`, `FEATURE_KEYS` contract
2. **Phase segmentation** (`jwcore/phase_segmentation.py`): `segment_phases_with_airtime_v2()`, `phase_result_from_pose()`
3. **Trick classification** (`jwcore/trick_classifier.py`): `TrickClassifier.predict()`, `.predict_with_proba()`, heuristic fallback
4. **Comparison metrics** (`jwcore/compare_metrics.py`): `compare_metrics_from_xyz()` — DTW alignment, scalar metrics, phase scores
5. **Coaching** (`jwcore/coach.py`): `coach()` — rule-based path (no LLM)
6. **Coaching thresholds** (`jwcore/coaching_thresholds.py`): `material_delta()`, `select_top_metrics()`
7. **Normalization** (`jwcore/normalize.py`): `normalize_prerot()`, `height_series_up()`
8. **Pose I/O** (`jwcore/posetrack_io.py`): `load_posetrack()`, layout handling
9. **Import chain**: `jwcore` package imports without requiring cv2/mediapipe

### Minimal production code change required:
- `jwcore/__init__.py`: make `io_cache` import lazy so the package can be imported without cv2. This does **not** change the public API — `load_pose` remains importable from `jwcore` but only fails if actually called without cv2 installed.

---

## 5. Existing Test Inventory

| Test File | Functions | Status (pre-harness) | Covers |
|-----------|-----------|---------------------|--------|
| `test_core_modules.py` | `test_io_cache`, `test_normalize` | FAIL (cv2 import) | io_cache.load_pose, normalize |
| `test_feedback_generator.py` | `test_compute_and_summarize` | FAIL (missing module) | experiments/feedback_generator (NOT in jwcore) |
| `test_phase.py` | `test_basic_airtime` | FAIL (cv2 import) | phase_segmentation |
| `test_pose_extract.py` | `test_process_file` | FAIL (cv2 import) | pose_extract.process_file |
| `test_pose_utils.py` | `test_extract_trick_features_includes_rotation_sign`, `test_rotation_sign_direction_front_vs_back` | FAIL (cv2 import) | pose_utils |
| `test_trick_classifier.py` | `test_feature_key_source_of_truth`, `test_train_save_reload_and_predict`, `test_predict_with_confidence_object_shape`, `test_heuristic_fallback_when_no_sklearn` | FAIL (cv2 import) | trick_classifier |

**Total: 0/10 tests pass before harness fix.**

### Post-Harness Test Status

After the `jwcore/__init__.py` lazy import fix, the pre-existing tests still have additional breakage:

| Test File | Post-Harness Status | Root Cause |
|-----------|-------------------|------------|
| `test_core_modules.py` | FAIL | Directly imports `jwcore.io_cache` (bypasses lazy wrapper) |
| `test_feedback_generator.py` | FAIL | Imports `jwcore.feedback_generator` which doesn't exist |
| `test_phase.py` | FAIL | Treats `segment_phases_with_airtime_v2` return as Phases, but it returns `(Phases, debug)` tuple |
| `test_pose_extract.py` | FAIL | Calls `process_file()` with `fps_override=` and `strict=` kwargs that don't exist in the current signature |
| `test_pose_utils.py` | FAIL | Imports `extract_trick_features`, `rotation_sign`, `features_to_vector` — none exist in `pose_utils.py` |
| `test_trick_classifier.py` | FAIL | Uses `model_base_path=` kwarg; actual param is `model_path_or_base` |

**All 6 pre-existing test files contain API mismatches against the current code.** The safety harness (`tests/test_safety_harness.py`) provides 37 new tests that correctly target the current API.

### Phase Segmentation Bug (Pre-Existing)

The `_hysteresis_mask()` function in `phase_segmentation.py:94-106` has a threshold logic issue:
- It sets `low = air_y - hysteresis` which is *below* the actual air y-values
- The state machine enters "active" when `y < high` (always true) but immediately exits when `y >= low` (also always true)
- **Result:** synthetic data with step-function air/ground transitions never produces a valid airtime detection
- This likely only works with real pose data where continuous y-values and smoothing create gradual transitions
- The existing `test_phase.py` test uses the same synthetic pattern and would also fail if it could be collected

### Safety Harness Summary

| File | Tests | Status |
|------|-------|--------|
| `tests/test_safety_harness.py` | 37 | **ALL PASS** |

**Command:** `python -m pytest tests/test_safety_harness.py -v`

---

## 6. Dependency Analysis

### Declared (`pyproject.toml`):
- **Core:** numpy >= 1.24
- **Optional [ml]:** scikit-learn >= 1.4, joblib >= 1.3
- **Optional [viz]:** matplotlib >= 3.7, imageio >= 2.31
- **Optional [dev]:** pytest >= 7.4, pytest-cov >= 4.1

### Undeclared but used at import time:
- `cv2` (OpenCV) — **hard import** in `jwcore/io_cache.py`, propagated via `jwcore/__init__.py`
- `mediapipe` — guarded import in `io_cache.py` (deferred, OK)

### Undeclared but used at runtime (optional):
- `openai` — for LLM coaching in `scripts/generate_coaching.py`

### CI/CD:
- **None.** No `.github/workflows/`, no `Makefile`, no `tox.ini`, no `pytest.ini`.

---

## 7. Security & Risk Assessment

| Area | Risk | Evidence | Notes |
|------|------|----------|-------|
| Model deserialization | Medium | `trick_classifier.py:102` uses `joblib.load()` | No integrity verification on .joblib/.pkl files |
| LLM prompt injection | Low | `coach.py:185-271` includes user feature data in prompt | Display-only output; no action execution |
| Subprocess calls | Low | `pro_index.py` uses `subprocess.check_call()` | No `shell=True`; arg-list form (safe) |
| File path handling | Low | `pose_extract.py` infers trick label from filename | Could mislabel sign if filename convention not followed |
| NaN interpolation | Low | `extract_keypoints.py` interpolates gaps <= 4 frames | Could mask data corruption |

---

## 8. Quickstart

### Prerequisites (from repo root)

```bash
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -e ".[dev,ml]"
```

- **`dev`** — installs pytest and pytest-cov (test runner tooling).
- **`ml`** — installs scikit-learn and joblib. Required by the safety harness because `TrickClassifier` tests load a joblib model from `models/`.

No network access, databases, or external services are needed.

### Recommended: run the safety harness (37 tests)

```bash
python -m pytest tests/test_safety_harness.py -v
```

### Run with coverage

```bash
python -m pytest tests/test_safety_harness.py -v --cov=jwcore --cov-report=term-missing
```

### Run full test suite (legacy tests currently failing; expect failures)

```bash
python -m pytest tests/ -v
```

> The six pre-existing test files under `tests/` have API mismatches against the
> current production code (wrong kwargs, missing symbols, incorrect return-type
> assumptions). Only `tests/test_safety_harness.py` targets the current API.

---

## 9. Refactor Plan

See section at bottom of this document.

---

## 10. Open Questions

1. **`test_feedback_generator.py`** imports `jwcore.feedback_generator` which does not exist. The module is in `experiments/feedback_generator.py`. Should this test be moved/updated, or should `feedback_generator.py` be promoted to `jwcore/`?
2. **`test_core_modules.py::test_io_cache`** requires OpenCV + MediaPipe + a video file. Should this remain as an integration test with a `pytest.mark.slow` skip, or be dropped?
3. **Model file integrity:** The `.joblib` and `.pkl` files in `models/` have no checksum verification. Is this acceptable for the deployment context?

---

## 11. Refactor Plan

### PR 1: Fix import chain — DONE ✓
- **Commit:** `963ba17` — `jwcore/__init__.py` lazy import for `io_cache`
- **Status:** Merged to main. Safety harness (37 tests) validates the fix.

### PR 3: Triage legacy tests
- **Scope:** Add `pytest.skip` (broken imports) or `@pytest.mark.xfail` (API mismatches) to the 6 pre-existing test files so `pytest -q` produces a meaningful report instead of a wall of import errors.
- **Files touched:** `tests/test_feedback_generator.py`, `tests/test_core_modules.py`, `tests/test_phase.py`, `tests/test_pose_extract.py`, `tests/test_pose_utils.py`, `tests/test_trick_classifier.py`
- **Acceptance criteria:**
  - `python -m pytest tests/test_safety_harness.py -v` → 37 passed (unchanged)
  - `python -m pytest tests/ -q` → 0 errors at collection time; legacy tests show as skipped/xfail
- **Rollback:** `git revert <commit>` — no production code touched.

### PR 5: Add CI pipeline
- **Scope:** `.github/workflows/test.yml` — run safety harness on push to main and on PRs.
- **Matrix:** Python 3.10, 3.11, 3.12 on ubuntu-latest.
- **Acceptance criteria:**
  - `python -m pytest tests/test_safety_harness.py -v` → 37 passed (local)
  - `.github/workflows/test.yml` is valid YAML (`python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"`)
- **Rollback:** `git revert <commit>` — additive only, no production impact.

### PR 4: Declare optional dependencies properly
- **Scope:** `pyproject.toml` — add `[project.optional-dependencies].video` group with `opencv-python`, `mediapipe`.
- **Acceptance criteria:**
  - `pip install -e ".[video]"` succeeds in a fresh venv
  - `python -m pytest tests/test_safety_harness.py -v` → 37 passed (no regressions)
- **Rollback:** `git revert <commit>` — metadata-only change.

### PR 6: Guard io_cache imports properly
- **Scope:** `jwcore/io_cache.py` — make `cv2` import conditional (try/except like `mediapipe`). `load_pose()` raises a clear `ImportError` if called without cv2.
- **Acceptance criteria:**
  - `python -c "import jwcore.io_cache"` succeeds without cv2 installed
  - `python -m pytest tests/test_safety_harness.py -v` → 37 passed
- **Rollback:** `git revert <commit>` — single-file change.

### PR 2: Expand safety harness tests
- **Scope:** Add tests to `tests/test_safety_harness.py` (or new files) for edge cases, coverage gaps identified by `--cov-report=term-missing`.
- **Acceptance criteria:**
  - `python -m pytest tests/test_safety_harness.py -v --cov=jwcore --cov-report=term-missing` → increased line coverage
  - No new external dependencies
- **Rollback:** `git revert <commit>` — test-only change.

### PR 7: Consolidate posetrack_io
- **Scope:** `jwcore/posetrack_io.py` and `jwcore/pose_utils.py` both contain `load_posetrack_npz` — deduplicate by removing one copy and updating all callers.
- **Acceptance criteria:**
  - `grep -r load_posetrack_npz jwcore/ scripts/` shows exactly one definition
  - `python -m pytest tests/test_safety_harness.py -v` → 37 passed
- **Rollback:** `git revert <commit>` — but requires caller audit first, so verify callers before merging.
