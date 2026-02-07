# ML Pipeline: Trick Classification

## Overview

The classification pipeline takes a `.posetrack.npz` file (3D pose landmarks over time) and predicts the trick type (e.g. `backflip`, `frontflip`).

```
.posetrack.npz  ->  features_from_posetrack()  ->  feature vector  ->  TrickClassifier  ->  label + proba
```

---

## 1. Inputs

**NPZ schema (canonical):** keys `["P", "V", "meta_json"]`

| Key         | Shape / Type            | Description |
|-------------|------------------------|-------------|
| `P`         | `(T, 33, 3)` float32   | Mediapipe BlazePose landmarks in pixel coordinates (x, y) and pixel-scaled z |
| `V`         | `(T, 33)` float32      | Per-landmark visibility scores in `[0, 1]` |
| `meta_json` | JSON string            | Must contain `{"fps": float, "image_h": int, "image_w": int}` |

Legacy NPZ files with keys `["kps_xyz", "visibility", "fps"]` are also accepted by `load_posetrack_npz()`.

---

## 2. Feature Contract

Defined in `jwcore/pose_utils.py` as `FEATURE_KEYS`:

```python
FEATURE_KEYS = [
    "pitch_total_rad",              # signed: backflip +, frontflip -
    "pitch_speed_abs_mean_rad_s",   # mean |d-theta/dt| over chosen window
    "height_clearance_px_p95",      # 95th percentile feet_y - baseline
]
```

`features_from_posetrack(P, V, meta)` returns a `dict[str, float]` with these keys. This function is the **single source of truth** for both training and inference.

**Sign convention:** `pitch_total_rad > 0` = backflip; `< 0` = frontflip.

---

## 3. Training

**Canonical script:** `scripts/train_zonly.py`

```bash
# From glob:
python scripts/train_zonly.py --glob "cache/*.posetrack.npz"

# From specific files:
python scripts/train_zonly.py \
    --npz cache/TRICK06_BACKFLIP.posetrack.npz \
    --npz cache/TRICK16_FRONTFLIP.posetrack.npz
```

The script:
1. Loads each NPZ via `load_posetrack_npz()`
2. Extracts features via `features_from_posetrack()`
3. Infers labels from the filename stem (`backflip`, `frontflip`)
4. Trains `StandardScaler + LogisticRegression` (C=3.0, balanced weights)
5. Saves a dict bundle to `models/trick_model_zonly.pkl`
6. Saves a classification report to `models/trick_model_zonly_report.json`

**Deprecated:** `scripts/train_trick_model.py` references feature names and TrickClassifier methods that no longer exist. Do not use it.

---

## 4. Model Bundle Format

The saved `.pkl` / `.joblib` file is a joblib-serialized Python object in one of two formats:

### Format A: Dict bundle (preferred, produced by `train_zonly.py`)

```python
{
    "model":         sklearn.pipeline.Pipeline,   # StandardScaler + LogisticRegression
    "feature_names": ["pitch_total_rad", ...],    # ordered list matching FEATURE_KEYS
    "class_names":   ["backflip", "frontflip"],    # from pipe.classes_
}
```

### Format B: Bare sklearn Pipeline

```python
joblib.dump(pipeline, "models/jumpworx_model.joblib")
```

When loading a bare pipeline, `TrickClassifier` uses `FEATURE_KEYS` from `jwcore.pose_utils` as the default feature ordering.

---

## 5. Inference

### Programmatic

```python
from jwcore.trick_classifier import TrickClassifier

clf = TrickClassifier("models/trick_model_zonly.pkl")
# or: clf = TrickClassifier()  # loads default models/jumpworx_model.joblib

label = clf.predict(features_dict)                # -> "backflip"
label, proba = clf.predict_with_proba(features_dict)  # -> ("backflip", 0.97)
proba_dict = clf.predict_proba_vector(features_dict)  # -> {"backflip": 0.97, "frontflip": 0.03}
```

### CLI

```bash
python -m jwcore.trick_classifier \
    --model models/trick_model_zonly.pkl \
    --glob "cache/*.posetrack.npz" \
    --summary
```

---

## 6. Validation

```bash
# Run integration tests (train -> save -> load -> predict round-trip):
python -m pytest tests/test_train_to_inference.py -v

# Run full test suite:
python -m pytest tests/ -q

# Classify cached NPZ files with the trained model:
python -m jwcore.trick_classifier \
    --model models/trick_model_zonly.pkl \
    --glob "cache/*.posetrack.npz" \
    --summary
```

---

## 7. File Map

| File | Role |
|------|------|
| `jwcore/pose_utils.py` | `FEATURE_KEYS`, `features_from_posetrack()` |
| `jwcore/posetrack_io.py` | `load_posetrack_npz()` (canonical NPZ loader) |
| `jwcore/trick_classifier.py` | `TrickClassifier` (inference adapter) |
| `scripts/train_zonly.py` | Canonical training script |
| `scripts/train_trick_model.py` | **Deprecated** - do not use |
| `models/trick_model_zonly.pkl` | Dict bundle (preferred model) |
| `models/jumpworx_model.joblib` | Bare pipeline (legacy model) |
| `tests/test_train_to_inference.py` | Integration tests for train-to-inference |
