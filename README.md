# 🏂 Jumpworx: AI Coaching for Aerial Ski Tricks

[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

Jumpworx is a computer vision and machine learning pipeline that analyzes trampoline-based ski training videos to provide **automated trick recognition**, **performance comparison**, and **AI-powered coaching insights**.

The system processes raw video through pose estimation, phase segmentation, feature extraction, classification, and side-by-side visualization — helping athletes refine their air awareness and technique.

---

## 🚀 Overview

**Jumpworx** takes a video of an athlete performing a trick (e.g. backflip) and compares it to professional reference footage using pose-based analysis and statistical feature matching.
It uses **Mediapipe** for keypoint extraction, **scikit-learn** models for classification, and optional **LLM coaching** (via OpenAI's GPT-4o-mini) to describe areas of improvement.

---

## 🧩 Core Pipeline

| Stage | Module | Description |
|--------|---------|-------------|
| 🎥 **Pose Extraction** | `scripts/extract_keypoints.py` | Uses Mediapipe to extract 3D body landmarks from a video and saves them as `.npz` pose tracks in `/cache`. |
| 🧠 **Feature Engineering** | `jwcore/pose_extract.py` | Computes biomechanical and kinematic features from the pose data (joint velocities, rotations, airtime metrics, etc.). |
| 🧩 **Phase Segmentation** | `jwcore/phase_segmentation.py` | Detects takeoff, airtime, and landing phases of a jump. |
| 🏷️ **Trick Classification** | `jwcore/trick_classifier.py`, `scripts/train_trick_model.py` | Uses Random Forest and Logistic Regression models to predict trick type from feature vectors. |
| 🎮 **Comparison & Visualization** | `scripts/viz_compare_side_by_side.py` | Renders synchronized side-by-side playback of an amateur vs. pro trick, aligned by takeoff, landing, or apex. |
| 🧑‍🏫 **Coaching (Optional)** | `jwcore/coach.py`, `scripts/demo_cli.py` | Generates feedback text comparing motion quality and timing. Integrates OpenAI GPT-4o-mini for natural-language advice. |

---

## 📂 Directory Structure

```
Jumpworx/
│
├── scripts/                        # CLI utilities & entry points
│   ├── demo_cli.py                 # Main demo runner (entry point)
│   ├── runner.py                   # Core orchestrator for the pipeline
│   ├── extract_keypoints.py        # Pose extraction via Mediapipe
│   ├── viz_compare_side_by_side.py # Side-by-side visualization
│   └── train_trick_model.py        # Train classification models
│
├── jwcore/                         # Core reusable logic
│   ├── phase_segmentation.py       # Phase segmentation
│   ├── trick_classifier.py         # Trick classification logic
│   ├── posetrack_io.py             # Input/output utilities for pose tracks
│   ├── compare_metrics.py          # Pose & motion similarity metrics
│   ├── coach.py                    # Feedback and LLM coaching interface
│   └── ...
│
├── models/                         # Trained model artifacts (.joblib, .pkl)
├── tests/                          # Pytest test suite
├── cache/                          # Generated NPZ pose files and features
├── features/                       # JSON feature summaries
├── videos/                         # Raw training and pro video clips
└── viz/                            # Rendered comparison videos and metrics
```

---

## ⚙️ Installation

**Requirements:**
- Python 3.10–3.12 (Mediapipe's `mp.solutions` API is not available on Python 3.13+)
- NumPy, scikit-learn, joblib (core)
- OpenCV, Mediapipe (for video pose extraction)
- (Optional) OpenAI Python SDK for LLM coaching

**Setup:**

```bash
git clone https://github.com/Iam1an/Jumpworx.git
cd Jumpworx
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,ml,video]"
```

The optional extras are:
- **`dev`** — pytest and pytest-cov (test runner tooling)
- **`ml`** — scikit-learn and joblib (classification models)
- **`video`** — opencv-python and mediapipe (pose extraction from video)

If you only need to run the safety harness tests (no video processing):

```bash
pip install -e ".[dev,ml]"
```

If you plan to use OpenAI for coaching:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 🧪 Usage

### 1. Extract Keypoints

```bash
python3 scripts/extract_keypoints.py --video videos/training/TRICK45_BACKFLIP.mov
```

Creates:

```
cache/TRICK45_BACKFLIP.posetrack.npz
```

These keypoints are used for feature generation, classification, comparison, and visualization.

### 2. Run Full Demo (with AI coaching)

```bash
python3 -m scripts.demo_cli videos/training/TRICK45_BACKFLIP.mov \
  --runner scripts.runner \
  --extra-args "--pro_videos_dir videos/pro --strategy closest --align dtw --align_feature ankle_y --llm_provider openai --llm_model gpt-4o-mini"
```

This:

- extracts or reuses cached poses
- finds the most similar professional example
- runs feature comparison, classification, and phase analysis
- generates side-by-side visualization in `/viz`
- optionally produces AI-written feedback

### 3. Run Tests

```bash
python3 -m pytest tests/test_safety_harness.py -v    # safety harness (28 tests)
python3 -m pytest tests/ -q                          # full suite
```

---

## 🧠 Key Learning Outcomes

- Pose quality matters: Mediapipe's keypoints are sensitive to camera angle and lighting. Multi-camera capture or better lighting drastically improves pose consistency.
- Granular analysis beats averages: Early versions only averaged features, hiding frame-level motion nuance. Frame-by-frame joint metrics yielded much richer coaching signals.
- Data completeness: Missing landmarks caused classifier noise; careful filtering and interpolation improved stability.
- Modularity wins: Each stage (keypoints, phases, classification, viz) is its own module, making it easy to debug or upgrade independently.

## 🎓 Techniques & Models

- Pose estimation: Mediapipe BlazePose full-body
- Feature extraction: Temporal and biomechanical metrics (joint angles, airtime, body pitch)
- Phase detection: Takeoff, airtime, landing segmentation with airtime validation
- Classification: Random Forest and Logistic Regression (>95% accuracy)
- Similarity scoring: Dynamic Time Warping, pose vector distances
- Visualization: OpenCV compositing and skeletal overlays
- Coaching LLM: GPT-4o-mini summarizing key performance gaps

## 📈 Example Outputs

- **Visualization:** A synchronized, labeled comparison video showing the amateur and professional performing the same trick side-by-side, with overlayed metrics and phase markers.
- **Metrics:** CSV and JSON summaries in `/viz` including:
  - Airtime difference
  - Takeoff/landing timing deltas
  - Overall trick similarity score
- **Coaching Output (example):**

  > "Your takeoff is smooth, but your mid-air rotation speed is slightly lower than the reference. Focus on earlier hip extension to match pro airtime."

## 🧹 Possible Next Steps

- Integrate multi-camera calibration for more reliable 3D landmarks.
- Expand classifier to detect off-axis spins and grabs.
- Improve automatic phase labeling and LLM contextual reasoning.
- Extend visualization to 3D replay using Blender or Three.js.

## 🪪 License

This project is open-source under the MIT License. You are free to use, modify, and distribute it with attribution.
