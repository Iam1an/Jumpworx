# 🏂 Jumpworx: AI Coaching for Aerial Ski Tricks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

Jumpworx is a computer vision and machine learning pipeline that analyzes trampoline-based ski training videos to provide **automated trick recognition**, **performance comparison**, and **AI-powered coaching insights**.

The system processes raw video through pose estimation, phase segmentation, feature extraction, classification, and side-by-side visualization — helping athletes refine their air awareness and technique.

---

## 🚀 Overview

**Jumpworx** takes a video of an athlete performing a trick (e.g. backflip) and compares it to professional reference footage using pose-based analysis and statistical feature matching.  
It uses **Mediapipe** for keypoint extraction, **scikit-learn** models for classification, and optional **LLM coaching** (via OpenAI’s GPT-4o-mini) to describe areas of improvement.

---

## 🧩 Core Pipeline

| Stage | Module | Description |
|--------|---------|-------------|
| 🎥 **Pose Extraction** | `scripts/extract_keypoints.py` | Uses Mediapipe to extract 3D body landmarks from a video and saves them as `.npz` pose tracks in `/cache`. |
| 🧠 **Feature Engineering** | `jwcore/pose_extract.py`, `scripts/build_features_for_csv.py` | Computes biomechanical and kinematic features from the pose data (joint velocities, rotations, airtime metrics, etc.). |
| 🧩 **Phase Segmentation** | `jwcore/phase_segmentation.py` | Detects takeoff, airtime, and landing phases of a jump. Replaces older `phase_detect.py` with a cleaner, API-driven design. |
| 🏷️ **Trick Classification** | `jwcore/trick_classifier.py`, `scripts/train_trick_model.py` | Uses Random Forest and Logistic Regression models to predict trick type from feature vectors. |
| 🧾 **Evaluation & Metrics** | `scripts/eval_from_labels.py`, `scripts/plot_confusion_matrix.py` | Evaluates model performance; previous training achieved >95% accuracy on known trick datasets. |
| 🎮 **Comparison & Visualization** | `scripts/viz_compare_side_by_side.py` | Renders synchronized side-by-side playback of an amateur vs. pro trick, aligned by takeoff, landing, or apex. |
| 🧑‍🏫 **Coaching (Optional)** | `jwcore/coach.py`, `scripts/demo_cli.py` | Generates feedback text comparing motion quality and timing. Integrates OpenAI GPT-4o-mini for natural-language advice. |

---

## 📂 Directory Structure

Jumpworx/
│
├── scripts/ # CLI utilities & entry points
│ ├── demo_cli.py # Main demo runner (entry point)
│ ├── runner.py # Core orchestrator for the pipeline
│ ├── extract_keypoints.py # Pose extraction via Mediapipe
│ ├── viz_compare_side_by_side.py # Side-by-side visualization
│ ├── train_trick_model.py # Train classification models
│ └── eval_from_labels.py # Evaluate trained models
│
├── jwcore/ # Core reusable logic
│ ├── phase_segmentation.py # Refactored phase segmentation
│ ├── trick_classifier.py # Trick classification logic
│ ├── posetrack_io.py # Input/output utilities for pose tracks
│ ├── compare_metrics.py # Pose & motion similarity metrics
│ ├── coach.py # Feedback and LLM coaching interface
│ └── ...
│
├── cache/ # Generated NPZ pose files and features
├── features/ # JSON feature summaries
├── videos/ # Raw training and pro video clips
└── viz/ # Rendered comparison videos and metrics

yaml
Copy code

---

## ⚙️ Installation

**Requirements:**
- Python 3.10+
- [Mediapipe](https://developers.google.com/mediapipe)
- OpenCV
- NumPy, SciPy, scikit-learn, joblib
- (Optional) OpenAI Python SDK for LLM coaching

**Setup:**

```bash
git clone https://github.com/Iam1an/Jumpworx.git
cd Jumpworx
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
If you plan to use OpenAI for coaching:

bash
Copy code
export OPENAI_API_KEY="your_api_key_here"
🧪 Usage
1. Extract Keypoints
bash
Copy code
python scripts/extract_keypoints.py --input videos/training/TRICK45_BACKFLIP.mov
Creates:

pgsql
Copy code
cache/TRICK45_BACKFLIP.posetrack.npz
2. Run Full Demo (with AI coaching)
bash
Copy code
python -m scripts.demo_cli videos/training/TRICK45_BACKFLIP.mov \
  --runner scripts.runner \
  --extra-args "--pro_videos_dir videos/pro --strategy closest \
  --align dtw --align_feature ankle_y \
  --llm_provider openai --llm_model gpt-4o-mini"
This:

extracts or reuses cached poses

finds the most similar professional example

runs feature comparison, classification, and phase analysis

generates side-by-side visualization in /viz

optionally produces AI-written feedback

🧠 Key Learning Outcomes
Pose quality matters: Mediapipe’s keypoints are sensitive to camera angle and lighting. Multi-camera capture or better lighting drastically improves pose consistency.

Granular analysis beats averages: Early versions only averaged features, hiding frame-level motion nuance. Frame-by-frame joint metrics yielded much richer coaching signals.

Data completeness: Missing landmarks caused classifier noise; careful filtering and interpolation improved stability.

Modularity wins: Each stage (keypoints, phases, classification, viz) is its own module, making it easy to debug or upgrade independently.

🎓 Techniques & Models
Pose estimation: Mediapipe BlazePose full-body

Feature extraction: Temporal and biomechanical metrics (joint angles, airtime, body pitch)

Phase detection: Takeoff, airtime, landing segmentation with airtime validation

Classification: Random Forest and Logistic Regression (>95% accuracy)

Similarity scoring: Dynamic Time Warping, pose vector distances

Visualization: OpenCV compositing and skeletal overlays

Coaching LLM: GPT-4o-mini summarizing key performance gaps

📈 Example Outputs
Visualization:
A synchronized, labeled comparison video showing the amateur and professional performing the same trick side-by-side, with overlayed metrics and phase markers.

Metrics:
CSV and JSON summaries in /viz including:

Airtime difference

Takeoff/landing timing deltas

Overall trick similarity score

Coaching Output (example):

“Your takeoff is smooth, but your mid-air rotation speed is slightly lower than the reference. Focus on earlier hip extension to match pro airtime.”

🧹 Next Steps
Integrate multi-camera calibration for more reliable 3D landmarks.

Expand classifier to detect off-axis spins and grabs.

Improve automatic phase labeling and LLM contextual reasoning.

Extend visualization to 3D replay using Blender or Three.js.

🪪 License
This project is open-source under the MIT License.
You are free to use, modify, and distribute it with attribution.

yaml
Copy code

---

📦 **How to save it:**
1. Copy the above text into a new file named `README.md`.
2. Place it in the **root of your repo**.
3. Commit and push:
   ```bash
   git add README.md
   git commit -m "Add polished README with badges and overview"
   git push origin main