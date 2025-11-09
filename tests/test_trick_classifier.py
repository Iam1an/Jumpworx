import os
import pytest

from jwcore.trick_classifier import TrickClassifier
from jwcore.pose_utils import FEATURE_KEYS

# ---- Minimal synthetic training rows (include rotation_sign) ----
PRO_BACKFLIP = {"airtime_s":0.85,"height_max":1.2,"height_mean":0.5,
                "angle_range":170,"angle_speed":22,"rotation_sign":-1.0,
                "n_frames":120,"fps":60}
PRO_SET      = {"airtime_s":0.25,"height_max":0.4,"height_mean":0.1,
                "angle_range":60,"angle_speed":8,"rotation_sign":+0.0,
                "n_frames":90,"fps":30}

def _has_sklearn(clf: TrickClassifier) -> bool:
    return getattr(clf, "model", None) is not None

def test_feature_key_source_of_truth():
    clf = TrickClassifier(model_base_path="./models/trick_model_test")
    # Classifier should adopt the same order as pose_utils.FEATURE_KEYS
    assert clf.feature_names == FEATURE_KEYS

def test_train_save_reload_and_predict(tmp_path):
    feats = [PRO_BACKFLIP, PRO_SET]
    labels = ["backflip", "set"]

    model_base = tmp_path / "trick_model"
    clf = TrickClassifier(model_base_path=str(model_base))

    if not _has_sklearn(clf):
        pytest.skip("scikit-learn not installed in this venv; skipping RF path")

    info = clf.train(feats, labels)
    assert os.path.exists(info["saved_pkl"])
    assert os.path.exists(info["saved_json"])

    # reload model and predict
    clf2 = TrickClassifier(model_base_path=str(model_base))
    pred = clf2.predict(PRO_BACKFLIP)
    assert pred in set(labels)

def test_predict_with_confidence_object_shape():
    clf = TrickClassifier(model_base_path="./models/trick_model_test2")
    out = clf.predict_with_conf(PRO_BACKFLIP)
    assert "label" in out and "confidence" in out

def test_heuristic_fallback_when_no_sklearn(monkeypatch):
    # Force the no-sklearn path and ensure we still get *some* label string
    monkeypatch.setattr("jwcore.trick_classifier._HAS_SK", False, raising=False)
    clf = TrickClassifier(model_base_path="./models/trick_model_dummy")
    label = clf.predict(PRO_BACKFLIP)
    assert isinstance(label, str)
