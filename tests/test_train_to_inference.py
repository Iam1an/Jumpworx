# tests/test_train_to_inference.py
#
# Integration test: train a small sklearn pipeline from synthetic features,
# save as a joblib bundle, load with TrickClassifier, and verify predict works.
#
# Deterministic (fixed seeds), no external files, no network.

import json
import os

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jwcore.pose_utils import features_from_posetrack, FEATURE_KEYS
from jwcore.trick_classifier import TrickClassifier


def _make_synthetic_posetrack(seed, pitch_bias):
    """Build a tiny (P, V, meta) with controllable pitch direction."""
    rng = np.random.RandomState(seed)
    T, J = 30, 33
    P = rng.randn(T, J, 3).astype(np.float32) * 10
    # Bias ankle-hip vector to produce different pitch_total_rad signs
    P[:, 27, 1] += pitch_bias * np.linspace(0, 1, T)  # left ankle y
    P[:, 28, 1] += pitch_bias * np.linspace(0, 1, T)  # right ankle y
    V = np.ones((T, J), dtype=np.float32)
    meta = {"fps": 30.0, "image_h": 720, "image_w": 1280}
    return P, V, meta


def _build_dataset(n_per_class=5):
    """Build a small labeled dataset from synthetic posetracks."""
    X, y = [], []
    for i in range(n_per_class):
        P, V, meta = _make_synthetic_posetrack(seed=100 + i, pitch_bias=+50.0)
        feats = features_from_posetrack(P, V, meta)
        X.append([float(feats[k]) for k in FEATURE_KEYS])
        y.append("backflip")

    for i in range(n_per_class):
        P, V, meta = _make_synthetic_posetrack(seed=200 + i, pitch_bias=-50.0)
        feats = features_from_posetrack(P, V, meta)
        X.append([float(feats[k]) for k in FEATURE_KEYS])
        y.append("frontflip")

    return np.array(X, dtype=np.float32), np.array(y)


def test_train_save_load_predict(tmp_path):
    """Full round-trip: train -> save bundle -> TrickClassifier load -> predict."""
    X, y = _build_dataset(n_per_class=5)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=200, random_state=42),
    )
    pipe.fit(X, y)

    # Save bundle in the format TrickClassifier expects
    bundle = {
        "model": pipe,
        "feature_names": list(FEATURE_KEYS),
        "class_names": list(pipe.classes_),
    }
    model_path = str(tmp_path / "test_model.pkl")
    joblib.dump(bundle, model_path)

    # Load with TrickClassifier
    clf = TrickClassifier(model_path)
    assert clf.feature_keys == list(FEATURE_KEYS)
    assert set(clf.class_names) == {"backflip", "frontflip"}

    # Predict with a feature dict
    test_feats = {k: float(X[0, i]) for i, k in enumerate(FEATURE_KEYS)}
    label = clf.predict(test_feats)
    assert isinstance(label, str)
    assert label in ("backflip", "frontflip")

    # Predict with proba
    label2, proba = clf.predict_with_proba(test_feats)
    assert isinstance(label2, str)
    assert label2 == label
    assert proba is not None
    assert 0.0 <= proba <= 1.0

    # Predict proba vector
    proba_dict = clf.predict_proba_vector(test_feats)
    assert proba_dict is not None
    assert set(proba_dict.keys()) == {"backflip", "frontflip"}
    assert abs(sum(proba_dict.values()) - 1.0) < 1e-6


def test_bundle_missing_model_key_raises(tmp_path):
    """TrickClassifier raises RuntimeError if bundle has no model pipeline."""
    bad_bundle = {"feature_names": FEATURE_KEYS, "class_names": ["a", "b"]}
    model_path = str(tmp_path / "bad_model.pkl")
    joblib.dump(bad_bundle, model_path)

    with pytest.raises(RuntimeError, match="no model pipeline"):
        TrickClassifier(model_path)


def test_bare_pipeline_bundle(tmp_path):
    """TrickClassifier loads a bare sklearn pipeline (not wrapped in dict)."""
    X, y = _build_dataset(n_per_class=3)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=200, random_state=42),
    )
    pipe.fit(X, y)

    model_path = str(tmp_path / "bare_model.joblib")
    joblib.dump(pipe, model_path)

    clf = TrickClassifier(model_path)
    assert clf.feature_keys == list(FEATURE_KEYS)

    test_feats = {k: float(X[0, i]) for i, k in enumerate(FEATURE_KEYS)}
    label = clf.predict(test_feats)
    assert label in ("backflip", "frontflip")


def test_feature_keys_mismatch_still_predicts(tmp_path):
    """Bundle with different feature_names still works (uses bundle's names)."""
    X, y = _build_dataset(n_per_class=3)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=200, random_state=42),
    )
    pipe.fit(X, y)

    custom_names = ["feat_a", "feat_b", "feat_c"]
    bundle = {
        "model": pipe,
        "feature_names": custom_names,
        "class_names": list(pipe.classes_),
    }
    model_path = str(tmp_path / "custom_model.pkl")
    joblib.dump(bundle, model_path)

    clf = TrickClassifier(model_path)
    assert clf.feature_keys == custom_names

    # Predict using the custom feature names
    test_feats = {k: float(X[0, i]) for i, k in enumerate(custom_names)}
    label = clf.predict(test_feats)
    assert label in ("backflip", "frontflip")
