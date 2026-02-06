# jwcore/__init__.py
from .normalize import normalize_prerot, height_series_up


def load_pose(*args, **kwargs):
    """Lazy wrapper so jwcore can be imported without cv2/mediapipe."""
    from .io_cache import load_pose as _load_pose
    return _load_pose(*args, **kwargs)
