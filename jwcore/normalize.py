# jwcore/normalize.py
import numpy as np

L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28
L_FTO, R_FTO = 31, 32

def _ffill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for t in range(1, out.shape[0]):
        bad = ~np.isfinite(out[t])
        if bad.any():
            out[t][bad] = out[t-1][bad]
    return out

def normalize_prerot(kps_pix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Pelvis-translate + torso-scale; DO NOT rotate XY. Returns (T,33,2).
    """
    T = kps_pix.shape[0]
    kps3 = np.dstack([kps_pix, np.zeros((T,33), dtype=np.float32)])  # (T,33,3)
    kps3 = _ffill(kps3)

    pelvis = 0.5 * (kps3[:, L_HIP, :] + kps3[:, R_HIP, :])
    shctr  = 0.5 * (kps3[:, L_SHO, :] + kps3[:, R_SHO, :])

    kps3 -= pelvis[:, None, :]
    torso_len = np.linalg.norm(shctr, axis=1) + eps
    vals = torso_len[np.isfinite(torso_len)]
    scale = np.median(vals) if vals.size else 1.0
    if not np.isfinite(scale) or scale < eps: scale = 1.0
    kps3 /= scale
    return kps3[:, :, :2]

def height_series_up(kps_norm: np.ndarray) -> np.ndarray:
    """
    Returns height (+up) using toes if present, else ankles.
    """
    toes_y = np.nanmean(kps_norm[:, [L_FTO, R_FTO], 1], axis=1)
    if not np.isfinite(toes_y).any():
        toes_y = np.nanmean(kps_norm[:, [L_ANK, R_ANK], 1], axis=1)
    baseline = np.nanmedian(toes_y)
    return baseline - toes_y
