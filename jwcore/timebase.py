# jwcore/timebase.py
import numpy as np

def _linear_resample(arr, src_fps: float, dst_fps: float):
    """
    arr: (T, ...) time-major array (e.g., kps_xyz or visibility)
    Returns resampled array with new length round(T * dst/src).
    NaNs are linearly interpolated along time where possible.
    """
    if src_fps <= 0 or dst_fps <= 0 or abs(src_fps - dst_fps) < 1e-6:
        return arr

    T = arr.shape[0]
    new_T = max(1, int(round(T * (dst_fps / src_fps))))

    # build time axes
    t_src = np.linspace(0.0, 1.0, T, dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, new_T, dtype=np.float32)

    flat = arr.reshape(T, -1).astype(np.float32)
    # handle NaNs by simple forward/back fill before interpolation
    mask = ~np.isfinite(flat)
    if mask.any():
        # forward fill
        idx = np.where(np.isfinite(flat), np.arange(T)[:, None], -1)
        np.maximum.accumulate(idx, axis=0, out=idx)
        ff = flat.copy()
        take = np.clip(idx, 0, T-1)
        ff = np.take_along_axis(ff, take, axis=0)
        # back fill
        idx2 = np.where(np.isfinite(flat), np.arange(T)[:, None], T)
        np.minimum.accumulate(idx2[::-1], axis=0, out=idx2[::-1])
        take2 = np.clip(idx2, 0, T-1)
        bf = np.take_along_axis(flat, take2, axis=0)
        # average ff & bf where original was NaN
        flat = np.where(mask, 0.5*(ff+bf), flat)

    out = np.empty((new_T, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, flat[:, j])
    return out.resha
