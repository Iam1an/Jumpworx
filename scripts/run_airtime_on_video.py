# scripts/run_airtime_on_video.py
# Airtime/phase report using shared cache + normalizer + v2 segmenter.

import os, sys, argparse, json
import numpy as np

# shared core
try:
    from jwcore import load_pose, normalize_prerot
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
except ImportError as e:
    print("ERROR: missing jwcore modules. Ensure jwcore/{io_cache,normalize,phase_segmentation}.py exist.")
    raise

# optional plotting (only if you pass --plot)
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# height series (+up) using toes then ankles, baseline-median
L_ANK, R_ANK = 27, 28
L_FTO, R_FTO = 31, 32
def height_series_up(kps_norm: np.ndarray) -> np.ndarray:
    toes_y = np.nanmean(kps_norm[:, [L_FTO, R_FTO], 1], axis=1)
    anks_y = np.nanmean(kps_norm[:, [L_ANK, R_ANK], 1], axis=1)
    both = np.stack([toes_y, anks_y], axis=1)
    y = np.nanmin(both, axis=1)
    baseline = np.nanmedian(y)
    return baseline - y  # +up

def main():
    ap = argparse.ArgumentParser(description="Compute airtime and phases for a single video.")
    ap.add_argument("--video", type=str, required=True, help="Path to video file")
    ap.add_argument("--fps_override", type=float, default=None, help="Override FPS (e.g., 60 for slo-mo clips)")
    ap.add_argument("--cache_dir", type=str, default="./cache", help="Pose cache directory")
    ap.add_argument("--mc", type=int, default=0, choices=[0,1,2], help="MediaPipe model complexity (only on cache miss)")
    ap.add_argument("--smooth_ms", type=int, default=50, help="Smoothing window (ms)")
    ap.add_argument("--hyst_frac", type=float, default=0.20, help="Hysteresis band fraction of ground-air range")
    ap.add_argument("--max_gap_ms", type=int, default=50, help="Merge contact gaps up to this many ms")
    ap.add_argument("--precontact_ms", type=int, default=250, help="Requested pre-contact window (soft)")
    ap.add_argument("--export_csv", type=str, default=None, help="Export height(+up) time series CSV here")
    ap.add_argument("--plot", action="store_true", help="Plot height(+up) with takeoff/landing (requires matplotlib)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"File not found: {args.video}")
        sys.exit(1)

    # 1) Load/cached pose (this runs MP only on cache miss)
    kps_pix, fps_file, w, h = load_pose(args.video, cache_dir=args.cache_dir, model_complexity=args.mc)
    fps = float(args.fps_override) if args.fps_override else float(fps_file)

    # 2) Normalize (pelvis-translate + torso-scale; no XY rotation)
    kps_norm = normalize_prerot(kps_pix)

    # 3) Segment phases
    phases = segment_phases_with_airtime_v2(
        np.dstack([kps_norm, np.zeros((kps_norm.shape[0], kps_norm.shape[1]), dtype=np.float32)]),
        fps=fps,
        smooth_win_ms=int(args.smooth_ms),
        hysteresis_frac=float(args.hyst_frac),
        max_gap_ms=int(args.max_gap_ms),
        require_precontact=True,
        min_precontact_ms=int(args.precontact_ms),
    )

    # 4) Report
    print("\n— Airtime Report —")
    print(f"Video:           {os.path.abspath(args.video)}")
    print(f"Frames:          {kps_norm.shape[0]} @ {fps:.2f} FPS (file: {fps_file:.2f})")
    print(f"Resolution:      {w}x{h}")
    print(f"Takeoff idx:     {phases.takeoff_idx}")
    print(f"Landing idx:     {phases.landing_idx}")
    print(f"Airtime (sec):   {phases.airtime_seconds if phases.airtime_seconds is not None else 'None'}")
    print(f"Phases (idx):    approach={phases.approach}, set={phases.set}, airtime={phases.airtime}, landing={phases.landing}")
    print(f"Quality:         {phases.quality}")

    # 5) Optional CSV export of height(+up)
    if args.export_csv:
        t = np.arange(kps_norm.shape[0]) / max(1e-6, fps)
        h_up = height_series_up(kps_norm)
        import csv
        with open(args.export_csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["time_s", "height_up"])
            for ti, hi in zip(t, h_up):
                wtr.writerow([f"{ti:.6f}", f"{hi:.6f}"])
        print(f"CSV exported:    {os.path.abspath(args.export_csv)}")

    # 6) Optional plot
    if args.plot:
        if not _HAS_PLT:
            print("matplotlib not installed. Run: pip install matplotlib")
        else:
            t = np.arange(kps_norm.shape[0]) / max(1e-6, fps)
            h_up = height_series_up(kps_norm)
            plt.figure()
            plt.plot(t, h_up, label="height (+up)")
            if phases.takeoff_idx is not None:
                plt.axvline(phases.takeoff_idx / fps, color="g", linestyle="--", label="takeoff")
            if phases.landing_idx is not None:
                plt.axvline(phases.landing_idx / fps, color="r", linestyle="--", label="landing")
            plt.xlabel("Time (s)"); plt.ylabel("Height (+up)")
            plt.title(os.path.basename(args.video))
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 7) Save a small JSON next to the video
    out_json = os.path.join(os.path.dirname(args.video), "airtime_report.json")
    with open(out_json, "w") as f:
        json.dump({
            "video": os.path.abspath(args.video),
            "frames": int(kps_norm.shape[0]),
            "fps_used": fps,
            "fps_file": fps_file,
            "resolution": [int(w), int(h)],
            "takeoff_idx": phases.takeoff_idx,
            "landing_idx": phases.landing_idx,
            "airtime_seconds": phases.airtime_seconds,
            "phases": {
                "approach": phases.approach,
                "set": phases.set,
                "airtime": phases.airtime,
                "landing": phases.landing
            },
            "quality": phases.quality
        }, f, indent=2)
    print(f"\nJSON saved:      {out_json}")

if __name__ == "__main__":
    main()
