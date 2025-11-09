# tools/audit_cache_fps.py
import os, glob, json, numpy as np

def json_fps(path):
    try:
        with open(path) as f:
            d = json.load(f)
        fdict = d.get("features", d)
        return float(fdict.get("fps"))
    except Exception:
        return None

def main(dir_cache="cache", dir_feat="features"):
    npzs = sorted(glob.glob(os.path.join(dir_cache, "*.posetrack.npz")))
    for npz in npzs:
        stem = os.path.basename(npz).split(".posetrack.npz")[0]
        d = np.load(npz)
        fps_npz = float(d["fps"][0]) if "fps" in d else None

        # try both "<stem>.json" and "<stem> 2.json"
        candidates = [os.path.join(dir_feat, f"{stem}.json"),
                      os.path.join(dir_feat, f"{stem} 2.json")]
        fps_json = None
        for c in candidates:
            if os.path.exists(c):
                fps_json = json_fps(c)
                if fps_json: break

        flag = "" if (fps_npz and fps_json and abs(fps_npz - fps_json) < 0.5) else "  <-- MISMATCH"
        print(f"{stem:28s} npz:{fps_npz}  json:{fps_json}{flag}")

if __name__ == "__main__":
    main()
