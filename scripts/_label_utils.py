CANON = {
    "frontflip": "frontflip",
    "front": "frontflip",
    "ff": "frontflip",
    "backflip": "backflip",
    "back": "backflip",
    "bf": "backflip",
}

TARGET_DEFAULT = {"frontflip", "backflip"}

def canon_label(s: str) -> str:
    return CANON.get((s or "").strip().lower(), (s or "").strip().lower())
