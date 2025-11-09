#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_cli.py

Thin wrapper to run runner.py (or another pipeline module) on a single video
and display a friendly, slightly dramatic CLI summary.

Usage:
  python -m scripts.demo_cli videos/training/TRICK16_FRONTFLIP.mov

Options:
  --runner PATH             Module to run as entrypoint (default: scripts.runner)
  --extra-args "..."        Extra args passed through to the runner module
  --delay SECONDS           Slow-typing delay per character (default: 0.01 unless --fast)
  --fast                    Disable slow typing (overrides --delay)
  --verbose                 Print raw JW_EVENT lines and stdout (no slow typing / progress bar)
  --no-color                Disable ANSI colors
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ========== Color helpers ==========

class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"


def colorize(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{Color.RESET}"


# ========== Slow typing helpers ==========

def slow_print(text: str, *, use_color: bool, color_code: Optional[str],
               slow: bool, char_delay: float, end: str = "\n") -> None:
    """Print text with optional color and slow-typing effect."""
    s = colorize(text, color_code, use_color) if color_code else text
    if not slow or char_delay <= 0.0:
        # Normal print
        sys.stdout.write(s + end)
        sys.stdout.flush()
        return

    # Slow typing
    for ch in s:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(char_delay)
    sys.stdout.write(end)
    sys.stdout.flush()


# ========== JW_EVENT parsing ==========

def parse_event(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith("JW_EVENT:"):
        return None
    try:
        return json.loads(line[len("JW_EVENT:"):])
    except Exception:
        return None


def stem(path: str) -> str:
    b = os.path.basename(path)
    return os.path.splitext(b)[0]


# ========== Progress bar ==========

# Logical milestones:
# 1: NPZ ready
# 2: Features extracted
# 3: Classification complete
# 4: Pro selected
# 5: Tips emitted
# 6: Viz complete
TOTAL_STEPS = 6


def progress_bar(step: int, total: int, width: int = 24) -> str:
    step = max(0, min(step, total))
    if total <= 0:
        return ""
    filled = int(round(width * step / float(total)))
    return "[" + "#" * filled + "." * (width - filled) + f"] ({step}/{total})"


# ========== Main ==========

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to amateur video")

    ap.add_argument(
        "--runner",
        default="scripts.runner",
        help="Module to run as entrypoint (default: scripts.runner)",
    )
    ap.add_argument(
        "--extra-args",
        default="",
        help="Extra args passed to runner module (e.g. \"--model_base models/jumpworx_model\")",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="Slow-typing delay per character. If 0, defaults to 0.01 unless --fast.",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Disable slow typing (overrides --delay).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw JW_EVENT lines and stdout; disables slow typing and progress bar.",
    )
    ap.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )

    args = ap.parse_args(argv)

    use_color = not args.no_color

    # Decide slow typing behavior
    if args.verbose or args.fast:
        slow = False
        char_delay = 0.0
    else:
        # If user set a delay, use it; else default to 0.01 for a subtle effect
        char_delay = args.delay if args.delay > 0.0 else 0.01
        slow = char_delay > 0.0

    # Build runner command
    # Use -W ignore to suppress noisy runtime warnings (e.g. NaN slice).
    cmd: List[str] = [
        sys.executable,
        "-W", "ignore",
        "-u",
        "-m",
        args.runner,
        "--amateur_video",
        args.video,
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    slow_print("▶ Starting runner", use_color=use_color,
               color_code=Color.CYAN, slow=slow, char_delay=char_delay)
    slow_print(f"  Video: {args.video}", use_color=use_color,
               color_code=Color.CYAN, slow=slow, char_delay=char_delay)
    slow_print(f"  Cmd:   {' '.join(shlex.quote(t) for t in cmd)}",
               use_color=use_color, color_code=Color.CYAN,
               slow=slow, char_delay=char_delay)
    slow_print("", use_color=use_color, color_code=None,
               slow=False, char_delay=0.0)  # blank line

    # State captured from JW_EVENTs
    final_label: Optional[str] = None
    final_proba: Optional[float] = None
    pro_clip: Optional[str] = None
    tips: List[str] = []
    viz_out_mp4: Optional[str] = None
    had_error = False
    printed_live_tips = False

    # Progress tracking
    prog_step = 0  # 0..TOTAL_STEPS

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            ev = parse_event(line)

            # Verbose mode: just dump everything, no slow typing / bar
            if args.verbose:
                if line.strip():
                    print(line)
            else:
                if ev is not None:
                    etype = ev.get("type")
                    msg = ev.get("msg", "")
                    name = ev.get("name")

                    # ===== Friendly UI =====
                    if etype == "stage":
                        if name == "start":
                            slow_print("▶ Starting pipeline",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)
                        elif name == "clf":
                            slow_print("▶ Loading classifier and predicting label",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)
                        elif name == "pick":
                            slow_print("▶ Selecting reference / pro clip",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)
                        elif name == "llm":
                            slow_print("▶ Generating coaching tips",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)
                        elif name == "viz":
                            slow_print("▶ Rendering side-by-side visualization",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)
                        else:
                            slow_print(f"▶ {msg}",
                                       use_color=use_color, color_code=Color.CYAN,
                                       slow=slow, char_delay=char_delay)

                    elif etype == "ok":
                        # NPZ ready
                        if msg in ("Amateur NPZ ready", "NPZ ready"):
                            slow_print("✓ Amateur NPZ ready",
                                       use_color=use_color, color_code=Color.GREEN,
                                       slow=slow, char_delay=char_delay)
                            prog_step = max(prog_step, 1)

                        # Features extracted → include feature count
                        elif msg == "Features extracted":
                            keys = ev.get("keys") or []
                            n = len(keys)
                            slow_print(f"✓ Features extracted ({n} feature(s))",
                                       use_color=use_color, color_code=Color.GREEN,
                                       slow=slow, char_delay=char_delay)
                            prog_step = max(prog_step, 2)

                        # Classification complete → show label + proba
                        elif msg == "Classification complete":
                            label = ev.get("label")
                            proba = ev.get("proba")
                            if label is not None and proba is not None:
                                slow_print(
                                    f"✓ Classification complete: {label} (p={float(proba):.3f})",
                                    use_color=use_color,
                                    color_code=Color.GREEN,
                                    slow=slow,
                                    char_delay=char_delay,
                                )
                            elif label is not None:
                                slow_print(
                                    "✓ Classification complete: "
                                    f"{label} (no probability available)",
                                    use_color=use_color,
                                    color_code=Color.GREEN,
                                    slow=slow,
                                    char_delay=char_delay,
                                )
                            else:
                                slow_print("✓ Classification complete",
                                           use_color=use_color,
                                           color_code=Color.GREEN,
                                           slow=slow,
                                           char_delay=char_delay)
                            prog_step = max(prog_step, 3)

                        # Pro selected → include clip name
                        elif msg == "Pro selected":
                            pro_video = ev.get("pro_video") or ""
                            pro_npz = ev.get("pro_npz") or ""
                            clip_name = stem(pro_video) if pro_video else stem(pro_npz) if pro_npz else "unknown"
                            slow_print(f"✓ Pro selected: {clip_name}",
                                       use_color=use_color, color_code=Color.GREEN,
                                       slow=slow, char_delay=char_delay)
                            prog_step = max(prog_step, 4)

                        # Viz complete
                        elif msg == "Visualization complete":
                            out_mp4 = ev.get("out_mp4")
                            if out_mp4:
                                slow_print(f"✓ Visualization complete: {out_mp4}",
                                           use_color=use_color, color_code=Color.GREEN,
                                           slow=slow, char_delay=char_delay)
                                viz_out_mp4 = out_mp4
                            else:
                                slow_print("✓ Visualization complete",
                                           use_color=use_color, color_code=Color.GREEN,
                                           slow=slow, char_delay=char_delay)
                            prog_step = max(prog_step, 6)

                        else:
                            slow_print(f"✓ {msg}",
                                       use_color=use_color, color_code=Color.GREEN,
                                       slow=slow, char_delay=char_delay)

                    elif etype == "warn":
                        slow_print(f"! {msg}",
                                   use_color=use_color, color_code=Color.YELLOW,
                                   slow=slow, char_delay=char_delay)

                    elif etype == "error":
                        slow_print(f"✗ {msg}",
                                   use_color=use_color, color_code=Color.RED,
                                   slow=slow, char_delay=char_delay)
                        had_error = True

                    elif etype == "done":
                        if msg == "Visualization complete":
                            out_mp4 = ev.get("out_mp4")
                            if out_mp4:
                                slow_print(f"✓ Visualization complete: {out_mp4}",
                                           use_color=use_color, color_code=Color.GREEN,
                                           slow=slow, char_delay=char_delay)
                                viz_out_mp4 = out_mp4
                            prog_step = max(prog_step, 6)

                else:
                    # Non-JW_EVENT stdout from runner
                    if line.strip():
                        slow_print(line,
                                   use_color=use_color,
                                   color_code=None,
                                   slow=slow,
                                   char_delay=char_delay)

            # ===== Capture structured info =====
            if ev is not None:
                etype = ev.get("type")
                msg = ev.get("msg", "")

                # Classification result
                if msg == "Classification complete":
                    if "label" in ev:
                        final_label = str(ev["label"])
                    if ev.get("proba") is not None:
                        try:
                            final_proba = float(ev["proba"])
                        except Exception:
                            final_proba = None

                # Pro selected
                if msg == "Pro selected":
                    pro_video = ev.get("pro_video") or ""
                    pro_npz = ev.get("pro_npz") or ""
                    if pro_video:
                        pro_clip = stem(pro_video)
                    elif pro_npz:
                        pro_clip = stem(pro_npz)

                # Coaching tips (store only; final summary printed after run)
                if etype == "tips":
                    items = ev.get("items") or []
                    tips = []
                    for t in items:
                        st = str(t).strip()
                        if st:
                            tips.append(st)
                    if items:
                        prog_step = max(prog_step, 5)


                # Viz path (if present)
                if etype in ("done", "ok") and msg == "Visualization complete":
                    if ev.get("out_mp4"):
                        viz_out_mp4 = ev["out_mp4"]

            # ===== Progress bar (non-verbose only) =====
            if not args.verbose and TOTAL_STEPS > 0:
                bar = progress_bar(prog_step, TOTAL_STEPS)
                if bar:
                    # Draw progress bar on its own line using carriage return tricks
                    sys.stdout.write("\r" + colorize(f"Progress {bar}", Color.CYAN, use_color))
                    sys.stdout.flush()

    finally:
        if not args.verbose and TOTAL_STEPS > 0:
            # Finish the progress bar line
            sys.stdout.write("\n")
            sys.stdout.flush()
        proc.wait()

    slow_print("", use_color=use_color, color_code=None, slow=False, char_delay=0.0)  # spacer

    # ===== Final summary =====
    slow_print("=== DEMO SUMMARY ===",
               use_color=use_color, color_code=Color.BOLD,
               slow=slow, char_delay=char_delay)

    # Prediction
    if final_label is not None:
        if final_proba is not None:
            slow_print(f"Prediction: {final_label} (p={final_proba:.3f})",
                       use_color=use_color, color_code=None,
                       slow=slow, char_delay=char_delay)
        else:
            slow_print(f"Prediction: {final_label} (no probability available)",
                       use_color=use_color, color_code=None,
                       slow=slow, char_delay=char_delay)
    else:
        slow_print("Prediction: (not reported)",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)

    # Reference
    if pro_clip is not None:
        slow_print(f"Reference clip: {pro_clip}",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)
    else:
        slow_print("Reference clip: (not reported)",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)

    # Coaching tips
    if tips:
        slow_print("", use_color=use_color, color_code=None,
                   slow=False, char_delay=0.0)
        slow_print("Coaching tips:",
                   use_color=use_color, color_code=Color.MAGENTA,
                   slow=slow, char_delay=char_delay)
        for i, t in enumerate(tips, 1):
            slow_print(f"  {i}. {t}",
                       use_color=use_color, color_code=None,
                       slow=slow, char_delay=char_delay)
    else:
        slow_print("Coaching tips: (none reported)",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)

    # Viz
    if viz_out_mp4:
        slow_print(f"Side-by-side video: {viz_out_mp4}",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)
    else:
        slow_print("Side-by-side video: (not reported)",
                   use_color=use_color, color_code=None,
                   slow=slow, char_delay=char_delay)

    if had_error or proc.returncode not in (0, None):
        return proc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
