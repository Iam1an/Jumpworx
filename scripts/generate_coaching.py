#!/usr/bin/env python3
# generate_coaching.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, List

# Ensure repo root is on path so `jwcore` is importable when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import core coach logic from jwcore
try:
    from jwcore import coach as coach_mod
except Exception as e:
    print(f"[generate_coaching] ERROR: unable to import jwcore.coach: {e}", file=sys.stderr)
    raise

CoachLLM = Callable[[str], str]


# -----------------------------
# Utilities
# -----------------------------
def _read_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _dedupe_tips(tips: List[str], max_n: int = 5) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tips:
        key = " ".join(t.lower().split())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(t.strip())
        if len(out) >= max_n:
            break
    return out


def _warn(msg: str) -> None:
    print(f"[generate_coaching] {msg}", file=sys.stderr)


# -----------------------------
# LLM callables (optional)
# -----------------------------
def _make_openai_llm(
    model: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_tokens: int,
) -> CoachLLM:
    """
    Returns a callable(prompt: str) -> str using openai SDK if available.
    If not available or no key, raises RuntimeError so caller can fallback.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK not available: {e}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    def _call(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise, expert freestyle coach. "
                                   "Follow the instructions in the prompt exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
            )
            txt = resp.choices[0].message.content or ""
            return txt.strip()
        except Exception as e:
            _warn(f"OpenAI call failed: {e}")
            return ""

    return _call


def _make_anthropic_llm(
    model: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_tokens: int,
) -> CoachLLM:
    """
    Returns a callable(prompt: str) -> str using anthropic SDK if available.
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Anthropic SDK not available: {e}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    def _call(prompt: str) -> str:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=[{"role": "user", "content": prompt}],
            )
            parts: List[str] = []
            for blk in resp.content:
                if getattr(blk, "type", "") == "text":
                    parts.append(getattr(blk, "text", ""))
            return "\n".join(p.strip() for p in parts if p).strip()
        except Exception as e:
            _warn(f"Anthropic call failed: {e}")
            return ""

    return _call


def _make_llm_callable(
    provider: str,
    model: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    presence_penalty: float = 0.6,
    frequency_penalty: float = 0.4,
    max_tokens: int = 220,
) -> CoachLLM:
    """
    Factory: returns a prompt->text callable for the requested provider.
    Raises RuntimeError if the provider can't be initialized.
    Exposed for run_master.py (legacy / integration).
    """
    provider = (provider or "").strip().lower()
    if provider in ("", "none", "off", "disabled"):
        raise RuntimeError("LLM disabled by provider")

    if provider in ("openai", "oai"):
        return _make_openai_llm(
            model, temperature, top_p, presence_penalty, frequency_penalty, max_tokens
        )
    if provider in ("anthropic", "claude"):
        return _make_anthropic_llm(
            model, temperature, top_p, presence_penalty, frequency_penalty, max_tokens
        )

    raise RuntimeError(f"Unknown provider '{provider}'")


# -----------------------------
# Orchestration
# -----------------------------
def generate(
    features: Dict[str, Any],
    predicted_label: Optional[str],
    pro_name: Optional[str],
    pro_features: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """
    Entry point for programmatic usage.
    - Attempts to build an LLM callable unless no_llm=True.
    - Delegates near-match/self-compare handling to jwcore.coach.coach (rule fallback).
    - De-duplicates & trims tips and returns a structured dict.
    """
    llm: Optional[CoachLLM] = None
    if not no_llm:
        try:
            llm = _make_llm_callable(provider=provider, model=model)
        except RuntimeError as e:
            _warn(f"LLM not available; falling back to rule-based: {e}")
            llm = None

    result = coach_mod.coach(
        features=features,
        predicted_label=predicted_label,
        pro_name=pro_name,
        pro_features=pro_features,
        llm=llm,
    )

    # Post-process tips
    tips = _dedupe_tips(result.get("tips", []), max_n=5)
    if not tips:
        _warn("Empty tips from primary path; retrying with rule-based fallback.")
        rb = coach_mod.coach(
            features=features,
            predicted_label=predicted_label,
            pro_name=pro_name,
            pro_features=pro_features,
            llm=None,
        )
        tips = _dedupe_tips(rb.get("tips", []), max_n=5)
        if rb and rb.get("meta") and not result.get("meta"):
            result["meta"] = rb["meta"]
        result["source"] = "rule"

    result["tips"] = tips

    # Minimal audit trail
    meta = result.get("meta", {}) or {}
    meta.setdefault("provider", provider if not no_llm else "none")
    meta.setdefault("model", model if not no_llm else "")
    result["meta"] = meta

    return result


# -----------------------------
# CLI
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate coaching cues from phase-aware features (with optional LLM)."
    )
    p.add_argument("--features_am", type=str, required=True,
                   help="Path to athlete features JSON (phase-aware keys).")
    p.add_argument("--features_pro", type=str, default=None,
                   help="Optional path to pro/reference features JSON.")
    p.add_argument("--label", type=str, default=None,
                   help="Optional predicted label / trick family.")
    p.add_argument("--pro_name", type=str, default=None,
                   help="Optional reference/pro name for context.")
    p.add_argument("--out", type=str, required=True,
                   help="Path to write coaching JSON.")
    # LLM config
    p.add_argument("--provider", type=str, default="openai",
                   help="LLM provider: openai | anthropic | none")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                   help="Provider model id")
    p.add_argument("--no-llm", action="store_true",
                   help="Force rule-based output (no LLM calls).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    am = _read_json(args.features_am)
    if am is None:
        _warn(f"Missing athlete features: {args.features_am}")
        return 2

    pro = _read_json(args.features_pro) if args.features_pro else None

    result = generate(
        features=am,
        predicted_label=args.label,
        pro_name=args.pro_name,
        pro_features=pro,
        provider=args.provider,
        model=args.model,
        no_llm=args.no_llm,
    )

    _write_json(args.out, result)
    print(f"[generate_coaching] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
