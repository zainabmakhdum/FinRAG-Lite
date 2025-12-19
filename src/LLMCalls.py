# LLMCalls.py
from __future__ import annotations

import os
import re
import json
import ast
import threading
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# MLX imports (loaded at import-time)
# ----------------------------
try:
    from mlx_lm import load, generate
except Exception as e:
    raise ImportError(
        "mlx_lm is required for LLMCalls.py. "
        "Install it in the same environment as your app (e.g., on macOS with MLX)."
    ) from e


# ----------------------------
# Globals (singleton model/tokenizer)
# ----------------------------
_MODEL = None
_TOKENIZER = None
_LOAD_LOCK = threading.Lock()

DEFAULT_MODEL_PATH = os.getenv("FINRAG_MODEL_PATH", "br2835/mistraladaptmerged-mlx-3bit")


def load_model(
    model_path: Optional[str] = None,
    *,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """
    Loads the MLX model/tokenizer once and caches them for future calls.

    - model_path defaults to env FINRAG_MODEL_PATH or DEFAULT_MODEL_PATH
    - tokenizer_config defaults to {"fix_mistral_regex": True} (same as your notebook)
    """
    global _MODEL, _TOKENIZER

    mp = model_path or DEFAULT_MODEL_PATH
    tok_cfg = tokenizer_config if tokenizer_config is not None else {"fix_mistral_regex": True}

    with _LOAD_LOCK:
        if _MODEL is not None and _TOKENIZER is not None and not force_reload:
            return _MODEL, _TOKENIZER

        _MODEL, _TOKENIZER = load(mp, tokenizer_config=tok_cfg)
        return _MODEL, _TOKENIZER


def run(
    prompt_text: str,
    *,
    max_tokens: int = 190,
    model_path: Optional[str] = None,
) -> str:
    """
    Runs inference and returns the raw model output string.

    This mirrors your notebook:
      - wraps prompt_text as a single user message
      - uses tokenizer.apply_chat_template(..., add_generation_prompt=True)
      - calls mlx_lm.generate(...)
    """
    model, tokenizer = load_model(model_path)

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # MLX generate() kwargs can vary by version; keep this maximally compatible.
    try:
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    except TypeError:
        # Some versions take positional prompt
        return generate(model, tokenizer, prompt, max_tokens=max_tokens)


# ----------------------------
# Parsing helpers
# ----------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Parse the model output into:
      {
        "template_id": str,
        "answer": str,
        "used_snippets": list[str]
      }

    Handles:
    - code fences ```json ... ```
    - extra leading/trailing text (extract first {...})
    - JSON output (double quotes)
    - Python dict-ish output (single quotes) via ast.literal_eval
    """
    t = _strip_code_fences(text)

    m = _JSON_BLOCK_RE.search(t)
    if not m:
        raise ValueError("No JSON-like object found in model output.")

    payload = m.group(0).strip()

    obj: Any = None

    # 1) Strict JSON
    try:
        obj = json.loads(payload)
    except Exception:
        # 2) Python literal dict (single quotes, etc.)
        try:
            obj = ast.literal_eval(payload)
        except Exception as e:
            raise ValueError(f"Could not parse model output as JSON or Python dict. Error: {e}") from e

    if not isinstance(obj, dict):
        raise TypeError(f"Parsed output is not a dict. Got {type(obj)}")

    required = {"template_id", "answer", "used_snippets"}
    if not required.issubset(set(obj.keys())):
        raise ValueError(f"Missing required keys. Expected at least {required}, got {set(obj.keys())}")

    # Normalize output (only keep what your app expects)
    template_id = obj.get("template_id", "")
    answer = obj.get("answer", "")
    used_snippets = obj.get("used_snippets", [])

    # Normalize types
    template_id = "" if template_id is None else str(template_id)
    answer = "" if answer is None else str(answer)

    if used_snippets is None:
        used_snippets = []
    elif isinstance(used_snippets, str):
        # Sometimes models return "['S1','S3']" or "S1, S3"
        try:
            tmp = ast.literal_eval(used_snippets)
            used_snippets = tmp if isinstance(tmp, list) else [used_snippets]
        except Exception:
            used_snippets = [x for x in re.split(r"[,\s]+", used_snippets) if x]

    if not isinstance(used_snippets, list):
        raise TypeError("used_snippets must be a list[str] (or parseable into one).")

    used_snippets = [str(x) for x in used_snippets]

    return {
        "template_id": template_id,
        "answer": answer,
        "used_snippets": used_snippets,
    }