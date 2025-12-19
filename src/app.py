from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import threading
import gradio as gr

# ---- your modules ----
from LLMCalls import load_model, run, parse_llm_json
import PromptMaker


# =========================
# Paths / Config
# =========================

SRC_DIR = Path(__file__).resolve().parent
EMBED_DIR = SRC_DIR / "embedding_data"

CHUNKS_PATH = Path(os.getenv("FINRAG_CHUNKS_PATH", str(EMBED_DIR / "chunks.json")))
META_PATH   = Path(os.getenv("FINRAG_META_PATH",   str(EMBED_DIR / "chunk_meta.json")))
FAISS_PATH  = Path(os.getenv("FINRAG_FAISS_PATH",  str(EMBED_DIR / "faiss.index")))

MODEL_PATH = os.getenv("FINRAG_MODEL_PATH", None)
MAX_TOKENS = int(os.getenv("FINRAG_MAX_TOKENS", "220"))

#EVENTREGISTRY_KEY = os.getenv("EVENTREGISTRY_API_KEY", None)

EVENTREGISTRY_KEY = "f8e1482c-c5cf-4f48-9049-31046857b5a9"

# =========================
# File → (Short Title, URL) mapping
# You said you'll provide the links — keep placeholders here.
# Match rules are substring-based so you don’t need exact filenames.
# =========================

FILE_SOURCES = [
    # ---- Intel ----
    {
        "match_any": ["intel2025_10k.json", "intel2025 10k"],
        "title": "Intel 10-K 2025",
        "url": "https://www.intc.com/filings-reports/annual-reports##document-5752-0000050863-25-000009-2",
    },
    {
        "match_any": ["intel2025_10qsep.json", "intel2025 10q sep", "10qsep"],
        "title": "Intel 10-Q (Sep 2025)",
        "url": "https://www.intc.com/filings-reports/quarterly-reports##document-5905-0000050863-25-000179-2",
    },

    # ---- NVIDIA ----
    {
        "match_any": ["nvidia-10k_2025jan.json", "nvidia 10k 2025 jan", "10k_2025jan"],
        "title": "NVIDIA 10-K (Jan 2025)",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q4/177440d5-3b32-4185-8cc8-95500a9dc783.pdf",
    },
    {
        "match_any": ["nvidia-10q_jul25.json", "nvidia 10q jul25", "10q_jul25"],
        "title": "NVIDIA 10-Q (Jul 2025)",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2026/q2/2e217538-c226-4d05-8f74-aaca89a21b33.pdf",
    },

    # ---- Samsung ----
    {
        # screenshot truncates, but this will still match your actual file
        "match_any": ["samsung_2024_4q_inter", "samsung 2024 4q inter"],
        "title": "Samsung 4Q 2024 Interim Report",
        "url": "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_4Q_Interim_Report.pdf",
    },
    {
        # screenshot truncates, but this will still match your actual file
        "match_any": ["samsung_2025_con_qua", "samsung 2025 con qua"],
        "title": "Samsung 2025 Consolidated Quarterly Report",
        "url": "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2025_con_quarter03_all.pdf",
    },

    # ---- TSMC ----
    {
        "match_any": ["tsmc_2024-10k.json", "tsmc 2024-10k", "2024-10k"],
        "title": "TSMC 10-K 2024",
        "url": "https://investor.tsmc.com/sites/ir/annual-report/2024/2024%20Annual%20Report.E.pdf",
    },
    {
        "match_any": ["tsmc_2025_q3.json", "tsmc 2025 q3", "2025_q3"],
        "title": "TSMC Q3 2025 Report",
        "url": "https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2025-10/5db3e377172cf60a48e4a3a2d7fb46963789ec51/FS.pdf",
    },
]

DEFAULT_SOURCE_TITLE = "Source Document"
DEFAULT_SOURCE_URL = "<UNKNOWN_SOURCE_URL>"  # placeholder


def _normalize_filename(filename: Optional[str]) -> str:
    return (filename or "").strip().lower()


def source_for_filename(filename: Optional[str]) -> Tuple[str, str]:
    """
    Returns (short_title, url) for the given snippet filename using FILE_SOURCES rules.
    """
    fn = _normalize_filename(filename)
    if not fn:
        return (DEFAULT_SOURCE_TITLE, DEFAULT_SOURCE_URL)

    for rule in FILE_SOURCES:
        for m in rule["match_any"]:
            if m.lower() in fn:
                return (rule["title"], rule["url"])

    return (DEFAULT_SOURCE_TITLE, DEFAULT_SOURCE_URL)


def url_with_page(url: str, page: Optional[int]) -> str:
    """
    Adds a page anchor to PDFs (common convention: #page=34).
    If page is None, returns the base url.
    """
    if not url:
        return url
    if page is None:
        return url
    try:
        p = int(page)
    except Exception:
        return url
    # only add if it looks like a PDF URL; otherwise still append #page=
    return f"{url}#page={p}"


def display_label(title: str, page: Optional[int]) -> str:
    if page is None:
        return title
    return f"{title} Page {page}"


# =========================
# Citation linking + final answer formatting
# =========================

# Matches [S1] or [S1, S4] etc.
CITATION_GROUP_RE = re.compile(r"\[(\s*S\d+(?:\s*,\s*S\d+)*\s*)\]")


def link_citations(answer_text: str, snippet_map: Dict[str, Dict[str, Any]]) -> str:
    """
    Converts:
      "... stabilizing [S1, S4] ..."
    into:
      "... stabilizing ([Samsung 10-K 2024 Page 34](URL#page=34), [TSMC 20-F 2024 Page 12](URL#page=12)) ..."
    """
    def repl(match: re.Match) -> str:
        group = match.group(1)
        ids = [x.strip() for x in group.split(",")]

        links = []
        for sid in ids:
            snip = snippet_map.get(sid)
            filename = (snip or {}).get("filename") or (snip or {}).get("file")
            page = (snip or {}).get("page_number")
            title, url = source_for_filename(filename)

            label = display_label(title, page)
            href = url_with_page(url, page)

            # If url placeholder missing, still show label (no link)
            if href and "<" not in href:
                links.append(f"[{label}]({href})")
            else:
                links.append(f"{label}")

        return "(" + ", ".join(links) + ")"

    return CITATION_GROUP_RE.sub(repl, answer_text)


def pick_top_snippets(
    used_snippets: List[str],
    snippet_map: Dict[str, Dict[str, Any]],
    fallback_snippets: List[Dict[str, Any]],
    k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Picks up to k snippet dicts.
    Prefer snippets referenced by the model (used_snippets), then fill from retrieved snippets.
    """
    chosen: List[Dict[str, Any]] = []
    seen_ids = set()

    for sid in used_snippets:
        if sid in snippet_map and sid not in seen_ids:
            chosen.append(snippet_map[sid])
            seen_ids.add(sid)
            if len(chosen) >= k:
                return chosen

    for snip in fallback_snippets:
        sid = snip.get("snippet_id")
        if sid and sid not in seen_ids:
            chosen.append(snip)
            seen_ids.add(sid)
            if len(chosen) >= k:
                return chosen

    return chosen


def snippet_preview_line(snip: Dict[str, Any], max_chars: int = 320) -> str:
    filename = snip.get("filename") or snip.get("file")
    page = snip.get("page_number")

    title, url = source_for_filename(filename)
    label = display_label(title, page)
    href = url_with_page(url, page)

    text = (snip.get("text") or "").strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    if href and "<" not in href:
        return f"- [{label}]({href}): {text}"
    return f"- {label}: {text}"


def pick_one_news(news_articles: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not news_articles:
        return None
    return news_articles[0]


def format_final_output(parsed: Dict[str, Any], prompt_payload: Dict[str, Any]) -> str:
    answer = parsed.get("answer", "") or ""
    used_snippets = parsed.get("used_snippets", []) or []

    snippets: List[Dict[str, Any]] = prompt_payload.get("snippets", []) or []
    snippet_map = {s.get("snippet_id"): s for s in snippets if s.get("snippet_id")}

    # 1) Link citations inside the answer
    answer_linked = link_citations(answer, snippet_map)

    # 2) Two snippet previews
    top2 = pick_top_snippets(used_snippets, snippet_map, snippets, k=2)
    if top2:
        snippet_lines = "\n".join(snippet_preview_line(s) for s in top2)
        snippets_block = f"**Snippets:**\n{snippet_lines}"
    else:
        snippets_block = "**Snippets:** None"

    # 3) One news item (if any)
    news = pick_one_news(prompt_payload.get("news_articles"))
    if news:
        n_url = news.get("url") or ""
        n_title = news.get("title") or "NEWS1"
        n_source = news.get("source") or "Unknown source"
        n_date = news.get("date") or "Unknown date"

        if n_url:
            news_block = f"**News:**\n \n [NEWS1]({n_url}) — **{n_title}** ({n_source}, {n_date})"
        else:
            news_block = f"**News:**\n \nNEWS1 — **{n_title}** ({n_source}, {n_date})"
    else:
        news_block = "**News:**\n \n None"

    return "\n\n".join([answer_linked.strip(), snippets_block, news_block]).strip()


# =========================
# Build PromptMaker objects once (startup)
# =========================

def build_prompt_builder():
    snippet_retriever = PromptMaker.SnippetRetriever(
        chunks_path=str(CHUNKS_PATH),
        chunk_meta_path=str(META_PATH),
        faiss_index_path=str(FAISS_PATH),
    )

    tmpl = None
    if hasattr(PromptMaker, "TemplateRetriever"):
        tmpl = PromptMaker.TemplateRetriever()

    news_client = None
    if EVENTREGISTRY_KEY and hasattr(PromptMaker, "NewsClient"):
        try:
            news_client = PromptMaker.NewsClient(api_key=EVENTREGISTRY_KEY)
        except Exception:
            news_client = None

    if not hasattr(PromptMaker, "PromptBuilder"):
        raise ImportError("PromptMaker.py must expose a PromptBuilder class.")

    if tmpl is not None:
        return PromptMaker.PromptBuilder(
            snippet_retriever=snippet_retriever,
            template_retriever=tmpl,
            news_client=news_client,
        )

    return PromptMaker.PromptBuilder(
        snippet_retriever=snippet_retriever,
        news_client=news_client,
    )

PROMPT_BUILDER = build_prompt_builder()

# Load LLM once at startup
load_model(MODEL_PATH)


# =========================
# Gradio backend
# =========================

def answer_query(user_query: str):
    user_query = (user_query or "").strip()
    if not user_query:
        yield "Please enter a question."
        return

    t0 = time.time()

    def render(status_lines: List[str]) -> str:
        elapsed = time.time() - t0
        # show elapsed time prominently
        return (
            f"⏳ **Running FinRAG-Lite...**  \n"
            f"**Elapsed:** `{elapsed:0.001f}s`\n\n"
            + "\n".join(status_lines)
        )

    status = [
        "- ⏳ Building prompt (embeddings)",
        "- ⏳ Running local LLM",
        "- ⏳ Formatting citations/snippets/news",
    ]
    yield render(status)

    # ---- Step 1: build prompt ----
    payload = PROMPT_BUILDER.updated_build_llm_prompt(
        user_question=user_query,
        top_k_snippets=7,
        top_k_templates=6,
        use_news=True,
    )
    prompt = payload["prompt"]

    status[0] = "- ✅ Building prompt (embeddings)"
    yield render(status)

    # ---- Step 2: run LLM in a background thread so we can tick time ----
    result_holder = {"raw": None, "err": None}
    done = threading.Event()

    def _worker():
        try:
            result_holder["raw"] = run(prompt, max_tokens=MAX_TOKENS, model_path=MODEL_PATH)
        except Exception as e:
            result_holder["err"] = e
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    # while running, update elapsed time + keep LLM step as running
    tick = 0
    while not done.is_set():
        # optional: make the "running LLM" line animate a bit
        dots = "." * (tick % 4)
        status[1] = f"- ⏳ Running local LLM{dots}"
        yield render(status)
        tick += 1
        time.sleep(0.6)  # update frequency (0.3–1.0s is nice)

    # handle LLM error if any
    if result_holder["err"] is not None:
        status[1] = "- ❌ Running local LLM"
        yield render(status) + f"\n\n**Error:** `{result_holder['err']}`"
        return

    status[1] = "- ✅ Running local LLM"
    yield render(status)

    # ---- Step 3: parse + format ----
    raw = result_holder["raw"]
    parsed = parse_llm_json(raw)
    final_text = format_final_output(parsed, payload)

    status[2] = "- ✅ Formatting citations/snippets/news"
    elapsed = time.time() - t0

    yield f"{final_text}\n\n<sub>Completed in {elapsed:.1f}s</sub>"


# =========================
# UI styling
# =========================

CSS = """
:root {
  --mp-blue: #2563eb;
  --mp-blue-dark: #1d4ed8;
  --mp-bg: #f3f8ff;
}

body { background: var(--mp-bg) !important; }
.gradio-container { background: var(--mp-bg) !important; }

/* Centered title + subtitle */
#mp_title { text-align: center; font-size: 32px; font-weight: 800; color: var(--mp-blue-dark); margin-bottom: 0; }
#mp_subtitle { text-align: center; font-size: 16px; color: #475569; margin-top: 0px; margin-bottom: 6px; }

/* Button styling */
#run_btn button {
  background: var(--mp-blue) !important;
  border: 1px solid var(--mp-blue-dark) !important;
  color: white !important;
  font-weight: 700 !important;
}
#run_btn button:hover {
  background: var(--mp-blue-dark) !important;
}

/* Blue-ish borders on inputs/outputs */
textarea, input, .gr-textbox, .gr-markdown, .gr-box, .gr-panel {
  border-color: rgba(37, 99, 235, 0.35) !important;
}
"""


with gr.Blocks(title="FinRAG-Lite") as demo:
    gr.Markdown("<div id='mp_title'>FinRAG-Lite</div>")
    gr.Markdown("<div id='mp_subtitle'>An app by Brandon Rodriguez, Zainab Makhdum, Mani Karunanidhi, and Ayon Roy<br> Enter a question about Nvidia / Intel / TSMC / Samsung to understand a company’s performance, risks, trajectory, and recent developments.<br> The average time (using a M1 Macbook on following tuning instructions) for a question is 180 seconds.</div>")

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g., What risks are impacting NVIDIA?, How is Intel doing? Summarize recent Samsung performance.",
        lines=2,
    )

    btn = gr.Button("Run", elem_id="run_btn")

    out = gr.Markdown(label="Answer")
    #dbg = gr.Textbox(label="Debug (parsed JSON)", lines=6)

    #btn.click(fn=answer_query, inputs=inp, outputs=[out, dbg])
    btn.click(fn=answer_query, inputs=inp, outputs=[out])
    # inp.submit(fn=answer_query, inputs=inp, outputs=[out, dbg])
    inp.submit(fn=answer_query, inputs=inp, outputs=[out])


if __name__ == "__main__":
    demo.launch(css = CSS, share = True)