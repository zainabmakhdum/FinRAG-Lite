"""
prompt_builder.py

Builds the full LLM prompt by:
- retrieving top-k snippets from a FAISS index over your doc chunks
- retrieving top-k candidate templates via embedding similarity
- optionally fetching + formatting recent news (EventRegistry)
- assembling the system/user/snippets/news/templates/instructions prompt

Expected local artifacts (same as your notebook):
- chunks.json
- chunk_meta.json
- faiss.index   (FAISS index for chunk embeddings)
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import requests
from sentence_transformers import SentenceTransformer, util


# =========================
# Templates (same content as notebook)
# =========================

TEMPLATES: List[Dict[str, Any]] = [
    # PERFORMANCE / SENTIMENT
    {
        "id": 1,
        "code": "STRONG_BULL_STABLE",
        "category": "Performance / Sentiment",
        "use_when": "Revenue, earnings, and key metrics are consistently strong; volatility and risk appear limited.",
        "pattern": (
            "Company ___ demonstrates consistently strong and stable performance, supported by ___ [Sx]. "
            "Growth in ___ and solid margins in ___ underpin a positive outlook, with only limited near-term risks from ___ [Sy]."
        ),
    },
    {
        "id": 2,
        "code": "MODERATE_BULL_IMPROVING",
        "category": "Performance / Sentiment",
        "use_when": "Trends are improving (growth re-accelerating, margins recovering) but not spectacular; some risks remain.",
        "pattern": (
            "Company ___ shows improving performance, with ___ trending higher and ___ stabilizing [Sx]. "
            "While risks such as ___ remain, the overall trajectory appears positive over the recent period [Sy]."
        ),
    },
    {
        "id": 3,
        "code": "NEUTRAL_MIXED",
        "category": "Performance / Sentiment",
        "use_when": "Signals are mixed (some metrics up, others down; conflicting commentary).",
        "pattern": (
            "Company ___ presents a mixed picture. On one hand, ___ has improved or remained resilient [Sx], "
            "but on the other, ___ has weakened or introduces uncertainty [Sy]. Overall, the outlook is balanced "
            "with both upside and downside factors to monitor."
        ),
    },
    {
        "id": 4,
        "code": "MODERATE_BEAR_WEAKENING",
        "category": "Performance / Sentiment",
        "use_when": "Key metrics are weakening, but not catastrophic; there is still runway.",
        "pattern": (
            "Company ___ appears to be weakening, as evidenced by declines in ___ and pressure on ___ [Sx]. "
            "Management commentary and external news highlight concerns around ___ [Sy]. While not yet distressed, "
            "the trend warrants caution."
        ),
    },
    {
        "id": 5,
        "code": "STRONG_BEAR_DISTRESSED",
        "category": "Performance / Sentiment",
        "use_when": "Clear distress: sharp revenue/earnings declines, liquidity stress, credit or going-concern risks.",
        "pattern": (
            "Company ___ shows signs of financial distress. Sharp declines in ___ and mounting pressures on ___ "
            "indicate a deteriorating position [Sx]. News and filings cite serious risks such as ___, which "
            "significantly increase downside risk for investors [Sy]."
        ),
    },
    {
        "id": 6,
        "code": "TURNAROUND_RECOVERY",
        "category": "Performance / Sentiment",
        "use_when": "Past weakness but recent credible improvements (restructuring, margin recovery, debt reduction).",
        "pattern": (
            "Company ___ appears to be in a turnaround phase. After prior weakness in ___ [Sx], recent developments "
            "such as ___ and improvements in ___ indicate early signs of recovery [Sy]. The trajectory is improving, "
            "but execution risk remains."
        ),
    },

    # GROWTH / RISK BALANCE
    {
        "id": 7,
        "code": "HIGH_GROWTH_HIGH_RISK",
        "category": "Growth / Risk Balance",
        "use_when": "Very strong growth but with meaningful uncertainties (early-stage, customer concentration, regulatory risk).",
        "pattern": (
            "Company ___ is delivering high growth, with strong expansion in ___ and rapid adoption in ___ [Sx]. "
            "However, this comes with elevated risk from factors such as ___ and dependence on ___ [Sy]. "
            "The profile is best characterized as high-growth, high-risk."
        ),
    },
    {
        "id": 8,
        "code": "MATURE_STABLE_INCOME",
        "category": "Growth / Risk Balance",
        "use_when": "Low-to-moderate growth, stable cash flows, often dividends; defensive profile.",
        "pattern": (
            "Company ___ operates as a mature, relatively stable business. Revenue and cash flows from ___ are steady, "
            "with limited but predictable growth [Sx]. This profile may appeal to investors seeking income and stability "
            "rather than aggressive upside."
        ),
    },
    {
        "id": 9,
        "code": "CYCLICAL_EXPOSURE",
        "category": "Growth / Risk Balance",
        "use_when": "Performance is clearly tied to macro/sector cycle (commodities, housing, autos, etc.).",
        "pattern": (
            "Company ___ is strongly exposed to the economic cycle, particularly through its dependence on ___ [Sx]. "
            "Current results reflect the phase of the cycle, with ___ benefiting/hurting performance [Sy]. Future outcomes "
            "will largely track broader sector and macro conditions."
        ),
    },
    {
        "id": 10,
        "code": "EVENT_DRIVEN_MIXED",
        "category": "Growth / Risk Balance",
        "use_when": "M&A, litigation, restructuring, or one-off events dominate the outlook (positive or negative, but uncertain).",
        "pattern": (
            "Company ___ is currently driven by specific events, including ___ [Sx]. These developments could materially "
            "reshape its financial profile, but their ultimate impact remains uncertain. Investors should focus on "
            "milestones such as ___ and potential outcomes around ___ [Sy]."
        ),
    },

    # RISK / BALANCE SHEET & GOVERNANCE
    {
        "id": 11,
        "code": "LEVERAGE_LIQUIDITY_CONCERN",
        "category": "Risk / Balance Sheet & Governance",
        "use_when": "High leverage, tight liquidity, covenant/rollover risk show up in filings or news.",
        "pattern": (
            "Company ___ carries notable balance sheet risk. Elevated leverage in ___ and liquidity pressures from ___ "
            "increase vulnerability to adverse conditions [Sx]. Management and external commentary highlight concerns "
            "around ___, which investors should monitor closely [Sy]."
        ),
    },
    {
        "id": 12,
        "code": "REGULATORY_OR_LEGAL_RISK",
        "category": "Risk / Balance Sheet & Governance",
        "use_when": "Material regulatory investigations, legal actions, or policy changes could hit the business.",
        "pattern": (
            "Company ___ faces material regulatory or legal risk. Filings and news highlight issues related to ___ and "
            "potential impacts from ___ [Sx]. While the ultimate outcome is uncertain, these exposures could significantly "
            "affect profitability or valuation [Sy]."
        ),
    },
    {
        "id": 13,
        "code": "GOVERNANCE_OR_EXECUTION_RISK",
        "category": "Risk / Balance Sheet & Governance",
        "use_when": "Management credibility, execution on strategy, or governance structures are flagged.",
        "pattern": (
            "For Company ___, governance and execution are key risks. Sources point to concerns around ___, management "
            "decisions on ___, or challenges delivering on ___ [Sx]. These factors may weigh on investor confidence even if "
            "the core fundamentals remain ___ [Sy]."
        ),
    },

    # DATA QUALITY / UNCERTAINTY
    {
        "id": 14,
        "code": "DATA_CONFLICTING_EVIDENCE",
        "category": "Data Quality / Uncertainty",
        "use_when": "Filings, news, or metrics disagree; you can’t form a clear directional call.",
        "pattern": (
            "The available evidence for Company ___ is conflicting. While some snippets indicate ___ [Sx], others highlight "
            "opposing signals such as ___ [Sy]. Given these inconsistencies, any conclusion should be treated with caution "
            "and updated as new information emerges."
        ),
    },

    # RISK FACTORS & EXPOSURES
    {
        "id": 15,
        "code": "RISK_FACTORS_OVERVIEW",
        "category": "Risk Factors & Exposures",
        "use_when": "User asks about main risk factors for a single company; multiple identifiable risk themes.",
        "pattern": (
            "Key risk factors for Company ___ include ___, ___, and ___ [Sx]. These relate to areas such as demand "
            "sensitivity in ___, operational or execution risk in ___, and financial or balance sheet exposure from ___ [Sy]. "
            "Taken together, they could affect growth, margins, or valuation if conditions worsen."
        ),
    },
    {
        "id": 16,
        "code": "MACRO_SECTOR_RISK_EXPOSURE",
        "category": "Risk Factors & Exposures",
        "use_when": "Company is highly exposed to macro variables or sector-wide conditions (rates, FX, commodities, etc.).",
        "pattern": (
            "Company ___ is meaningfully exposed to macro and sector conditions. Performance is sensitive to developments in "
            "___ (e.g., interest rates, commodity prices, or end-market demand) [Sx]. Recent commentary highlights that "
            "changes in ___ and broader sector trends in ___ are key swing factors for future results [Sy]."
        ),
    },
    {
        "id": 17,
        "code": "COMPETITIVE_INTENSITY_RISK",
        "category": "Risk Factors & Exposures",
        "use_when": "Competition, pricing pressure, or loss of share shows up as a key risk.",
        "pattern": (
            "For Company ___, competitive intensity is a notable risk. Sources point to pressure from rivals in ___, including "
            "pricing pressure in ___ and potential share loss in ___ [Sx]. If competition continues to intensify or new entrants "
            "emerge, this could weigh on revenue growth and margins [Sy]."
        ),
    },
    {
        "id": 18,
        "code": "CONCENTRATION_DEPENDENCE_RISK",
        "category": "Risk Factors & Exposures",
        "use_when": "Business depends heavily on a small number of customers, suppliers, products, or geographies.",
        "pattern": (
            "Company ___ faces concentration risk, with a significant dependence on ___ (e.g., a few key customers, products, "
            "or regions) [Sx]. Adverse changes such as contract losses, regulatory shifts in ___, or disruption at key suppliers "
            "could disproportionately impact results [Sy]. Diversification remains a medium-term mitigation focus."
        ),
    },
    {
        "id": 19,
        "code": "RISK_MITIGANTS_AND_CATALYSTS",
        "category": "Risk Factors & Exposures",
        "use_when": "User asks how risks might be mitigated or what could improve the situation; clear actions/catalysts exist.",
        "pattern": (
            "While Company ___ faces risks around ___ and ___ [Sx], there are also mitigating factors and potential catalysts. "
            "Management is pursuing measures such as ___ and ___, and upcoming events like ___ (e.g., product launches, regulatory "
            "decisions, or debt refinancings) could reduce risk or unlock upside [Sy]."
        ),
    },

    # PEER COMPARISON
    {
        "id": 20,
        "code": "PEER_COMPARISON_A_STRONGER",
        "category": "Peer Comparison & Relative Positioning",
        "use_when": "Evidence suggests Company A is clearly stronger overall than Company B.",
        "pattern": (
            "Compared with Company ___ (B), Company ___ (A) appears stronger overall. A shows better performance in ___ and a more "
            "resilient balance sheet or risk profile in ___ [Sx]. While B offers some strengths in ___, its exposure to risks such as "
            "___ makes it relatively less attractive on a risk-adjusted basis [Sy]."
        ),
    },
    {
        "id": 21,
        "code": "PEER_COMPARISON_B_STRONGER",
        "category": "Peer Comparison & Relative Positioning",
        "use_when": "Evidence suggests Company B is clearly stronger than Company A.",
        "pattern": (
            "Relative to Company ___ (B), Company ___ (A) looks weaker on several dimensions. A lags in areas such as ___ and faces "
            "higher risk from ___ [Sx]. By contrast, B benefits from stronger positioning in ___ and more manageable exposure to ___, "
            "suggesting a more favorable overall profile at present [Sy]."
        ),
    },
    {
        "id": 22,
        "code": "PEER_COMPARISON_MIXED_TRADEOFFS",
        "category": "Peer Comparison & Relative Positioning",
        "use_when": "Both companies have meaningful pros/cons; comparison is nuanced.",
        "pattern": (
            "Company ___ (A) and Company ___ (B) present a mixed comparison. A offers strengths in ___ [Sx], while B stands out in ___ "
            "[Sy]. The more attractive choice depends on whether the priority is upside potential (A) or stability and risk control (B)."
        ),
    },
    {
        "id": 23,
        "code": "PEER_COMPARISON_RISK_FOCUSED",
        "category": "Peer Comparison & Relative Positioning",
        "use_when": "Question is which of A or B is riskier, or how their risks compare.",
        "pattern": (
            "From a risk perspective, Company ___ faces greater exposure to ___, whereas Company ___ is more exposed to ___ [Sx]. "
            "A’s risk profile is driven by factors such as ___ and sensitivity to ___, while B’s key vulnerabilities relate to ___ [Sy]. "
            "Overall, ___ appears to carry higher downside risk, while ___ may be relatively more defensive, assuming current conditions persist."
        ),
    },

    # INVESTMENT VIEW / ALLOCATION
    {
        "id": 24,
        "code": "INVESTMENT_VIEW_A_PREFERRED",
        "category": "Investment View / Allocation",
        "use_when": "User asks whether to invest in A or B and evidence leans toward A.",
        "pattern": (
            "Based on the available information, Company ___ (A) currently looks more attractive than Company ___ (B) on a risk-reward basis. "
            "A offers advantages such as ___ [Sx], while B is weighed down by risks around ___ [Sy]. This is a general analytical view, not "
            "personalized investment advice; suitability still depends on your objectives, risk tolerance, and portfolio context."
        ),
    },
    {
        "id": 25,
        "code": "INVESTMENT_VIEW_B_PREFERRED",
        "category": "Investment View / Allocation",
        "use_when": "User asks whether to invest in A or B and evidence leans toward B.",
        "pattern": (
            "Given the current data, Company ___ (B) appears more compelling than Company ___ (A) from a risk-reward perspective. "
            "B benefits from strengths in ___ and more manageable exposure to ___ [Sx], whereas A faces headwinds from ___ or greater uncertainty around ___ [Sy]. "
            "This should be treated as a high-level assessment rather than individualized investment advice."
        ),
    },
    {
        "id": 26,
        "code": "INVESTMENT_VIEW_DEPENDS_ON_PROFILE",
        "category": "Investment View / Allocation",
        "use_when": "User asks about investing; correct framing is that it depends on risk/return preferences.",
        "pattern": (
            "Whether Company ___ is suitable as an investment depends heavily on your risk tolerance and goals. The company offers potential upside from ___ and ___ [Sx], "
            "but also carries risks related to ___ and ___ [Sy]. Investors seeking higher growth and willing to accept volatility may view the profile more positively, "
            "while more conservative investors might prefer steadier cash flows and fewer specific risk factors. This is general information, not personalized investment advice."
        ),
    },
    {
        "id": 27,
        "code": "INVESTMENT_VIEW_TOO_UNCERTAIN",
        "category": "Investment View / Allocation",
        "use_when": "User asks 'Should I invest?' but data is conflicting/limited.",
        "pattern": (
            "It is difficult to form a clear investment view on Company ___ based on the current information. While there are potential positives such as ___ [Sx], "
            "there are also material uncertainties or data gaps around ___ [Sy]. Given this uncertainty, any investment decision would require additional, up-to-date analysis "
            "of the company’s financials, competitive position, and your own circumstances."
        ),
    },
]


# =========================
# Prompt formatting helpers
# =========================

def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")  # nbsp
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def snippet_for_prompt(snippets: List[Dict[str, Any]], max_chars: int = 500) -> str:
    lines = ["Each snippet includes an ID and its source.\n"]
    for s in snippets:
        text = normalize_ws((s.get("text") or "").replace("\n", " "))
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        source_parts = []
        if s.get("filename"):
            source_parts.append(f"file={s['filename']}")
        if s.get("page_number") is not None:
            source_parts.append(f"page={s['page_number']}")
        source_str = " | ".join(source_parts)
        lines.append(f"[{s['snippet_id']}] ({source_str})\n{text}\n")
    return "\n".join(lines)


def templates_for_prompt(candidate_templates: List[Dict[str, str]]) -> str:
    lines = []
    lines.append("[CANDIDATE TEMPLATES]\n")
    lines.append("You must choose exactly ONE of the candidate templates below.\n")
    for i, t in enumerate(candidate_templates, start=1):
        lines.append(
            f"\n[T{i}] TEMPLATE_ID: {t['code']}\n"
            f"Use when: {t['use_when']}\n"
            f"Pattern: {t['pattern']}\n"
        )
    return "\n".join(lines)


# =========================
# Template retrieval (FAISS over template patterns)
# =========================

class TemplateRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.embed_model = SentenceTransformer(model_name)

        self.template_texts = [t["pattern"] for t in TEMPLATES]
        self.template_meta = [
            {"id": t["id"], "code": t["code"], "category": t["category"], "use_when": t["use_when"]}
            for t in TEMPLATES
        ]

        embs = self.embed_model.encode(self.template_texts, batch_size=16, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def get_candidate_templates(self, user_question: str, top_k: int = 6) -> List[Dict[str, str]]:
        q_emb = self.embed_model.encode([user_question]).astype("float32")
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, top_k)

        cleaned: List[Dict[str, str]] = []
        for idx in indices[0]:
            meta = self.template_meta[int(idx)]
            cleaned.append(
                {"code": meta["code"], "use_when": meta["use_when"], "pattern": self.template_texts[int(idx)]}
            )
        return cleaned


# =========================
# Snippet retrieval (FAISS over document chunks)
# =========================

class SnippetRetriever:
    def __init__(
        self,
        chunks_path: str,
        chunk_meta_path: str,
        faiss_index_path: str,
        embed_model_name: str = "BAAI/bge-small-en",
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(chunk_meta_path, "r", encoding="utf-8") as f:
            self.chunk_meta = json.load(f)

        self.index_vals = faiss.read_index(faiss_index_path)
        self.embed_model = SentenceTransformer(embed_model_name)

        if len(self.chunks) != len(self.chunk_meta):
            raise ValueError(f"chunks ({len(self.chunks)}) and chunk_meta ({len(self.chunk_meta)}) length mismatch")

    def retrieve_snippets(self, query: str, top_k: int = 7) -> List[Dict[str, Any]]:
        q_emb = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        distances, indices = self.index_vals.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if int(idx) == -1:
                continue
            meta = self.chunk_meta[int(idx)]
            text = self.chunks[int(idx)]
            results.append(
                {
                    "snippet_id": f"S{rank}",
                    "score": float(score),
                    "filename": meta.get("filename") or meta.get("file"),
                    "page_number": meta.get("page_number"),
                    "chunk_in_page": meta.get("chunk_in_page"),
                    "char_start": meta.get("char_start"),
                    "char_end": meta.get("char_end"),
                    "text": text,
                }
            )
        return results


# =========================
# Optional News (EventRegistry) – same logic, but safer
# =========================

SUPPORTED_COMPANIES = [
    {"name": "NVIDIA", "aliases": ["nvidia", "nvda", "nvidia corp", "nvidia corporation"]},
    {"name": "Intel", "aliases": ["intel", "intc", "intel corp", "intel corporation"]},
    {"name": "Samsung Electronics", "aliases": ["samsung", "samsung electronics", "samsung elec"]},
    {"name": "Taiwan Semiconductor", "aliases": ["tsmc", "taiwan semiconductor", "taiwan semi"]},
]

RELEVANT_BASE_KEYWORDS = [
    "revenue growth", "earnings", "profitability", "operating margin", "guidance", "forecast", "outlook",
    "cash flow", "capital expenditure", "debt", "liquidity", "dividends", "share buyback", "valuation",
    "competition", "competitive landscape", "emerging rivals", "market share", "regulatory risk",
    "macroeconomic risk", "geopolitical risk", "supply chain issues",
    "semiconductor", "AI chips", "data center", "cloud computing", "smartphones", "memory chips", "foundry business",
]

DEFAULT_NEWS_SOURCES = [
    "Bloomberg", "Reuters", "Financial Times", "Wall Street Journal", "Yahoo Finance", "CNBC",
    "MarketWatch", "Business Insider", "The Economist", "New York Times", "Washington Post",
    "BBC", "The Guardian", "AP News", "NPR",
]


def clean_query_format(text: str) -> str:
    table = str.maketrans({c: " " for c in string.punctuation})
    return text.lower().translate(table)


def interpret_sentiment(score: Optional[float]) -> str:
    if score is None:
        return "Unknown sentiment"
    if score > 0.35:
        return f"Positive ({score:.2f})"
    if score > 0.1:
        return f"Slightly Positive ({score:.2f})"
    if score >= -0.1:
        return f"Neutral ({score:.2f})"
    if score >= -0.35:
        return f"Slightly Negative ({score:.2f})"
    return f"Negative ({score:.2f})"


class NewsClient:
    def __init__(
        self,
        api_key: str,
        sources: Optional[List[str]] = None,
        keyword_model_name: str = "all-MiniLM-L6-v2",
    ):
        from eventregistry import EventRegistry  # keep import optional

        self.api_key = api_key
        self.er = EventRegistry(apiKey=api_key)
        self.news_embed_model = SentenceTransformer(keyword_model_name)

        self.sources = sources or DEFAULT_NEWS_SOURCES
        self.source_uris = {name: self.er.getSourceUri(name) for name in self.sources}

        # Precompute embeddings
        self.base_key_embed = self.news_embed_model.encode(RELEVANT_BASE_KEYWORDS, convert_to_tensor=True)

        alias_texts = []
        alias_to_company = []
        for c in SUPPORTED_COMPANIES:
            for alias in c["aliases"]:
                alias_texts.append(alias.lower())
                alias_to_company.append(c["name"])
        self.company_alias_texts = alias_texts
        self.company_alias_to_company = alias_to_company
        self.company_alias_embeds = self.news_embed_model.encode(alias_texts, convert_to_tensor=True)

    def detect_company_from_query(self, user_query: str, min_sim: float = 0.35) -> Optional[str]:
        cleaned = clean_query_format(user_query)
        q_emb = self.news_embed_model.encode(cleaned, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, self.company_alias_embeds)[0]
        best_score, best_idx = sims.max(dim=0)
        if float(best_score) < min_sim:
            return None
        return self.company_alias_to_company[int(best_idx)]

    def build_keywords_string(
        self,
        user_query: str,
        company: Optional[str] = None,
        max_terms: int = 3,
        min_sim: float = 0.35,
    ) -> str:
        if company is None:
            company = self.detect_company_from_query(user_query) or ""
        cleaned = clean_query_format(user_query)
        query_emb = self.news_embed_model.encode(cleaned, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, self.base_key_embed)[0]

        k = min(max_terms, len(RELEVANT_BASE_KEYWORDS))
        top_vals, top_idxs = sims.topk(k=k)
        chosen_terms = []
        for score, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            if score >= min_sim:
                chosen_terms.append(RELEVANT_BASE_KEYWORDS[idx])

        parts = [p for p in [company, *chosen_terms] if p]
        return " ".join(parts).strip()

    def fetch_news(
        self,
        user_query: str,
        company: Optional[str] = None,
        articles_count: int = 3,
        days_window: int = 7,
    ) -> List[Dict[str, Any]]:
        keyword_str = self.build_keywords_string(user_query=user_query, company=company, max_terms=3, min_sim=0.35)
        sources_list = list(self.source_uris.values())

        body = {
            "apiKey": self.api_key,
            "keyword": keyword_str,
            "lang": "eng",
            "sourceUri": sources_list,
            "articlesCount": articles_count,
            "articlesSortBy": "rel",
            "articlesSortByAsc": False,
            "dataType": ["news"],
            "isDuplicateFilter": "skipDuplicates",
            "forceMaxDataTimeWindow": days_window,
            "includeArticleBody": True,
            "includeArticleSentiment": True,
        }

        resp = requests.post(
            "https://eventregistry.org/api/v1/article/getArticles",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_articles = data.get("articles", {}).get("results", [])

        cleaned = []
        for art in raw_articles:
            cleaned.append(
                {
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("title"),
                    "date": art.get("dateTimePub"),
                    "url": art.get("url"),
                    "sentiment": interpret_sentiment(art.get("sentiment")),
                    "body": art.get("body"),
                }
            )
        return cleaned


def news_for_prompt(articles: List[Dict[str, Any]], max_articles: int = 3) -> str:
    if not articles:
        return (
            "[RECENT NEWS]\n"
            "Below are recent news articles related to the company.\n"
            "You can use them as supplementary context for sentiment, risks, and catalysts.\n"
            "No relevant news articles within the past 7 days were retrieved.\n"
        )

    lines = []
    lines.append("[RECENT NEWS]")
    lines.append("Below are recent news articles related to the company.")
    lines.append("You can use them as supplementary context for sentiment, risks, and catalysts.\n")

    for i, art in enumerate(articles[:max_articles], start=1):
        body = (art.get("body") or "").replace("\n", " ")
        body_preview = body[:300] + ("..." if len(body) > 300 else "")
        lines.append(f"[NEWS{i}] {art.get('title') or 'Untitled'}")
        lines.append(
            f"Source: {art.get('source') or 'Unknown'} | "
            f"Date: {art.get('date') or 'Unknown'} | "
            f"Sentiment: {art.get('sentiment') or 'Unknown'}"
        )
        lines.append(f"URL: {art.get('url') or 'N/A'}")
        lines.append(f"Article preview: {body_preview}")
        lines.append("")
    return "\n".join(lines)


# =========================
# Main: updated_build_llm_prompt
# =========================

class PromptBuilder:
    def __init__(
        self,
        snippet_retriever: SnippetRetriever,
        template_retriever: Optional[TemplateRetriever] = None,
        news_client: Optional[NewsClient] = None,
    ):
        self.snippet_retriever = snippet_retriever
        self.template_retriever = template_retriever or TemplateRetriever()
        self.news_client = news_client

    def updated_build_llm_prompt(
        self,
        user_question: str,
        top_k_snippets: int = 7,
        top_k_templates: int = 6,
        use_news: bool = True,
        news_articles: Optional[List[Dict[str, Any]]] = None,
        news_max_articles: int = 3,
        news_days_window: int = 7,
    ) -> Dict[str, Any]:
        # 1) retrieve snippets
        snippets = self.snippet_retriever.retrieve_snippets(user_question, top_k=top_k_snippets)

        # 2) retrieve templates
        candidate_templates = self.template_retriever.get_candidate_templates(user_question, top_k=top_k_templates)

        # 3) news (optional)
        if news_articles is None and use_news and self.news_client is not None:
            try:
                news_articles = self.news_client.fetch_news(
                    user_query=user_question,
                    articles_count=news_max_articles,
                    days_window=news_days_window,
                )
            except Exception:
                news_articles = None

        system_prompt = """[SYSTEM]
You are a cautious financial analysis assistant.

Company scope:
- In-scope companies: Nvidia (NVDA), Intel (INTC), TSMC / Taiwan Semiconductor Manufacturing Company (TSM), Samsung / Samsung Electronics.
- If the question is ONLY about companies outside this set:
  - Do NOT analyze them.
  - Use the OUT_OF_SCOPE_COMPANY template and explain that you lack data for those companies.
- If the question mixes in-scope and out-of-scope companies:
  - Clearly say you only have data for the in-scope names.
  - Focus the analysis on in-scope companies and say you cannot properly analyze the others.

Evidence rules:
- Treat snippets as the only source of truth.
- If snippets conflict, say so briefly.
- If snippets are sparse or don’t really answer the question, use the LIMITED_INFORMATION_RESPONSE template.
- Never invent numbers, dates, or details not supported by snippets.
- Never give personalized investment advice (no “you should buy/sell”).
- Use the snippets as evidence and cite them using [S1], [S2], etc.
- Use the news articles as supplementary context. Do not cite them.

High-level steps:
1. Identify which companies the user is asking about and whether they are in-scope or out-of-scope.
2. Review snippets for relevant information about the in-scope companies and the question.
3. Choose exactly ONE template whose “Use when” condition best matches the situation (performance, risk, comparison, limited info, out-of-scope, etc.).
4. Fill in the template pattern with grounded phrases from the snippets and respond in 2–5 sentences with citations.

Tone:
- Professional, neutral, and concise.
"""

        user_block = f"[USER QUESTION]\n{user_question}\n"
        snippets_block = snippet_for_prompt(snippets)
        templates_block = templates_for_prompt(candidate_templates)

        if use_news:
            news_block = news_for_prompt(news_articles or [], max_articles=news_max_articles)
        else:
            news_block = ""

        instructions_block = r"""
[FALLBACK TEMPLATES — ONLY IF NONE OF THE CANDIDATES APPLY]
[TX] TEMPLATE_ID: LIMITED_INFORMATION_RESPONSE
Use when: Snippets provide little or no relevant information to answer the question confidently.
Pattern: "The available information about Company ___ is limited. The snippets mainly discuss ___ [Sx] and provide little detail on ___ [Sy], so any conclusion would be uncertain."

[TY] TEMPLATE_ID: OUT_OF_SCOPE_COMPANY
Use when: The question is primarily or entirely about companies outside the allowed set.
Pattern: "I do not have sufficient data to analyze the companies mentioned in this question. This system is restricted to discussing Nvidia, Intel, TSMC, and Samsung, and the snippets do not cover the requested company or companies, so I cannot provide a reliable analysis."

[TZ] TEMPLATE_ID: FREEFORM_SUPPORTED
Use when: None of the templates cleanly match, but snippets do contain enough to answer.
Pattern: "Based on the snippets, … [Sx] … [Sy] …"

[RESPONSE FORMAT]
Return ONLY a single-line JSON object (no markdown, no extra text) with EXACTLY these keys:
{
  "template_id": "<ONE template ID you chose (prefer a candidate template; use fallback only if none apply) like MATURE_STABLE_INCOME>",
  "answer": "<2–5 sentences answering the user, grounded in snippets, with citations like [S1], [S2]>",
  "used_snippets": ["S1","S3"]
}
"""

        blocks = [system_prompt, user_block, snippets_block]
        if news_block:
            blocks.append(news_block)
        blocks.extend([templates_block, instructions_block])

        full_prompt = "\n\n".join(blocks)

        return {
            "prompt": full_prompt,
            "snippets": snippets,
            "templates": templates_block,
            "news_articles": news_articles,
        }