"""Shared data prep, BM25 shortlist, and metrics for the reranker comparison.

Mirrors the setup in the TypeSafe "Re-ranking" cookbook so that numbers from
different rerankers are directly comparable:

  * CLERC legal retrieval dataset, 170 rows pooled into one corpus
  * 40 of those rows evaluated as queries, the other 130 only as candidates
  * BM25 shortlists 30 candidates per query
  * seed = 0
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)

CLERC_FILE = (
    "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/main/"
    "teva_train_dir/train_data.jsonl.gz"
)

N_ROWS = 170  # CLERC rows pooled into the shared corpus
N_QUERIES = 40  # rows evaluated as queries
TOP_K = 30  # candidates handed to the reranker, per query
SEED = 0
BM25_K = 100  # BM25 retrieves this many, we keep the first TOP_K


def cid(text: str) -> str:
    """Corpus id: a content hash, so passages shared across queries dedupe."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_slice(n_rows: int = N_ROWS, n_queries: int = N_QUERIES, seed: int = SEED) -> dict:
    """Stream CLERC rows, pool `n_rows` into a corpus, pick `n_queries` to evaluate."""
    cache = ART / f"slice_{n_rows}_{n_queries}_{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    from datasets import load_dataset

    stream = load_dataset("json", data_files=CLERC_FILE, streaming=True, split="train")
    rows = []
    for row in stream:
        if row.get("positive_passages") and len(row.get("negative_passages") or []) == 20:
            rows.append(row)
        if len(rows) >= 1000:
            break

    rng = random.Random(seed)
    picked = rng.sample(rows, n_rows)
    corpus: dict[str, str] = {}
    pool: list[dict] = []
    for row in picked:
        gold = row["positive_passages"][0]["text"]
        corpus[cid(gold)] = gold
        for neg in row["negative_passages"]:
            corpus[cid(neg["text"])] = neg["text"]
        pool.append({"qid": str(row["query_id"]), "query": row["query"], "gold": cid(gold)})

    # hold out the first 20 pooled rows; evaluate on the rest
    queries = rng.sample(pool[20:], n_queries)
    # sort the corpus by id so every run iterates it identically
    ds = {"queries": queries, "corpus": dict(sorted(corpus.items()))}
    cache.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
    return ds


def bm25_rankings(corpus: dict[str, str], queries: dict[str, str], k: int = BM25_K) -> dict:
    """Rank every passage in the corpus by word overlap with each query."""
    import bm25s

    cache = ART / f"bm25_top{k}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    cids = list(corpus)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([corpus[c] for c in cids], stopwords="en"))
    qids = list(queries)
    idxs, _ = retriever.retrieve(
        bm25s.tokenize([queries[q] for q in qids], stopwords="en"), k=min(k, len(cids))
    )
    ranked = {q: [cids[i] for i in idxs[row]] for row, q in enumerate(qids)}
    cache.write_text(json.dumps(ranked), encoding="utf-8")
    return ranked


def load_dataset_objects() -> tuple[dict, dict, dict, dict]:
    """Return (corpus, queries, golds, candidates) with candidates = BM25 top-K."""
    ds = build_slice()
    corpus = ds["corpus"]
    queries = {q["qid"]: q["query"] for q in ds["queries"]}
    golds = {q["qid"]: q["gold"] for q in ds["queries"]}
    candidates = {q: ranked[:TOP_K] for q, ranked in bm25_rankings(corpus, queries).items()}
    return corpus, queries, golds, candidates


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def gold_rank(ranked: list[str], gold: str) -> int | None:
    """1-based rank of the gold id, or None if it isn't in the list."""
    return ranked.index(gold) + 1 if gold in ranked else None


def topn(rankings: dict[str, list[str]], golds: dict[str, str], k: int) -> float:
    """Share of queries whose gold passage lands at rank <= k."""
    n = len(rankings)
    return sum(gold_rank(rankings[q], golds[q]) in range(1, k + 1) for q in rankings) / n


def summarize(name: str, rankings: dict[str, list[str]], golds: dict[str, str]) -> dict:
    return {"name": name, **{f"top{k}": topn(rankings, golds, k) for k in (1, 5, 10)}}


def rerank_by_scores(candidates: dict, scores: dict) -> dict:
    """Sort each shortlist by score, highest first."""
    return {
        q: sorted(candidates[q], key=lambda c: -scores[q][c]) for q in candidates
    }
