"""Score the same BM25 shortlists with TypeSafe (jev-1.12).

This is the LLM-as-reranker side: one yes/no Noul per (query, candidate) pair,
exactly as in the TypeSafe "Re-ranking" cookbook.

Requires TYPESAFE_API_KEY in the environment. Results are cached to disk, so a
re-run costs nothing.

Usage:
    set TYPESAFE_API_KEY=...        (Windows cmd)
    $env:TYPESAFE_API_KEY="..."     (PowerShell)
    python run_typesafe.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ART, load_dataset_objects

MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
PRICE = (0.042, 0.00)  # $ per 1M tokens (input, output), jev-1.12 as of 2026-08
CACHE = ART / "typesafe_cache.json"


def load_dotenv(path) -> None:
    """Minimal .env loader so the key never has to be pasted into a shell command."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _key(model: str, query: str, candidate: str, question_json: str) -> str:
    h = hashlib.sha1()
    for part in (model, query, candidate, question_json):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    out_path = ART / "typesafe_scores.json"
    if out_path.exists():
        print(f"already have {out_path}, delete it to re-run")
        return

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit(
            "TYPESAFE_API_KEY is not set. Provide it, or run only the Qwen side first."
        )

    from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

    client = TypeSafeClient(
        api_key=api_key,
        base_url=os.environ.get("TYPESAFE_ENDPOINT"),
        timeout=120.0,
    )

    is_cited_source = Noul(
        instructions=(
            "The query excerpt comes from a US federal court opinion and was written "
            "immediately around a citation to a precedent; the citation itself has been "
            "removed. Could the candidate passage be from that cited precedent — does it "
            "establish the specific legal proposition the query excerpt invokes at its "
            "citation point?"
        ),
        criteria=NoulCriteria(
            true=(
                "The candidate passage states or establishes the specific rule, standard, "
                "holding, or fact pattern that the query excerpt attributes to its removed "
                "citation."
            ),
            false=(
                "The candidate passage is merely on a similar topic or doctrine; it does not "
                "supply the specific proposition the query excerpt relies on."
            ),
        ),
    )
    question_json = is_cited_source.model_dump_json(exclude_none=True)

    corpus, queries, golds, candidates = load_dataset_objects()
    pairs = [(q, c) for q in candidates for c in candidates[q]]
    print(f"{len(pairs)} pairs across {len(queries)} queries")

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    def score(q: str, c: str) -> dict:
        k = _key(MODEL, queries[q], corpus[c], question_json)
        if k in cache:
            return cache[k]
        response = client.system_one(
            state={"query_excerpt": queries[q], "candidate_passage": corpus[c]},
            questions={"is_cited_source": json.loads(question_json)},
            model=MODEL,
        )
        rec = {
            "noul": response.answers["is_cited_source"].noul,
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
        }
        cache[k] = rec
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda p: score(*p), pairs))
    elapsed = time.time() - t0

    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    pair_scores = {q: {} for q in candidates}
    for (q, c), res in zip(pairs, results):
        pair_scores[q][c] = res["noul"]

    input_tokens = sum(r["input_tokens"] for r in results)
    output_tokens = sum(r["output_tokens"] for r in results)
    cost = input_tokens / 1_000_000 * PRICE[0] + output_tokens / 1_000_000 * PRICE[1]

    result = {
        "method": f"TypeSafe {MODEL}",
        "model_id": MODEL,
        "pairs": len(pairs),
        "seconds": elapsed,
        "seconds_per_query": elapsed / len(queries),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "scores": pair_scores,
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")
    print(
        f"wrote {out_path}  ({len(pairs)} calls, {input_tokens:,} in / "
        f"{output_tokens:,} out tokens, ${cost:.4f}, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
