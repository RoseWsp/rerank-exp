"""Compare every reranker that has scores on disk and draw the charts.

Reads artifacts/*_scores_*.json (whichever exist) plus the shared BM25
shortlists, computes top-1/5/10 for each, and writes results.json + PNGs.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import ART, gold_rank, load_dataset_objects, summarize, topn

SURFACE, INK, INK2, MUTED = "#f8f8f2", "#34342f", "#34342f", "#7c7c77"
GRID, AXIS, BLUE, GREEN, ORANGE = "#d8d8cf", "#d8d8cf", "#5d76a2", "#6f9b52", "#c98a3b"
PALETTE = [BLUE, GREEN, ORANGE, "#a26f9b", "#b5533f", "#3f8b8b"]

# Chinese labels render with a CJK font when one is present (Windows: YaHei/SimHei)
_rcp = None


def _fonts():
    global _rcp
    if _rcp is None:
        import matplotlib

        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        _rcp = True


def auc_gold(scores: dict, golds: dict) -> float:
    """Exact AUC: P(a random gold pair outscores a random non-gold pair), ties = 0.5."""
    vals = [(s, 1 if c == golds[q] else 0) for q in scores for c, s in scores[q].items()]
    vals.sort(key=lambda t: t[0])
    n = len(vals)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(1 for v in vals if v[1] == 1)
    nneg = n - npos
    pos_rank_sum = sum(ranks[k] for k in range(n) if vals[k][1] == 1)
    return (pos_rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def mean_gold_rank(rankings: dict, golds: dict) -> float:
    ranks = [gold_rank(rankings[q], golds[q]) for q in rankings]
    ranks = [r for r in ranks if r]
    return sum(ranks) / len(ranks) if ranks else float("nan")


def discover_runs() -> list[dict]:
    """Every score file present in artifacts/, as a run dict."""
    runs = []
    for path in sorted(ART.glob("*_scores*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append(data)
    return runs


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def grouped_bars(labels: list[str], runs: dict[str, list[float]], title: str, out: Path,
                 ylabel: str = "share of queries", pct: bool = True) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _fonts()
    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(labels))
    width = 0.8 / max(len(runs), 1)
    for i, (name, vals) in enumerate(runs.items()):
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, color=PALETTE[i % len(PALETTE)], label=name)
        ax.bar_label(
            bars,
            labels=[f"{v * 100:.0f}%" if pct else f"{v:.2f}" for v in vals],
            padding=3, color=INK2, fontsize=8.5,
        )
    ax.set_xticks(x, labels)
    if pct:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    corpus, queries, golds, candidates = load_dataset_objects()
    runs = discover_runs()
    if not runs:
        raise SystemExit("no score files in artifacts/ — run run_qwen.py / run_typesafe.py first")

    results = {"queries": len(queries), "corpus": len(corpus), "runs": []}

    baseline = summarize("BM25 (fast search)", candidates, golds)
    # AUC from ranks: gold at rank r of n beats (n - r) of the n-1 others
    base_auc = sum(
        (len(candidates[q]) - gold_rank(candidates[q], golds[q])) / (len(candidates[q]) - 1)
        for q in queries
    ) / len(queries)
    baseline["auc"] = base_auc
    baseline["mean_gold_rank"] = mean_gold_rank(candidates, golds)
    results["runs"].append(baseline)
    hdr = (
        f"\n{'method':30s} {'top1':>6s} {'top5':>6s} {'top10':>6s} "
        f"{'AUC':>6s} {'meanGT':>7s} {'s/query':>8s} {'cost':>8s}"
    )
    print(hdr)

    def row(name, d):
        print(
            f"{name:30s} {d['top1']*100:5.0f}% {d['top5']*100:5.0f}% "
            f"{d['top10']*100:5.0f}% {d['auc']:6.3f} {d['mean_gold_rank']:7.2f} "
            f"{d.get('seconds_per_query', float('nan')):8.2f} "
            f"{('$' + format(d['cost_usd'], '.4f')) if 'cost_usd' in d else '~$0':>8s}"
        )

    row(baseline["name"], baseline)

    rankings_by_method = {}
    for r in runs:
        scores = r["scores"]
        reranked = {q: sorted(candidates[q], key=lambda c: -scores[q][c]) for q in candidates}
        rankings_by_method[r["method"]] = reranked
        d = summarize(r["method"], reranked, golds)
        d["auc"] = auc_gold(scores, golds)
        d["mean_gold_rank"] = mean_gold_rank(reranked, golds)
        d.update({k: r[k] for k in (
            "pairs", "seconds", "seconds_per_query", "input_tokens",
        ) if k in r})
        if "cost_usd" in r:
            d["cost_usd"] = r["cost_usd"]
        results["runs"].append(d)
        row(r["method"], d)

    # agreement: for each query, does the method put a different passage first?
    names = list(rankings_by_method)
    if len(names) >= 2:
        agree = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                same = sum(
                    rankings_by_method[a][q][0] == rankings_by_method[b][q][0] for q in queries
                )
                agree[f"{a} vs {b}"] = same / len(queries)
        results["top1_agreement"] = agree
        print("\ntop-1 agreement:")
        for k, v in agree.items():
            print(f"  {k:44s} {v*100:.0f}%")

    (ART / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {ART / 'results.json'}")

    # ---- charts ----
    labels = ["Top-1", "Top-5", "Top-10"]
    series = {"BM25 (fast search)": [baseline["top1"], baseline["top5"], baseline["top10"]]}
    for d in results["runs"][1:]:
        series[d["name"]] = [d["top1"], d["top5"], d["top10"]]
    grouped_bars(labels, series, "正确段落落在前 N 的比例（40 条 query / 3565 篇语料）",
                 ART / "chart_accuracy.png", ylabel="命中的 query 占比")

    lat = {d["name"]: [d.get("seconds_per_query", 0.0)] for d in results["runs"][1:]}
    if lat:
        grouped_bars(list(lat), {k: v for k, v in lat.items()},
                     "每条 query 的重排耗时（含模型加载后的稳定速率）",
                     ART / "chart_latency.png", ylabel="秒 / query", pct=False)


if __name__ == "__main__":
    main()
