"""Score BM25 shortlists with Qwen3-Reranker (0.6B / 4B).

Uses the official Qwen3-Reranker recipe: wrap the pair in a fixed chat template,
read the last-token logits, and take softmax over the "yes" / "no" tokens.

Usage:
    python run_qwen.py --size 0.6B
    python run_qwen.py --size 4B
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from common import ART, load_dataset_objects

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODELS = {
    "0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "4B": "Qwen/Qwen3-Reranker-4B",
    "8B": "Qwen/Qwen3-Reranker-8B",
}

# The instruction plays the same role as TypeSafe's Noul criteria: it fixes what
# counts as a match, so every candidate is judged against one standard.
INSTRUCTION = (
    "The query is an excerpt from a US federal court opinion with a citation "
    "removed. Retrieve the passage from the cited precedent that establishes the "
    "specific legal proposition the excerpt invokes at that citation point."
)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def yes_no_ids(tokenizer) -> tuple[int, int]:
    true_id = tokenizer("yes", add_special_tokens=False)["input_ids"][-1]
    false_id = tokenizer("no", add_special_tokens=False)["input_ids"][-1]
    return true_id, false_id


@torch.no_grad()
def score_batch(model, tokenizer, texts: list[str], true_id: int, false_id: int, max_len: int) -> list[float]:
    inputs = tokenizer(
        [PREFIX + t + SUFFIX for t in texts],
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=max_len,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    try:
        # only the last position's logits are needed; the full [B, seq, vocab]
        # tensor is what OOMs the 4B model on a 16 GB card
        out = model(**inputs, num_logits_to_keep=1)
    except TypeError:
        out = model(**inputs)
    logits = out.logits[:, -1, :]
    pair = torch.stack([logits[:, false_id], logits[:, true_id]], dim=1)
    # softmax over [no, yes], take P(yes)
    return pair.softmax(dim=1)[:, 1].float().cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=list(MODELS))
    ap.add_argument("--model-path", default=None,
                    help="local directory to load instead of the HuggingFace id")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"],
                    help="bitsandbytes quantization")
    ap.add_argument("--device-map", default="cuda",
                    help="'cuda' or 'auto' (auto offloads layers when the model exceeds VRAM)")
    ap.add_argument("--label", default=None, help="suffix for the output file / method name")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()

    label = args.label or (args.size if args.quant == "none" else f"{args.size}_{args.quant}")
    method = f"Qwen3-Reranker-{args.size}" + ("" if args.quant == "none" else f" ({args.quant})")
    out_path = ART / f"qwen_scores_{label}.json"
    if out_path.exists():
        print(f"already have {out_path}, delete it to re-run")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODELS[args.size]
    if args.model_path:
        model_id = args.model_path
    else:
        local = Path(__file__).resolve().parent / "models" / model_id.split("/")[-1]
        if local.is_dir() and any(local.glob("*.safetensors")):
            model_id = str(local)
    print(f"loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if args.quant == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=args.device_map
        ).eval()
    else:
        from transformers import BitsAndBytesConfig

        cfg = BitsAndBytesConfig(
            load_in_8bit=args.quant == "8bit", load_in_4bit=args.quant == "4bit"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map=args.device_map
        ).eval()
    true_id, false_id = yes_no_ids(tokenizer)

    corpus, queries, golds, candidates = load_dataset_objects()
    pairs = [(q, c) for q in candidates for c in candidates[q]]
    texts = [format_pair(INSTRUCTION, queries[q], corpus[c]) for q, c in pairs]
    print(f"{len(pairs)} pairs across {len(queries)} queries")

    scores: list[float] = []
    t0 = time.time()
    for i in range(0, len(texts), args.batch):
        scores.extend(
            score_batch(model, tokenizer, texts[i : i + args.batch], true_id, false_id, args.max_len)
        )
        done = min(i + args.batch, len(texts))
        if (i // args.batch) % 10 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}  {time.time() - t0:.1f}s", flush=True)
    elapsed = time.time() - t0

    pair_scores = {q: {} for q in candidates}
    for (q, c), s in zip(pairs, scores):
        pair_scores[q][c] = s

    n_tok = sum(len(tokenizer.encode(t)) for t in texts)
    result = {
        "method": method,
        "model_id": model_id,
        "instruction": INSTRUCTION,
        "pairs": len(pairs),
        "seconds": elapsed,
        "seconds_per_query": elapsed / len(queries),
        "input_tokens": n_tok,
        "scores": pair_scores,
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out_path}  ({elapsed:.1f}s, {elapsed / len(queries):.2f}s/query)")


if __name__ == "__main__":
    main()
