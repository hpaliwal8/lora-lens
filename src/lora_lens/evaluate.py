"""Phase 2 + 4 evaluation.

Two subcommands, deliberately separated so re-scoring doesn't re-pay GPU time:

  predict  — load a model, run on test rows, save predictions JSONL.
  score    — load predictions, compute step-level F1 + per-error-type breakdown,
             and call an LLM judge on a stratified subsample for explanation quality.

Phase 2 runs both with the base model. Phase 4 reruns with each trained adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from lora_lens.prompts import build_chat_messages, parse_response


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _pick_device_and_dtype(cfg_dtype: str) -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(cfg_dtype, torch.bfloat16)
    # MPS bf16 has rough edges in older torch; fall back to fp16 there.
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float16
    return device, dtype


def _load_model(cfg: dict):
    """Load the base model + tokenizer with sensible defaults per platform."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = cfg["hf_id"]
    device, dtype = _pick_device_and_dtype(cfg.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device == "cuda" and cfg.get("quantize_4bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    if device != "cuda" or "device_map" not in load_kwargs:
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _build_prompt_text(tokenizer, ex: dict) -> str:
    msgs = build_chat_messages(ex["problem"], ex["prior_steps"], ex["candidate_step"])
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@dataclass
class Prediction:
    idx: int
    source: str
    problem: str
    prior_steps: list[str]
    candidate_step: str
    label_gt: str
    error_type: str | None
    rationale_gt: str | None
    label_pred: str | None
    explanation_pred: str | None
    raw_completion: str


def cmd_predict(cfg: dict, args: argparse.Namespace) -> None:
    import torch

    test_path = Path(cfg["predict"]["test_path"])
    out_path = Path(args.output or cfg["predict"]["output_path"])
    batch_size = args.batch_size or cfg["predict"]["batch_size"]
    max_new_tokens = cfg["model"]["max_new_tokens"]
    limit = args.limit if args.limit is not None else cfg["predict"].get("limit")

    test = load_jsonl(test_path)
    if limit is not None and limit < len(test):
        rng = random.Random(cfg["predict"].get("subsample_seed", 0))
        idxs = sorted(rng.sample(range(len(test)), limit))
        test = [test[i] for i in idxs]
        print(f"  subsampled to {len(test)} rows (seed={cfg['predict'].get('subsample_seed', 0)})")
    else:
        print(f"  evaluating on full test set ({len(test)} rows)")

    print(f"  loading model: {cfg['model']['hf_id']}")
    model, tokenizer, device = _load_model(cfg["model"])
    print(f"  device={device}, dtype={next(model.parameters()).dtype}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream predictions out so a long run isn't lost on a crash.
    with out_path.open("w") as fout:
        for start in tqdm(range(0, len(test), batch_size), desc="predict"):
            batch = test[start : start + batch_size]
            prompts = [_build_prompt_text(tokenizer, ex) for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=cfg["model"].get("max_input_tokens", 2048)).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completions = tokenizer.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for ex, completion in zip(batch, completions):
                parsed = parse_response(completion)
                pred = Prediction(
                    idx=ex.get("__idx__", -1),
                    source=ex["source"],
                    problem=ex["problem"],
                    prior_steps=ex["prior_steps"],
                    candidate_step=ex["candidate_step"],
                    label_gt=ex["label"],
                    error_type=ex.get("error_type"),
                    rationale_gt=ex.get("rationale"),
                    label_pred=parsed.label,
                    explanation_pred=parsed.explanation,
                    raw_completion=parsed.raw,
                )
                fout.write(json.dumps(asdict(pred)) + "\n")
    print(f"  wrote predictions → {out_path}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _confusion(preds: list[dict]) -> dict[str, int]:
    c = Counter()
    for p in preds:
        gt, pr = p["label_gt"], p["label_pred"]
        if pr is None:
            c["unparseable"] += 1
            continue
        c[f"{gt}->{pr}"] += 1
    return dict(c)


def _f1(preds: list[dict], positive: str = "incorrect") -> dict[str, float]:
    tp = fp = fn = tn = 0
    parseable = 0
    for p in preds:
        if p["label_pred"] is None:
            continue
        parseable += 1
        gt = p["label_gt"]
        pr = p["label_pred"]
        if gt == positive and pr == positive:
            tp += 1
        elif gt != positive and pr == positive:
            fp += 1
        elif gt == positive and pr != positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / parseable if parseable else 0.0
    return {
        "n_total": len(preds),
        "n_parseable": parseable,
        "parseable_rate": parseable / len(preds) if preds else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _per_error_type_f1(preds: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        if p["source"] != "synthetic":
            continue
        et = p.get("error_type") or "unknown"
        by_type[et].append(p)
    for et, rows in by_type.items():
        # All synthetic rows are gt=incorrect, so this is just recall on those.
        correct = sum(1 for r in rows if r["label_pred"] == "incorrect")
        parseable = sum(1 for r in rows if r["label_pred"] is not None)
        out[et] = {
            "n": len(rows),
            "n_parseable": parseable,
            "recall_incorrect": correct / parseable if parseable else 0.0,
        }
    return out


JUDGE_PROMPT = """You are evaluating a math step-verification model's *explanation* of its verdict.

PROBLEM:
{problem}

PRIOR STEPS:
{prior_block}

CANDIDATE STEP (the model judged this one):
{candidate_step}

GROUND-TRUTH VERDICT: {gt_label}
{rationale_block}MODEL'S EXPLANATION: {explanation}

Rate the model's explanation on a 1–5 scale:
  5: correctly identifies the key issue (incorrect case) or affirms correctness with valid reasoning (correct case)
  4: mostly correct, minor errors or imprecision
  3: partially correct, misses something important
  2: incorrect or misleading reasoning despite reaching the right verdict
  1: nonsensical, empty, or wholly wrong

Return STRICT JSON, no prose, no fences:
{{"score": <1-5 integer>, "reason": "<one short sentence>"}}"""


async def _judge_one(
    client: Any, model: str, p: dict, max_retries: int
) -> dict | None:
    import anthropic

    prior_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(p["prior_steps"])) or "(none)"
    rationale_block = (
        f"REFERENCE RATIONALE: {p['rationale_gt']}\n" if p.get("rationale_gt") else ""
    )
    prompt = JUDGE_PROMPT.format(
        problem=p["problem"],
        prior_block=prior_block,
        candidate_step=p["candidate_step"],
        gt_label=p["label_gt"],
        rationale_block=rationale_block,
        explanation=p.get("explanation_pred") or "(empty)",
    )
    for attempt in range(max_retries):
        try:
            msg = await client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            return json.loads(text)
        except (json.JSONDecodeError, KeyError, AttributeError, IndexError):
            await asyncio.sleep(2**attempt)
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ):
            await asyncio.sleep(2 ** (attempt + 1))
    return None


async def _judge_all(preds: list[dict], cfg: dict) -> list[dict]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(cfg["max_concurrent"])
    model = cfg["working_judge_model"]
    max_retries = cfg["max_retries"]
    results: list[dict | None] = [None] * len(preds)
    pbar = tqdm(total=len(preds), desc=f"judge[{model}]")

    async def one(slot: int, p: dict) -> None:
        async with sem:
            results[slot] = await _judge_one(client, model, p, max_retries)
            pbar.update(1)

    await asyncio.gather(*[one(i, p) for i, p in enumerate(preds)])
    pbar.close()
    out: list[dict] = []
    for p, r in zip(preds, results):
        out.append({**p, "judge_score": r.get("score") if r else None,
                    "judge_reason": r.get("reason") if r else None})
    return out


def _stratified_judge_subsample(
    preds: list[dict], n: int, seed: int
) -> list[dict]:
    """Sample n predictions balanced across (source, label, error_type).

    Per-bucket pass takes min(n // n_buckets, bucket_size) from each. Any unused
    budget (when small buckets cap out) is filled from leftover items so the
    final count is as close to n as the available data allows.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in preds:
        if p["label_pred"] is None:
            continue  # unparseable predictions have nothing to judge
        key = (p["source"], p["label_gt"], p.get("error_type"))
        buckets[key].append(p)
    rng = random.Random(seed)
    keys = sorted(buckets.keys(), key=lambda k: tuple(str(x) for x in k))
    per_bucket = max(1, n // max(1, len(keys)))

    out: list[dict] = []
    leftover: list[dict] = []
    for k in keys:
        bucket = sorted(buckets[k], key=lambda p: (p["problem"], p["candidate_step"]))
        rng.shuffle(bucket)
        out.extend(bucket[:per_bucket])
        leftover.extend(bucket[per_bucket:])
    if len(out) < n:
        rng.shuffle(leftover)
        out.extend(leftover[: n - len(out)])
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
    return out


def cmd_score(cfg: dict, args: argparse.Namespace) -> None:
    preds_path = Path(args.predictions or cfg["score"]["predictions_path"])
    metrics_path = Path(args.metrics or cfg["score"]["metrics_path"])
    judge_n = args.judge_n if args.judge_n is not None else cfg["score"]["judge_subsample_size"]
    judge_seed = cfg["score"].get("judge_subsample_seed", 0)

    preds = load_jsonl(preds_path)
    print(f"  loaded {len(preds)} predictions from {preds_path}")

    metrics: dict[str, Any] = {
        "n_predictions": len(preds),
        "overall": _f1(preds),
        "confusion": _confusion(preds),
        "per_error_type": _per_error_type_f1(preds),
    }

    if args.skip_judge:
        print("  --skip-judge passed; no LLM judge calls")
    else:
        load_dotenv()
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing; set it in .env or pass --skip-judge.")
        sub = _stratified_judge_subsample(preds, judge_n, judge_seed)
        print(f"  judging {len(sub)} stratified subsample with {cfg['judging']['working_judge_model']}")
        judged = asyncio.run(_judge_all(sub, {**cfg["judging"]}))
        scores = [r["judge_score"] for r in judged if r["judge_score"] is not None]
        metrics["judge"] = {
            "n_judged": len(sub),
            "n_scored": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "score_distribution": dict(Counter(scores)),
            "judge_model": cfg["judging"]["working_judge_model"],
        }
        # Also save the judged subsample for the agreement check / human spot-check later.
        judged_path = metrics_path.parent / "judged_subsample.jsonl"
        write_jsonl(judged_path, judged)
        print(f"  wrote judged subsample → {judged_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  wrote metrics → {metrics_path}")
    print(json.dumps(metrics["overall"], indent=2))


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--output", type=Path, default=None)
    p_pred.add_argument("--limit", type=int, default=None,
                        help="Subsample N test rows; useful for shakeout. Default = full test.")
    p_pred.add_argument("--batch-size", type=int, default=None)

    p_score = sub.add_parser("score")
    p_score.add_argument("--predictions", type=Path, default=None)
    p_score.add_argument("--metrics", type=Path, default=None)
    p_score.add_argument("--judge-n", type=int, default=None)
    p_score.add_argument("--skip-judge", action="store_true")

    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.cmd == "predict":
        cmd_predict(cfg, args)
    elif args.cmd == "score":
        cmd_score(cfg, args)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
