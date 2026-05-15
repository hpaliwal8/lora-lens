"""Phase 1: build the step-level reasoning verification dataset.

Two sources are merged:
  1. PRM800K — human step-level labels on MATH solutions (real labels, no API cost).
  2. Synthetic — GSM8K solutions with one step corrupted by Claude, with a known
     error type and a natural-language rationale.

Output: train.jsonl / val.jsonl / test.jsonl in `output_dir`. Each record:

    {
      "source": "prm800k" | "synthetic",
      "problem": str,
      "prior_steps": list[str],   # the chain of thought up to (but not including) the candidate
      "candidate_step": str,      # the step being judged
      "label": "correct" | "incorrect",
      "error_type": str | null,   # only populated for synthetic
      "rationale": str | null,    # explanation of why incorrect (synthetic only by default)
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


@dataclass
class Example:
    source: str
    problem: str
    prior_steps: list[str]
    candidate_step: str
    label: str
    error_type: str | None = None
    rationale: str | None = None


# ---------------------------------------------------------------------------
# PRM800K loader
# ---------------------------------------------------------------------------


def load_prm800k(cfg: dict) -> list[Example]:
    """Load PRM800K and flatten into per-step Example records.

    PRM800K stores per-step ratings in {-1, 0, +1}. We collapse to binary using
    `neutral_policy`. The schema of community mirrors varies slightly; this loader
    targets the common {problem, steps:[{text, rating}]} shape and will need a
    small tweak if your chosen mirror differs.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_dataset"], split=cfg["split"])
    max_n = cfg.get("max_examples")
    if max_n is not None:
        ds = ds.select(range(min(max_n, len(ds))))

    neutral_policy = cfg.get("neutral_policy", "drop")
    examples: list[Example] = []

    for row in ds:
        problem = _extract_problem(row)
        steps = _extract_steps(row)
        if problem is None or not steps:
            continue

        prior: list[str] = []
        for step_text, rating in steps:
            label = _rating_to_label(rating, neutral_policy)
            if label is None:
                prior.append(step_text)
                continue
            examples.append(
                Example(
                    source="prm800k",
                    problem=problem,
                    prior_steps=list(prior),
                    candidate_step=step_text,
                    label=label,
                )
            )
            prior.append(step_text)

    return examples


def _extract_problem(row: dict) -> str | None:
    for key in ("problem", "question", "prompt"):
        if key in row and isinstance(row[key], str):
            return row[key]
    return None


def _extract_steps(row: dict) -> list[tuple[str, int]]:
    """Return list of (step_text, rating) tuples. Handles the common shapes."""
    steps_field = row.get("steps") or row.get("label", {}).get("steps")
    if not steps_field:
        return []
    out: list[tuple[str, int]] = []
    for step in steps_field:
        text = step.get("text") or step.get("completion") or step.get("step")
        rating = step.get("rating")
        if rating is None:
            rating = step.get("label")
        if text is None or rating is None:
            continue
        try:
            rating_int = int(rating)
        except (TypeError, ValueError):
            continue
        out.append((str(text), rating_int))
    return out


def _rating_to_label(rating: int, policy: str) -> str | None:
    if rating > 0:
        return "correct"
    if rating < 0:
        return "incorrect"
    if policy == "as_correct":
        return "correct"
    if policy == "as_incorrect":
        return "incorrect"
    return None  # drop


# ---------------------------------------------------------------------------
# Synthetic corruption via Claude
# ---------------------------------------------------------------------------


CORRUPTION_PROMPT = """You will corrupt one step of a correct math solution to create a labeled training example for an error-detection model.

PROBLEM:
{problem}

CORRECT SOLUTION (steps numbered):
{numbered_steps}

ERROR TYPE TO INJECT: {error_type}
  - arithmetic: a calculation mistake (wrong sum, off-by-one, wrong operation applied correctly).
  - logical: an invalid inference — the conclusion does not follow from prior steps even though the arithmetic is fine.
  - premise: introduces a fact not given in the problem, or contradicts a fact that was given.
  - unit: drops, mixes, or wrongly converts units.

Pick ONE step (not the first, not the last if avoidable) and rewrite it so it contains exactly that error type. The corrupted step must be plausible — it should look like the kind of mistake a student might make, not gibberish. Everything else stays the same.

Return STRICT JSON, no prose, no code fences:
{{
  "step_index": <1-based index of the step you corrupted>,
  "corrupted_step": "<the rewritten step>",
  "rationale": "<one sentence explaining what is wrong with the corrupted step>"
}}"""


async def generate_synthetic(cfg: dict) -> list[Example]:
    """Generate corrupted-step examples from GSM8K via Claude."""
    from anthropic import AsyncAnthropic
    from datasets import load_dataset

    client = AsyncAnthropic()
    ds = load_dataset(cfg["source_dataset"], cfg["source_config"], split=cfg["source_split"])

    n_target = cfg["n_examples"]
    error_types = cfg["error_types"]
    if len(ds) < n_target:
        n_target = len(ds)

    rng = random.Random(0)
    indices = rng.sample(range(len(ds)), n_target)
    assignments = [(i, error_types[k % len(error_types)]) for k, i in enumerate(indices)]

    sem = asyncio.Semaphore(cfg["max_concurrent"])
    results: list[Example | None] = [None] * len(assignments)
    pbar = tqdm(total=len(assignments), desc="synthetic")

    async def one(slot: int, row_idx: int, error_type: str) -> None:
        async with sem:
            row = ds[row_idx]
            problem = row["question"]
            steps = _split_gsm8k_answer(row["answer"])
            if len(steps) < 3:
                pbar.update(1)
                return
            try:
                parsed = await _call_corruption(client, problem, steps, error_type, cfg)
            except Exception:
                pbar.update(1)
                return
            idx = parsed.get("step_index")
            if not isinstance(idx, int) or not (1 <= idx <= len(steps)):
                pbar.update(1)
                return
            corrupted = parsed.get("corrupted_step")
            rationale = parsed.get("rationale")
            if not isinstance(corrupted, str) or not isinstance(rationale, str):
                pbar.update(1)
                return
            prior = steps[: idx - 1]
            results[slot] = Example(
                source="synthetic",
                problem=problem,
                prior_steps=prior,
                candidate_step=corrupted,
                label="incorrect",
                error_type=error_type,
                rationale=rationale,
            )
            pbar.update(1)

    await asyncio.gather(*[one(s, i, e) for s, (i, e) in enumerate(assignments)])
    pbar.close()
    return [r for r in results if r is not None]


async def _call_corruption(
    client: Any,
    problem: str,
    steps: list[str],
    error_type: str,
    cfg: dict,
) -> dict:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    prompt = CORRUPTION_PROMPT.format(
        problem=problem, numbered_steps=numbered, error_type=error_type
    )
    for attempt in range(cfg["max_retries"]):
        try:
            msg = await client.messages.create(
                model=cfg["generator_model"],
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            return json.loads(_strip_code_fence(text))
        except (json.JSONDecodeError, KeyError, AttributeError):
            if attempt == cfg["max_retries"] - 1:
                raise
            await asyncio.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _strip_code_fence(text: str) -> str:
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def _split_gsm8k_answer(answer: str) -> list[str]:
    """GSM8K answers separate steps with newlines; the final line is '#### <number>'."""
    lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
    return [ln for ln in lines if not ln.startswith("####")]


# ---------------------------------------------------------------------------
# Splitting & I/O
# ---------------------------------------------------------------------------


def stratified_split(
    examples: list[Example], cfg: dict
) -> tuple[list[Example], list[Example], list[Example]]:
    """Stratified split by the tuple of fields in cfg['stratify_by']."""
    fracs = (cfg["train_frac"], cfg["val_frac"], cfg["test_frac"])
    assert abs(sum(fracs) - 1.0) < 1e-6, "split fractions must sum to 1"
    keys = cfg["stratify_by"]
    rng = random.Random(cfg["seed"])

    buckets: dict[tuple, list[Example]] = {}
    for ex in examples:
        k = tuple(getattr(ex, key) for key in keys)
        buckets.setdefault(k, []).append(ex)

    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * fracs[0])
        n_val = int(n * fracs[1])
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a tiny subset to sanity-check the pipeline before a full run.",
    )
    args = parser.parse_args()
    load_dotenv()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        cfg["prm800k"]["max_examples"] = 20
        cfg["synthetic"]["n_examples"] = 10
        cfg["output_dir"] = cfg["output_dir"].rstrip("/") + "_dryrun"

    print(f"[1/4] loading PRM800K ({cfg['prm800k']['hf_dataset']})")
    prm = load_prm800k(cfg["prm800k"])
    print(f"      → {len(prm)} step-level examples")

    synth: list[Example] = []
    if cfg["synthetic"]["enabled"]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing; set it in .env or disable synthetic.")
        print(f"[2/4] generating {cfg['synthetic']['n_examples']} synthetic corruptions")
        synth = asyncio.run(generate_synthetic(cfg["synthetic"]))
        print(f"      → {len(synth)} successful")

    all_examples = prm + synth
    print(f"[3/4] splitting {len(all_examples)} examples")
    train, val, test = stratified_split(all_examples, cfg["split"])

    out = Path(cfg["output_dir"])
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    print(f"[4/4] wrote {len(train)}/{len(val)}/{len(test)} to {out}/")


if __name__ == "__main__":
    main()
