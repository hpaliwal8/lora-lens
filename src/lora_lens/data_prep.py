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
from collections import Counter
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
    """Load PRM800K from the Birchlabs stepwise-critic mirror.

    The mirror is pre-flattened: each row is already one step-level example with
    fields {instruction, responses, next_response, rating}. We just rename and
    map the rating to a binary label.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_dataset"], split=cfg["split"])
    max_n = cfg.get("max_examples")
    if max_n is not None:
        ds = ds.select(range(min(max_n, len(ds))))

    neutral_policy = cfg.get("neutral_policy", "drop")
    examples: list[Example] = []

    for row in ds:
        problem = row.get("instruction")
        candidate = row.get("next_response")
        prior = row.get("responses") or []
        rating = row.get("rating")
        if not isinstance(problem, str) or not problem.strip():
            continue
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        if rating is None:
            continue
        try:
            rating_int = int(rating)
        except (TypeError, ValueError):
            continue
        label = _rating_to_label(rating_int, neutral_policy)
        if label is None:
            continue
        examples.append(
            Example(
                source="prm800k",
                problem=problem,
                prior_steps=list(prior),
                candidate_step=candidate,
                label=label,
            )
        )

    return examples


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
    failures: Counter[str] = Counter()
    pbar = tqdm(total=len(assignments), desc="synthetic")

    async def one(slot: int, row_idx: int, error_type: str) -> None:
        async with sem:
            row = ds[row_idx]
            problem = row["question"]
            steps = _split_gsm8k_answer(row["answer"])
            if len(steps) < 3:
                failures["too_few_steps"] += 1
                pbar.update(1)
                return
            try:
                parsed = await _call_corruption(client, problem, steps, error_type, cfg)
            except json.JSONDecodeError:
                failures["json_decode"] += 1
                pbar.update(1)
                return
            except Exception as e:
                failures[f"api_{type(e).__name__}"] += 1
                pbar.update(1)
                return
            idx = parsed.get("step_index")
            if not isinstance(idx, int) or not (1 <= idx <= len(steps)):
                failures["bad_step_index"] += 1
                pbar.update(1)
                return
            corrupted = parsed.get("corrupted_step")
            rationale = parsed.get("rationale")
            if not isinstance(corrupted, str) or not isinstance(rationale, str):
                failures["bad_fields"] += 1
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
    if failures:
        print("      failure breakdown:")
        for reason, count in failures.most_common():
            print(f"        {reason}: {count}")
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
    import anthropic

    last_exc: Exception | None = None
    for attempt in range(cfg["max_retries"]):
        try:
            msg = await client.messages.create(
                model=cfg["generator_model"],
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            return json.loads(_strip_code_fence(text))
        except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
            last_exc = e
            await asyncio.sleep(2**attempt)
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as e:
            last_exc = e
            await asyncio.sleep(2 ** (attempt + 1))
    assert last_exc is not None
    raise last_exc


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


def group_aware_split(
    examples: list[Example], cfg: dict
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split so a single problem never appears in more than one split.

    PRM800K is row-per-step: many rows share the same `problem`. A naive row-level
    split leaks problem context across train/val/test (97%+ of problems land in
    multiple splits, inflating eval). We instead assign each unique PRM problem
    to exactly one split and put all of its step-level rows there.

    Synthetic examples have one row per source GSM8K problem (each problem is
    sampled once), so a row-level split stratified by error_type is already
    group-safe and gives balanced error-type coverage in each split.
    """
    fracs = (cfg["train_frac"], cfg["val_frac"], cfg["test_frac"])
    assert abs(sum(fracs) - 1.0) < 1e-6, "split fractions must sum to 1"
    rng = random.Random(cfg["seed"])

    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []

    prm = [e for e in examples if e.source == "prm800k"]
    syn = [e for e in examples if e.source == "synthetic"]

    by_problem: dict[str, list[Example]] = {}
    for ex in prm:
        by_problem.setdefault(ex.problem, []).append(ex)
    problems = list(by_problem.keys())
    rng.shuffle(problems)
    n = len(problems)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    for p in problems[:n_train]:
        train.extend(by_problem[p])
    for p in problems[n_train : n_train + n_val]:
        val.extend(by_problem[p])
    for p in problems[n_train + n_val :]:
        test.extend(by_problem[p])

    syn_buckets: dict[str | None, list[Example]] = {}
    for ex in syn:
        syn_buckets.setdefault(ex.error_type, []).append(ex)
    for bucket in syn_buckets.values():
        rng.shuffle(bucket)
        m = len(bucket)
        m_train = int(m * fracs[0])
        m_val = int(m * fracs[1])
        train.extend(bucket[:m_train])
        val.extend(bucket[m_train : m_train + m_val])
        test.extend(bucket[m_train + m_val :])

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
    parser.add_argument(
        "--reshuffle-only",
        action="store_true",
        help="Re-split existing JSONL outputs in place. No HF download, no API calls.",
    )
    args = parser.parse_args()
    load_dotenv()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        cfg["prm800k"]["max_examples"] = 20
        cfg["synthetic"]["n_examples"] = 10
        cfg["output_dir"] = cfg["output_dir"].rstrip("/") + "_dryrun"

    out = Path(cfg["output_dir"])

    if args.reshuffle_only:
        all_examples: list[Example] = []
        skipped = 0
        for name in ("train", "val", "test"):
            with (out / f"{name}.jsonl").open() as f:
                for line in f:
                    d = json.loads(line)
                    p = d.get("problem") or ""
                    c = d.get("candidate_step") or ""
                    if not p.strip() or not c.strip():
                        skipped += 1
                        continue
                    all_examples.append(Example(**d))
        print(f"[1/2] loaded {len(all_examples)} examples from {out}/  (dropped {skipped} empty)")
        train, val, test = group_aware_split(all_examples, cfg["split"])
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", val)
        write_jsonl(out / "test.jsonl", test)
        print(f"[2/2] wrote {len(train)}/{len(val)}/{len(test)} to {out}/")
        return

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
    train, val, test = group_aware_split(all_examples, cfg["split"])

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    print(f"[4/4] wrote {len(train)}/{len(val)}/{len(test)} to {out}/")


if __name__ == "__main__":
    main()
