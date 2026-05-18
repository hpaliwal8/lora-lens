"""Prompt template + response parser for step-level verification.

Shared by Phase 2 (zero-shot baseline) and Phase 3 (LoRA fine-tuning) so that
the input/output format the model is graded on matches what it was trained on.

The expected model output format is two trailing lines:
    Verdict: <correct|incorrect>
    Explanation: <one short sentence>

R1-Distill reasoning models will typically produce a <think>...</think> block
before the verdict; the parser strips that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are an expert math tutor verifying each step of a student's solution.\n\n"
    "Given a problem and the prior steps, judge whether the NEXT STEP is "
    "mathematically correct given what came before.\n\n"
    "End your response with exactly two lines:\n"
    "Verdict: correct\n"
    "Explanation: <one short sentence justifying the verdict>\n\n"
    "(Use `Verdict: incorrect` if the next step contains an error.)"
)

USER_TEMPLATE = (
    "Problem:\n{problem}\n\n"
    "Prior steps so far:\n{prior_block}\n\n"
    "Next step (judge this one):\n{candidate_step}"
)


def build_chat_messages(
    problem: str, prior_steps: list[str], candidate_step: str
) -> list[dict]:
    if not prior_steps:
        prior_block = "(none — this is the first step)"
    else:
        prior_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(prior_steps))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                problem=problem,
                prior_block=prior_block,
                candidate_step=candidate_step,
            ),
        },
    ]


@dataclass
class ParsedPrediction:
    label: str | None
    explanation: str | None
    raw: str


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_VERDICT_RE = re.compile(r"Verdict:\s*(correct|incorrect)\b", re.IGNORECASE)
_EXPL_RE = re.compile(r"Explanation:\s*(.+?)\s*\Z", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> ParsedPrediction:
    cleaned = _THINK_RE.sub("", text).strip()
    label_match = _VERDICT_RE.search(cleaned)
    expl_match = _EXPL_RE.search(cleaned)
    return ParsedPrediction(
        label=label_match.group(1).lower() if label_match else None,
        explanation=expl_match.group(1).strip() if expl_match else None,
        raw=cleaned,
    )
