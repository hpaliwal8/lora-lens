# lora-lens

Probing what a LoRA adapter actually learns when fine-tuned for step-level reasoning verification.

Most LoRA projects stop at "the fine-tuned model scores higher." This one fine-tunes a small reasoning model on step-level error detection, then opens up the adapter and asks: *where in the network did the new capability land, and what does it look like?*

## Pitch

- **Task.** Given a math problem and a partial chain-of-thought, predict whether the next step is correct and explain why.
- **Model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- **Data.** PRM800K (human step labels on MATH) + synthetic step corruptions on GSM8K with labeled error types.
- **Method.** Train three adapters on identical data: vanilla LoRA, DoRA, rsLoRA.
- **Headline contribution.** Use lens-style techniques — SVD of the delta weights, layer-wise update magnitudes, activation diffing, tuned lens comparison, layer ablation — to characterize *what* each adapter learned and *where* it lives. Compare across the three methods.

Framing throughout is **probing, not explaining**. Claims are calibrated to what the techniques can actually support (see Hase et al. 2023 on the gap between localization and editing).

## Phases

1. **Data prep** — load PRM800K, generate synthetic GSM8K corruptions via Claude, split into train/val/test.
2. **Baseline** — zero-shot DeepSeek-R1-Distill on the test set. Step-level F1 + LLM-as-judge explanation score (Sonnet 4.6 during dev, Opus 4.7 for the final number — see Judging below).
3. **Fine-tuning** — train LoRA / DoRA / rsLoRA adapters with PEFT.
4. **Evaluation** — step F1, confusion matrix by error type, pass@1 lift on solution reranking. Same judge protocol as Phase 2.
5. **Interpretability** — SVD top directions → vocab projection, `‖ΔW‖` per layer, activation diffing, tuned lens, layer ablation, direct logit attribution. Compare across adapter methods.
6. **Write-up** — README results, plots, honest framing.

## Setup

```bash
uv sync
cp .env.example .env  # add ANTHROPIC_API_KEY and HF_TOKEN
```

## Phase 1 — building the dataset

Dry-run with a tiny sample first to sanity-check synthetic corruption quality before paying for the full generation:

```bash
uv run python -m lora_lens.data_prep --config configs/data.yaml --dry-run
```

Full run:

```bash
uv run python -m lora_lens.data_prep --config configs/data.yaml
```

Output lands in `data/processed/{train,val,test}.jsonl`.

## Evaluation strategy

`evaluate.py` is split into two subcommands so re-scoring doesn't re-pay GPU time:

```bash
uv run python -m lora_lens.evaluate --config configs/eval.yaml predict
uv run python -m lora_lens.evaluate --config configs/eval.yaml score
```

- **Step-level F1, accuracy, per-error-type recall** are reported on the **full test set** (~108k rows). Cheap — just GPU time, no API calls.
- **LLM-as-judge explanation scoring** runs on a **stratified 2k subsample** of the predictions (balanced across `(source, label, error_type)`). Both working and final judges score the same subsample so inter-judge agreement is well-defined.

For a Mac shakeout before GPU access, pass `--limit 50` to `predict` to validate the prompt + parser on a tiny subset.

## Judging

We use a hybrid protocol to balance cost and credibility:

- **Working judge — Claude Sonnet 4.6.** Used for all dev iteration: baseline runs, training checkpoint evals, quick comparisons. Cheap enough to run dozens of times during the project.
- **Final judge — Claude Opus 4.7.** Used once per finished model (base + each trained adapter) on the held-out test subsample. The numbers reported in the README come from this judge.
- **Agreement check.** Both judges score a shared ~100-example subset; we report inter-judge agreement (Cohen's κ). High agreement means the cheap working judge's numbers were trustworthy.
- **Human spot-check.** ~30 random final-judge labels are reviewed manually before reporting. Catches systematic judge failures that no metric will.

Configured in [configs/eval.yaml](configs/eval.yaml). Reasoning for the split: judging chain-of-thought correctness is itself a reasoning task, so the final-number judge benefits from a stronger model — but paying 5× per pass during iteration is wasteful when relative ordering, not absolute scores, is what matters.

## Layout

```
src/lora_lens/
  data_prep.py    # Phase 1
  prompts.py      # shared step-verification prompt + parser
  evaluate.py     # Phases 2 & 4
  train.py        # Phase 3 (not yet)
  interpret.py    # Phase 5 (not yet)
configs/          # one yaml per run
notebooks/        # plots, exploratory analysis
data/             # gitignored
outputs/          # gitignored adapter checkpoints + predictions
```

## Reading list

The framing of this project leans on:

- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME), 2022.
- Hase et al., *Does Localization Inform Editing?*, 2023.
- Bricken et al., *Towards Monosemanticity*, Anthropic 2023.
- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*, 2023.
- Biderman et al., *LoRA Learns Less and Forgets Less*, 2024.
