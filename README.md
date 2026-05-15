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
2. **Baseline** — zero-shot DeepSeek-R1-Distill on the test set. Step-level F1 + LLM-as-judge explanation score.
3. **Fine-tuning** — train LoRA / DoRA / rsLoRA adapters with PEFT.
4. **Evaluation** — step F1, confusion matrix by error type, pass@1 lift on solution reranking.
5. **Interpretability** — SVD top directions → vocab projection, `‖ΔW‖` per layer, activation diffing, tuned lens, layer ablation. Compare across adapter methods.
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

## Layout

```
src/lora_lens/
  data_prep.py    # Phase 1
  train.py        # Phase 3 (not yet)
  evaluate.py     # Phases 2 & 4 (not yet)
  interpret.py    # Phase 5 (not yet)
configs/          # one yaml per run
notebooks/        # plots, exploratory analysis
data/             # gitignored
outputs/          # gitignored adapter checkpoints
```

## Reading list

The framing of this project leans on:

- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME), 2022.
- Hase et al., *Does Localization Inform Editing?*, 2023.
- Bricken et al., *Towards Monosemanticity*, Anthropic 2023.
- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*, 2023.
- Biderman et al., *LoRA Learns Less and Forgets Less*, 2024.
