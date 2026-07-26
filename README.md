# EIGRPO

Effective Information Gain for Sample Efficient Group Policy Optimization.

## Vanilla GRPO reward-conditioning experiment

This checkout also contains a two-arm GSM8K experiment that does **not** use
EIG selection or supervised fine-tuning:

1. Regular GRPO rewards a rollout when its answer is correct.
2. Conditioned GRPO flips a fair coin once per sampled training turn:
   - `<CORRECT_ANSWER>` receives reward 1 for a correct answer.
   - `<PLAUSIBLE_INCORRECT_ANSWER>` receives reward 1 for an incorrect answer.

The selected condition is part of the prompt and is shared by every rollout
in that turn's GRPO group. Validation always supplies `<CORRECT_ANSWER>`.
Conditioned runs log raw `task_accuracy` separately from the possibly inverted
`agreement_reward`.

Both variants call verl's unmodified `run_ppo` and default
`RayPPOTrainer`. There is no SFT stage and no `EIGRPOTrainer`.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Running the comparison

```bash
python3 experiments/scripts/run_grpo_baseline.py \
    --variant regular \
    --model Qwen/Qwen2.5-0.5B-Instruct

python3 experiments/scripts/run_grpo_baseline.py \
    --variant conditioned \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

Or run both arms in separate Python processes, from the same initial model:

```bash
./experiments/scripts/run_grpo_comparison.sh Qwen/Qwen2.5-0.5B-Instruct
```

Any trailing arguments are verl dotlist overrides:

```bash
./experiments/scripts/run_grpo_comparison.sh \
    Qwen/Qwen2.5-0.5B-Instruct \
    trainer.total_epochs=1 \
    actor_rollout_ref.rollout.n=8
```

## Tests

```bash
pytest
```

The two runs share optimizer, rollout, seed, validation, and model settings.
Their checkpoints go to separate directories and their Weights & Biases runs
share the `grpo-reward-conditioning` project.

## RC-GRPO baseline

The repository also includes a GSM8K adaptation of
[RC-GRPO](https://arxiv.org/abs/2602.03025). Unlike the no-SFT experiment
above, this paper baseline has two stages:

1. **RC-SFT** collects exploration rollouts, pairs each verified failure with
   the corresponding expert GSM8K solution, and trains a 50/50 mixture labeled
   with `<|high_reward|>` and `<|low_reward|>`.
2. **RC-GRPO** samples one reward goal independently for each member of the
   16-rollout group. High-reward (GOOD) trajectories target a correct answer;
   low-reward (BAD) trajectories target an incorrect answer that must still be
   well formed and plausible. Validation always uses the high-reward goal.

RC-GRPO uses one gated composite reward for both paths:

```text
0.25 * format_score + 0.25 * reasonability_score + 0.5 * target_match_score
```

All three components are in `[0, 1]`. A malformed response is hard-gated to
`-1`; a format-valid response below the configured reasonability threshold
(`0.25` by default) is hard-gated to `0`. `target_match_score` is correctness
for HIGH/GOOD and inverted correctness for LOW/BAD. The GPT-4.1-mini
reasonability judge receives the question and candidate response, but never the
ground-truth answer, and is instructed not to evaluate factual or mathematical
correctness. The hard format penalty and normalized four-dimension quality
rubric are adapted from the
[PNS reward](https://arxiv.org/abs/2602.03516).

Run RC-SFT on two GPUs (offline failure collection uses one GPU by default):

```bash
python3 experiments/scripts/run_rc_sft.py \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

The command writes a directly loadable Hugging Face model under
`checkpoints/rc_sft/global_step_*/huggingface`. RC-GRPO automatically selects the
latest one:

```bash
python3 experiments/scripts/run_rc_grpo.py
```

Set `OPENAI_API_KEY` before RC-GRPO so the reasonability judge can call the
OpenAI Responses API. Independent rollout judgments run with bounded
concurrency (`16` by default); adjust
`reward_model.reward_kwargs.judge_max_concurrency` to fit the account's rate
limits.

Prepared RC-SFT data is reused on subsequent runs. Use `--force-data-prep` to
regenerate it, or `--skip-data-prep` when supplying existing
`data/rc_sft/{train,test}.parquet` files. Both commands accept trailing Hydra
dotlist overrides.


## Architecture Decisions

**GSM8K is single-turn, no tool use.** GSM8K is a math reasoning benchmark
where the model should solve problems via chain-of-thought, not tool calls.
The task verifier (`eigrpo.environments.gsm8k_reward`) uses strict
`<answer>...</answer>` XML formatting with binary 1/0 accuracy. RC-GRPO wraps
that verifier with the gated format/reasonability/target-match reward described
above.

**sglang remains the rollout backend.** verl's tool-call support (multi-turn
agentic RL) only works with sglang, not vLLM. While GSM8K doesn't need
tools, sglang is kept as the default so that future tool-use environments
work without config changes.

**Log probs as a proxy for trajectory information content.** verl does not
expose hidden states through its rollout interface. Log probs—already
available after the actor forward pass—serve as a lightweight signal for
trajectory-level diversity in the EIG selector, avoiding modifications to
verl internals.

**NCCL for all distributed communication.** Gradient all-reduce (FSDP),
rollout weight sync, and sglang tensor-parallel inference all use NCCL
(or HCCL on Huawei NPUs).

**Updates happen every batch.** Each iteration of the dataloader loop is one
global step: generate rollouts, compute rewards, compute advantages, update
actor. No step-skipping; total steps = `len(dataloader) * total_epochs`.

## Kill processes
Sometimes things don't shut down properly after a run.
`kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}') $(nvidia-smi | awk '$5 ~ /^[0-9]+$/ {print $5}') 2>/dev/null || true`
