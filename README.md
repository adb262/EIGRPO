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


## Architecture Decisions

**GSM8K is single-turn, no tool use.** GSM8K is a math reasoning benchmark
where the model should solve problems via chain-of-thought, not tool calls.
The reward function (`eigrpo.environments.gsm8k_reward`) uses strict
`<answer>...</answer>` XML formatting with binary 1/0 rewards.

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
