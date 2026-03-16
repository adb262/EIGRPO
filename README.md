# EIGRPO

Effective Information Gain for Sample Efficient Group Policy Optimization.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Running experiments

```bash
python experiments/scripts/run_grpo_baseline.py --config configs/base.yaml
```

Override any config value with dotlist syntax:

```bash
python experiments/scripts/run_grpo_baseline.py \
    eigrpo.selector.name=eig \
    eigrpo.selector.n_effective_samples=4
```

## Tests

```bash
pytest
```


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