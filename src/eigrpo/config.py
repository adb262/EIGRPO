"""Typed configuration for the EIGRPO trainer and environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectorConfig:
    """Controls trajectory selection (filtering) before the policy update.

    Attributes:
        name: Selector strategy. Built-in options: ``"uniform"``, ``"eig"``.
            You can also pass a fully-qualified class path.
        n_effective_samples: Number of rollouts to keep after selection.
            ``None`` disables selection (all rollouts are used).
    """

    name: str = "uniform"
    n_effective_samples: int | None = None


@dataclass
class DiversityConfig:
    """Diversity and diagnostic logging toggles.

    Attributes:
        log_gradient_norms: Reserved for future use (gradient-norm logging).
        log_trajectory_similarity: When ``True``, compute and log the
            unique-trajectory ratio per step (can be expensive).
    """

    log_gradient_norms: bool = False
    log_trajectory_similarity: bool = False


@dataclass
class EnvironmentConfig:
    """Environment / reward function settings.

    Attributes:
        name: Short name of the environment (informational).
        reward_fn: Fully-qualified path to the reward scoring function.
    """

    name: str = "gsm8k"
    reward_fn: str = "eigrpo.environments.gsm8k_reward.compute_score"


@dataclass
class LoggingConfig:
    """Rollout table and extra logging options.

    Attributes:
        rollout_table_freq: Log a wandb table of rollout samples every N
            training steps. ``0`` disables table logging.  Step 0 is always
            skipped regardless.
        rollout_table_max_rows: Maximum number of rows per table snapshot
            (legacy field, unused by the selection table).
        rollout_table_max_groups: Maximum number of prompt groups to include
            in the selection table.  Each group contributes G rows (one per
            rollout), so the total row count is at most
            ``rollout_table_max_groups * G``.
    """

    rollout_table_freq: int = 0
    rollout_table_max_rows: int = 32
    rollout_table_max_groups: int = 2


@dataclass
class EIGRPOConfig:
    """Top-level EIGRPO configuration.

    Lives under the ``eigrpo:`` key in the YAML config.

    Attributes:
        use_wandb: Whether wandb is available for table logging.
        selector: Trajectory selection settings.
        diversity: Diversity / diagnostic logging toggles.
        environment: Environment and reward function settings.
        logging: Rollout table and extra logging options.
    """

    use_wandb: bool = False
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def parse_eigrpo_config(raw_config: Any) -> EIGRPOConfig:
    """Parse an OmegaConf / dict ``raw_config`` into a typed :class:`EIGRPOConfig`."""
    cfg = raw_config.get("eigrpo", {})

    selector = SelectorConfig(
        name=cfg.get("selector", {}).get("name", "uniform"),
        n_effective_samples=cfg.get("selector", {}).get("n_effective_samples", None),
    )
    diversity = DiversityConfig(
        log_gradient_norms=cfg.get("diversity", {}).get("log_gradient_norms", False),
        log_trajectory_similarity=cfg.get("diversity", {}).get("log_trajectory_similarity", False),
    )
    environment = EnvironmentConfig(
        name=cfg.get("environment", {}).get("name", "gsm8k"),
        reward_fn=cfg.get("environment", {}).get(
            "reward_fn", "eigrpo.environments.gsm8k_reward.compute_score"
        ),
    )

    logging_cfg = LoggingConfig(
        rollout_table_freq=cfg.get("logging", {}).get("rollout_table_freq", 0),
        rollout_table_max_rows=cfg.get("logging", {}).get("rollout_table_max_rows", 32),
        rollout_table_max_groups=cfg.get("logging", {}).get("rollout_table_max_groups", 2),
    )

    return EIGRPOConfig(
        use_wandb=cfg.get("use_wandb", False),
        selector=selector,
        diversity=diversity,
        environment=environment,
        logging=logging_cfg,
    )
