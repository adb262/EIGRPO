"""Metrics transport layer — console + optional wandb.

No metric *computation* lives here.  This module only moves flat
``dict[str, float]`` payloads to their destinations.
"""

from __future__ import annotations

import logging
from typing import Any

import wandb

logger = logging.getLogger("eigrpo")


class MetricsLogger:
    """Thin facade over console logging and wandb.

    Every metric key is automatically prefixed (e.g. ``eigrpo/entropy/mean``)
    so that dashboards stay organised without the caller worrying about
    namespacing.
    """

    def __init__(self, use_wandb: bool = False, prefix: str = "eigrpo") -> None:
        self._use_wandb = use_wandb
        self._prefix = prefix

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Send *metrics* to the console and, optionally, to wandb."""
        prefixed = {f"{self._prefix}/{k}": v for k, v in metrics.items()}
        logger.info("step=%d  %s", step, prefixed)
        if self._use_wandb:
            wandb.log({"global_step": step, **prefixed})


def log_diversity_metrics(
    step: int,
    metrics: dict[str, Any],
    *,
    use_wandb: bool = False,
) -> None:
    """Legacy helper — delegates to :class:`MetricsLogger`.

    Kept for backward-compatibility with call-sites that have not yet
    migrated to the class-based API.
    """
    flat = {k: v for k, v in metrics.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    MetricsLogger(use_wandb=use_wandb, prefix="eigrpo/diversity").log(step, flat)
