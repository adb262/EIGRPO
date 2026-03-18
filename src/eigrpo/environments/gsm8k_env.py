"""GSM8K multi-turn tool environment for EIGRPO experiments.

Mirrors verl's built-in ``Gsm8kTool`` but routes through
``EIGRPOBaseTool`` so that trajectories are automatically recorded
for diversity analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.reward_score import gsm8k

from eigrpo.environments.base_env import EIGRPOBaseTool

logger = logging.getLogger(__name__)


class GSM8KTool(EIGRPOBaseTool):
    """Multi-turn GSM8K evaluation tool with trajectory recording."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        self._trajectory_store[instance_id] = []
        return instance_id, ToolResponse()

    async def _execute_impl(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer

        reward = await self.calc_reward(instance_id)
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        self._instance_dict[instance_id]["reward"] = reward

        return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instance_dict.pop(instance_id, None)
        await super().release(instance_id, **kwargs)
