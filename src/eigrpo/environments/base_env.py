"""Thin wrapper over verl's BaseTool that adds EIGRPO-specific hooks.

The main additions are:
- Trajectory recording: every tool call / response pair is stored so
  that the selector can inspect the full multi-turn trajectory.
- A ``get_trajectories`` method consumed by the selector after
  rollout generation completes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class EIGRPOBaseTool(ABC, BaseTool):
    """Base environment for EIGRPO experiments.

    Subclasses must implement ``execute`` and ``calc_reward`` as in
    verl's ``BaseTool``.  This wrapper transparently records every
    tool-call interaction so that trajectory-level diversity analysis
    can be performed after rollout generation.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._trajectory_store: dict[str, list[dict[str, Any]]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._trajectory_store[instance_id] = []
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        response, reward, metrics = await self._execute_impl(instance_id, parameters, **kwargs)

        self._trajectory_store.setdefault(instance_id, []).append(
            {
                "parameters": parameters,
                "response": response.text if response.text else "",
                "step_reward": reward,
            }
        )
        return response, reward, metrics

    @abstractmethod
    async def _execute_impl(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        """Subclasses override this instead of ``execute`` directly."""
        raise NotImplementedError

    def get_trajectory(self, instance_id: str) -> list[dict[str, Any]]:
        """Return the recorded interaction history for one rollout."""
        return list(self._trajectory_store.get(instance_id, []))

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._trajectory_store.pop(instance_id, None)
