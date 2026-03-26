"""EIGRPO FSDP worker: substitutes EIGRPOActor for the default
DataParallelPPOActor.

Jacobian rank is measured at two points:
  - Pre-selection: hooked into ``compute_log_prob``.  One backward pass per
    prompt-group piggybacks on the already-required forward; sketch metrics are
    stored in ``self.actor._pre_jacobian_metrics`` and returned in the
    DataProto meta_info so the trainer can log them.
  - Post-selection: hooked into ``EIGRPOActor.update_policy``; metrics are
    returned in ``meta_info["metrics"]`` as ``jacobian/post_*``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.fsdp_workers import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

from eigrpo.workers.actor import EIGRPOActor

logger = logging.getLogger("eigrpo.workers.fsdp_worker")


class EIGRPOActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """AsyncActorRolloutRefWorker that uses EIGRPOActor."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Delegate to parent, then swap in EIGRPOActor."""
        super().init_model()
        if hasattr(self, "actor"):
            self.actor = EIGRPOActor(
                config=self.actor.config,
                actor_module=self.actor.actor_module,
                actor_optimizer=self.actor.actor_optimizer,
            )
            logger.info("EIGRPOActorRolloutRefWorker: replaced actor with EIGRPOActor")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """Like parent but adds group_size to meta_info and surfaces pre-selection Jacobian rank."""
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        if not is_lora:
            # group_size = rollouts per prompt; EIGRPOActor uses this to slice
            # the flat batch into per-group backward passes for the Jacobian sketch.
            data.meta_info["group_size"] = self.config.rollout.n

        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(
                    data=data, calculate_entropy=not is_lora
                )
            tensors = {"ref_log_prob": output} if is_lora else {"old_log_probs": output}
            if not is_lora:
                tensors["entropys"] = entropys

            pre_metrics = {}
            if not is_lora and isinstance(self.actor, EIGRPOActor):
                pre_metrics = getattr(self.actor, "_pre_jacobian_metrics", {})

            result = DataProto.from_dict(
                tensors=tensors,
                meta_info={
                    "temperature": self.config.rollout.temperature,
                    "jacobian_pre_metrics": pre_metrics,
                },
            )

        result = result.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return result
