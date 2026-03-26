"""EIGRPO-aware verl TaskRunner.

Overrides verl's default ``TaskRunner.run()`` to instantiate
:class:`~eigrpo.trainers.eigrpo_trainer.EIGRPOTrainer` instead of
``RayPPOTrainer``.
"""

from __future__ import annotations

import logging
from pprint import pprint

import ray
from omegaconf import OmegaConf

from eigrpo.trainers.eigrpo_trainer import EIGRPOTrainer
from eigrpo.utils.logging import configure_logging
from eigrpo.workers.fsdp_worker import EIGRPOActorRolloutRefWorker
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local


class EIGRPOTaskRunner(TaskRunner):
    """TaskRunner that creates an :class:`EIGRPOTrainer` instead of
    verl's default ``RayPPOTrainer``, and uses
    :class:`~eigrpo.workers.fsdp_worker.EIGRPOActorRolloutRefWorker` so that
    the ``compute_jacobian_rank`` remote method is available on the actor
    worker group.
    """

    def add_actor_rollout_worker(self, config):
        """Override to substitute EIGRPOActorRolloutRefWorker for the default
        AsyncActorRolloutRefWorker when the actor strategy is fsdp or fsdp2.

        For any other strategy (megatron, engine) we fall back to the base
        implementation — the Jacobian rank feature will simply be unavailable.
        """
        from verl.single_controller.ray import RayWorkerGroup

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            actor_rollout_cls = EIGRPOActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
            self.mapping[Role.ActorRollout] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Fallback for non-FSDP strategies.
        return super().add_actor_rollout_worker(config)

    def run(self, config):
        configure_logging()
        logger = logging.getLogger("eigrpo.task_runner")
        logger.info("EIGRPOTaskRunner.run() started in worker pid=%d", __import__("os").getpid())
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        if need_critic(config):
            self.add_critic_worker(config)

        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = EIGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
