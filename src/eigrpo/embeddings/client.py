"""In-process vLLM embedding model."""

from __future__ import annotations

import os

import torch
from vllm import LLM

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.15
DEFAULT_DEVICE_INDEX = 0
DEFAULT_MAX_MODEL_LEN = 8192


class VLLMEmbeddingClient:
    """Wraps a vLLM pooling model loaded in-process.

    The model is loaded directly onto the GPU via vLLM's Python API.
    No separate HTTP server or subprocess is required.

    ``device_index`` pins the model to a specific physical GPU.  This is
    needed when the enclosing Ray actor was created with ``num_gpus=0``
    (Ray sets ``CUDA_VISIBLE_DEVICES=""`` in that case).  We override the
    env-var *before* vLLM spawns its ``EngineCore`` subprocess so the child
    process inherits the corrected value.

    ``enforce_eager=True`` is required to avoid CUDA-graph conflicts with the
    training rollout vLLM server.  The rollout server uses
    ``free_cache_engine=True`` and re-captures CUDA graphs between steps on
    the same GPU.  If the embedding model also captures CUDA graphs on that
    GPU the two runtime instances deadlock.  Eager mode is fast enough for
    embedding workloads and eliminates the conflict entirely.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        dtype: str = "bfloat16",
        device_index: int = DEFAULT_DEVICE_INDEX,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    ) -> None:
        # Ray sets CUDA_VISIBLE_DEVICES="" for actors with num_gpus=0.
        # Override it before vLLM's EngineCore subprocess is spawned so that
        # the child process (which copies the *current* env) can see the GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

        self._max_model_len = max_model_len
        self._llm = LLM(
            model=model,
            task="embed",
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=max_model_len,
        )

    def create_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Embed a batch of texts and return a (len(texts), dim) float32 tensor.

        Texts longer than ``max_model_len`` are silently truncated by vLLM so
        that responses of arbitrary length never crash the embedding step.
        """
        outputs = self._llm.embed(texts, truncate_prompt_tokens=self._max_model_len)
        # Build the matrix in one allocation rather than 1 tensor per response.
        return torch.tensor([o.outputs.embedding for o in outputs], dtype=torch.float32)
