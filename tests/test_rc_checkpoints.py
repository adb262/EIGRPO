"""Tests for RC-SFT checkpoint discovery."""

from eigrpo.rc_grpo.checkpoints import find_latest_hf_checkpoint


def test_find_latest_hf_checkpoint(tmp_path):
    older = tmp_path / "global_step_9" / "huggingface"
    newer = tmp_path / "global_step_10" / "huggingface"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    assert find_latest_hf_checkpoint(tmp_path) == newer
