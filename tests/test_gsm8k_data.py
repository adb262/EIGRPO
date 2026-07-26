"""Tests for GSM8K preprocessing helpers."""

from eigrpo.data.gsm8k import _format_example, extract_ground_truth


def test_extract_ground_truth_normalizes_commas():
    assert extract_ground_truth("reasoning\n#### 1,234") == "1234"


def test_formatted_example_uses_xml_answer_contract():
    row = _format_example(
        {"question": "What is 40 + 2?", "answer": "Add them.\n#### 42"},
        index=7,
        split="train",
    )
    assert row["reward_model"]["ground_truth"] == "42"
    assert "<answer>...</answer>" in row["prompt"][0]["content"]
    assert row["extra_info"]["split"] == "train"
