"""Tests for the structured GPT-4.1-mini reasonability judge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eigrpo.environments import reasonability_judge


def test_reasonability_scores_are_normalized(monkeypatch):
    judgment = reasonability_judge.ReasonabilityJudgment(
        relevance=3,
        coherence=2,
        instruction_following=1,
        generation_quality=0,
    )
    monkeypatch.setattr(reasonability_judge, "_request_judgment", lambda question, response, model: judgment)
    reasonability_judge.score_reasonability.cache_clear()

    assert reasonability_judge.score_reasonability("Question", "Response") == pytest.approx(0.5)


def test_request_uses_structured_output_without_ground_truth(monkeypatch):
    captured = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=reasonability_judge.ReasonabilityJudgment(
                    relevance=3,
                    coherence=3,
                    instruction_following=3,
                    generation_quality=3,
                )
            )

    fake_client = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(reasonability_judge, "_get_client", lambda: fake_client)

    result = reasonability_judge._request_judgment("Question text", "Candidate response", "gpt-4.1-mini")

    assert result.relevance == 3
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["text_format"] is reasonability_judge.ReasonabilityJudgment
    assert captured["store"] is False
    assert "Do not solve" in captured["input"][0]["content"]
    payload = json.loads(captured["input"][1]["content"].split("\n", 1)[1])
    assert payload == {"question": "Question text", "response": "Candidate response"}
    assert "ground_truth" not in json.dumps(captured["input"])
