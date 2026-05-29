from __future__ import annotations

import json

from flask_tools.pipette.judges import LLMJudge
from flask_tools.pipette.constants import FinalGrade, ToolResult, ToolStatus


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, messages: list[dict[str, str]]) -> _FakeResponse:
        self.calls.append({"model": model, "messages": messages})
        return self.response


class _FakeChat:
    def __init__(self, response: _FakeResponse) -> None:
        self.completions = _FakeCompletions(response)


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.chat = _FakeChat(response)


def test_llm_judge_posts_openai_payload_and_omits_grade_hint(
    tmp_path, monkeypatch
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Judge the reaction.", encoding="utf-8")

    client = _FakeClient(
        _FakeResponse(
            json.dumps(
                {
                    "final_grade": "possible",
                    "short_reason": "ai.mock",
                    "comment": "Mass is slightly off but still plausible.",
                }
            )
        )
    )
    judge = LLMJudge(
        url="https://example.test/v1/chat/completions",
        model="gpt-5.4",
        api_key="fake-key",
        prompt_path=prompt_path,
        client=client,
    )

    result = judge.judge(
        "CCO>>CC=O",
        [
            ToolResult(
                name="mass_conservation",
                status=ToolStatus.POTENTIAL,
                grade_hint=FinalGrade.POSSIBLE,
                data={"mass_difference_amu": 18.0, "possible_missing_products": ["O"]},
                comment="Missing water is plausible.",
            ),
            ToolResult(
                name="reaction_energy",
                status=ToolStatus.PASS,
                grade_hint=FinalGrade.LIKELY,
                data={"source": "fake_dft", "energy_difference": -3.2},
                comment="Energy is favorable.",
            ),
        ],
    )

    captured = client.chat.completions.calls[0]
    assert result.final_grade is FinalGrade.POSSIBLE
    assert result.short_reason == "ai.mock"
    assert result.comment == "Mass is slightly off but still plausible."
    payload = captured
    assert payload["model"] == "gpt-5.4"
    assert payload["messages"][0]["content"] == "Judge the reaction."

    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["reaction_smiles"] == "CCO>>CC=O"
    assert len(user_payload["tool_results"]) == 2
    assert all(
        "grade_hint" not in tool_result for tool_result in user_payload["tool_results"]
    )
    assert user_payload["tool_results"][0]["name"] == "mass_conservation"
    assert user_payload["tool_results"][0]["data"]["mass_difference_amu"] == 18.0
    assert user_payload["tool_results"][1]["data"]["source"] == "fake_dft"
