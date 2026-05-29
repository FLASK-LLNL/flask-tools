from __future__ import annotations

import json

from flask_tools.pipette.reaction_fixer import LLMReactionFixer


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


def test_llm_reaction_fixer_derives_added_products_from_fixed_reaction() -> None:
    original_reaction_smiles = (
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    )
    fixed_reaction_smiles = (
        "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[I-]"
    )

    client = _FakeClient(
        _FakeResponse(
            json.dumps(
                {
                    "fixed_reaction_smiles": fixed_reaction_smiles,
                    "comment": "Added iodide to the product side.",
                }
            )
        )
    )
    fixer = LLMReactionFixer(
        model="gpt-5.4",
        url="https://example.test/v1/chat/completions",
        api_key="fake-key",
        client=client,
    )

    fix = fixer.fix(original_reaction_smiles, [])

    assert (
        fix.fixed_reaction_smiles
        == "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[I-]"
    )
    assert fix.added_products == ["[I-]"]
    assert fix.added_reactants == []
    assert fix.removed_agents == []
    assert fix.reasoning_summary == "Added iodide to the product side."
