import pytest

from gen_ai.interfaces.llm import BaseLLM


class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"


class BrokenLLM(BaseLLM):
    pass


def test_custom_llm_implementation():

    llm = MockLLM()

    response = llm.generate("What is Python?")

    assert response == "Mock response: What is Python?"


def test_llm_is_abstract():

    with pytest.raises(TypeError):
        BaseLLM()


def test_missing_generate_method():

    with pytest.raises(TypeError):
        BrokenLLM()
