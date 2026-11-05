import pytest

from gen_ai.adapters.llms.callable import CallableLLM
from gen_ai.core.exceptions import LLMError
from gen_ai.interfaces.llm import BaseLLM


def test_callable_llm_subclasses_base_llm():
    llm = CallableLLM(generate=lambda prompt: "Hello")
    assert isinstance(llm, BaseLLM)


def test_callable_llm_generate():
    def mock_generate(prompt: str) -> str:
        return f"Response to: {prompt}"

    llm = CallableLLM(generate=mock_generate)
    assert llm.generate("test prompt") == "Response to: test prompt"


def test_callable_llm_non_callable_init():
    with pytest.raises(TypeError, match="generate must be callable"):
        CallableLLM(generate="not_a_function")  # type: ignore


def test_callable_llm_invalid_return_type():
    def bad_generate(prompt: str) -> int:
        return 123  # type: ignore

    llm = CallableLLM(generate=bad_generate)  # type: ignore
    with pytest.raises(LLMError, match="must return a string"):
        llm.generate("prompt")


def test_callable_llm_exception_propagation():
    def failing_generate(prompt: str) -> str:
        raise ValueError("Model service unreachable")

    llm = CallableLLM(generate=failing_generate)
    with pytest.raises(LLMError, match="LLM generation failed"):
        llm.generate("prompt")
