import logging

from gen_ai.utils.logging import get_logger


def test_get_logger():
    root_logger = get_logger()
    assert root_logger.name == "gen_ai"

    named_logger = get_logger("retrieval")
    assert named_logger.name == "gen_ai.retrieval"

    already_prefixed = get_logger("gen_ai.custom")
    assert already_prefixed.name == "gen_ai.custom"

    assert isinstance(named_logger, logging.Logger)
