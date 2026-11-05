"""Logging utilities for the gen-ai framework."""

import logging

_ROOT_LOGGER_NAME = "gen_ai"


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a hierarchical logger namespaced under 'gen_ai'.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)

    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)

    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
