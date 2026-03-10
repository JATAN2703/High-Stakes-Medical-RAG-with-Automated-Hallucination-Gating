"""
src/utils.py
============
Shared utilities: configuration loading, logging setup, and OpenRouter client.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently formatted logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


# ── Config ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load and cache the YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the config YAML file, relative to project root.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_prompts(path: str = "configs/prompts.yaml") -> dict[str, Any]:
    """
    Load and cache the versioned prompts file.

    Parameters
    ----------
    path : str
        Path to prompts YAML file.

    Returns
    -------
    dict[str, Any]
        Nested prompt dictionary keyed by version then prompt name.
    """
    prompts_path = Path(path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path.resolve()}")
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


# ── OpenRouter Client ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    """
    Build and cache a reusable OpenRouter client.

    OpenRouter uses an OpenAI-compatible API, so the standard ``openai``
    library works with just a ``base_url`` swap.

    Returns
    -------
    OpenAI
        Configured client pointed at OpenRouter.

    Raises
    ------
    EnvironmentError
        If ``OPENROUTER_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Make a single chat completion call via OpenRouter.

    Parameters
    ----------
    system_prompt : str
        System-level instruction for the model.
    user_prompt : str
        User message / query.
    model : str
        OpenRouter model string (e.g. ``"openai/gpt-4o-mini"``).
    temperature : float
        Sampling temperature. Use 0.0 for deterministic outputs.
    max_tokens : int
        Maximum tokens in the completion.

    Returns
    -------
    str
        The model's response text, stripped of leading/trailing whitespace.
    """
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
