"""Canonical filesystem layout for the portfolio repo."""
from __future__ import annotations
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
DATA: Path = ROOT / "data"
RAW: Path = DATA / "raw"
PROCESSED: Path = DATA / "processed"
MODELS: Path = ROOT / "models"
OUTPUT: Path = ROOT / "output"


def raw(*parts: str) -> Path:
    return RAW.joinpath(*parts)


def processed(*parts: str) -> Path:
    return PROCESSED.joinpath(*parts)


def model(*parts: str) -> Path:
    return MODELS.joinpath(*parts)
