"""Smoke test: el paquete unravel importa y Python es 3.12."""
from __future__ import annotations

import sys

import unravel


def test_paquete_importa() -> None:
    assert unravel is not None


def test_python_es_312() -> None:
    assert sys.version_info[:2] == (3, 12)
