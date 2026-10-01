"""Shared test setup: make the project root importable and expose the fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = Path(__file__).parent / "fixtures" / "aurora_manual.md"


@pytest.fixture(scope="session")
def fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()
