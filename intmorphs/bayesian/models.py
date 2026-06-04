"""
IntMorphs Bayesian models — per-lemma confidence tracking and DB schema.

Tracks each lemma through:
  - NEW → BURST → SOAK_IN → SRS → MATURE
  - Bayesian confidence score (-5 to +5)

The lemma_confidence table is separate from AnkiMorphs' existing morph tracking.
It adds Bayesian confidence on top of AnkiMorphs' binary known/unknown tracking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


class ConfidenceStage(Enum):
    NEW = "new"
    BURST = "burst"       # intensive front-loading
    SOAK_IN = "soak_in"   # steady exposure
    SRS = "srs"           # managed by SRS intervals
    MATURE = "mature"     # fully known


@dataclass
class LemmaConfidence:
    """Bayesian confidence state for a single lemma."""

    lemma: str
    score: float = 0.0          # -5 (very unknown) to +5 (very known)
    stage: ConfidenceStage = ConfidenceStage.NEW
    appearances: int = 0         # total times seen
    pass_count: int = 0          # PASS ratings
    fail_count: int = 0          # FAIL ratings
    burst_count: int = 0         # appearances in current burst
    burst_started_at: float = 0.0
    srs_interval: int = 0        # days
    next_due: float = 0.0
    is_target: bool = True       # all tracked lemmas are targets by default
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def pass_rate(self) -> float:
        total = self.pass_count + self.fail_count
        return self.pass_count / total if total > 0 else 0.0

    @property
    def uncertainty(self) -> float:
        """0.0 (fully certain) to 1.0 (fully uncertain)."""
        return 1.0 - abs(self.score) / 5.0

    @property
    def is_unknown(self) -> bool:
        """Convenience: score < 0 means net failing."""
        return self.score <= 0

    @property
    def is_known(self) -> bool:
        return self.stage == ConfidenceStage.MATURE


# SQL for creating the lemma_confidence table
LEMMA_CONFIDENCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lemma_confidence (
    lemma TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.0,
    stage TEXT NOT NULL DEFAULT 'new',
    appearances INTEGER NOT NULL DEFAULT 0,
    pass_count INTEGER NOT NULL DEFAULT 0,
    fail_count INTEGER NOT NULL DEFAULT 0,
    burst_count INTEGER NOT NULL DEFAULT 0,
    burst_started_at REAL NOT NULL DEFAULT 0.0,
    srs_interval INTEGER NOT NULL DEFAULT 0,
    next_due REAL NOT NULL DEFAULT 0.0,
    is_target INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    updated_at REAL NOT NULL DEFAULT (unixepoch())
);
"""
