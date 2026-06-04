"""
IntMorphs Bayesian inference engine.

When a card with multiple target lemmas is rated PASS or FAIL,
this engine distributes the signal across lemmas proportional to
their uncertainty. Confident lemmas barely change; uncertain lemmas
move more.

Key improvement over AnkiMorphs' binary known/unknown: continuous
confidence scores that reflect partial knowledge.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set

from .models import LemmaConfidence, ConfidenceStage


class BayesianInference:
    """
    Updates LemmaConfidence scores based on PASS/FAIL ratings of
    cards containing multiple target lemmas.
    """

    # Default config values (overridable from IntMorphs settings)
    DEFAULT_PASS_BOOST: float = 0.15
    DEFAULT_FAIL_PENALTY: float = 0.20
    DEFAULT_FAIL_FRACTION: float = 0.25  # weakest fraction taking the blame
    DEFAULT_BURST_SIZE: int = 8
    DEFAULT_GRADUATE_SCORE: float = 2.5
    DEFAULT_MIN_APPEARANCES: int = 10
    DEFAULT_MIN_PASS_RATE: float = 0.7
    DEFAULT_SRS_INITIAL: int = 1  # days
    DEFAULT_SRS_MAX: int = 180  # days

    def __init__(
        self,
        pass_boost: float = DEFAULT_PASS_BOOST,
        fail_penalty: float = DEFAULT_FAIL_PENALTY,
        fail_fraction: float = DEFAULT_FAIL_FRACTION,
        burst_size: int = DEFAULT_BURST_SIZE,
        graduate_score: float = DEFAULT_GRADUATE_SCORE,
        min_appearances: int = DEFAULT_MIN_APPEARANCES,
        min_pass_rate: float = DEFAULT_MIN_PASS_RATE,
        srs_initial: int = DEFAULT_SRS_INITIAL,
        srs_max: int = DEFAULT_SRS_MAX,
    ):
        self.pass_boost = pass_boost
        self.fail_penalty = fail_penalty
        self.fail_fraction = fail_fraction
        self.burst_size = burst_size
        self.graduate_score = graduate_score
        self.min_appearances = min_appearances
        self.min_pass_rate = min_pass_rate
        self.srs_initial = srs_initial
        self.srs_max = srs_max

    def process_review(
        self,
        lemmas: List[str],
        passed: bool,
        confidence_map: Dict[str, LemmaConfidence],
    ) -> Dict[str, float]:
        """
        Process a PASS/FAIL rating on a card containing `lemmas`.

        Args:
            lemmas: All lemmas on the card (from morphemizer).
            passed: True if ease >= 3 (PASS), False if ease <= 2 (FAIL).
            confidence_map: Dict of lemma → LemmaConfidence (modified in-place).

        Returns:
            Dict mapping lemma → updated score.
        """
        # Filter to lemmas we track
        tracked = {}
        for lemma in lemmas:
            lc = confidence_map.get(lemma)
            if lc is not None:
                tracked[lemma] = lc
            else:
                # Auto-create for any lemma we see
                lc = LemmaConfidence(lemma=lemma)
                confidence_map[lemma] = lc
                tracked[lemma] = lc

        if not tracked:
            return {}

        if passed:
            self._apply_pass(tracked)
        else:
            self._apply_fail(tracked)

        # Check graduations and persist timestamps
        now = time.time()
        for lemma, lc in tracked.items():
            lc.updated_at = now
            self._check_graduation(lc)

        return {lemma: lc.score for lemma, lc in tracked.items()}

    def _apply_pass(self, tracked: Dict[str, LemmaConfidence]) -> None:
        """PASS: all tracked lemmas get a boost proportional to uncertainty."""
        for lemma, lc in tracked.items():
            delta = self._uncertainty_weighted_delta(lc.score, self.pass_boost)
            lc.score = min(5.0, lc.score + delta)
            lc.appearances += 1
            lc.pass_count += 1

            # Burst tracking
            if lc.stage == ConfidenceStage.NEW:
                lc.stage = ConfidenceStage.BURST
                lc.burst_count = 1
                lc.burst_started_at = time.time()
            elif lc.stage == ConfidenceStage.BURST:
                lc.burst_count += 1
                if lc.burst_count >= self.burst_size:
                    lc.stage = ConfidenceStage.SOAK_IN

    def _apply_fail(self, tracked: Dict[str, LemmaConfidence]) -> None:
        """
        FAIL: only the weakest fraction of lemmas take the hit.
        This concentrates learning on what's actually unknown.
        """
        # Sort by score ascending — weakest first
        sorted_lemmas = sorted(tracked.items(), key=lambda x: x[1].score)
        n_culprits = max(1, int(len(sorted_lemmas) * self.fail_fraction))
        culprit_lemmas = {lemma for lemma, _ in sorted_lemmas[:n_culprits]}

        for lemma, lc in tracked.items():
            lc.appearances += 1

            if lemma in culprit_lemmas:
                delta = self._uncertainty_weighted_delta(lc.score, self.fail_penalty)
                lc.score = max(-5.0, lc.score - delta)
                lc.fail_count += 1

                # Any FAIL resets burst progress
                if lc.stage == ConfidenceStage.BURST:
                    lc.stage = ConfidenceStage.SOAK_IN
                    lc.burst_count = 0
            else:
                # Non-culprit: slight positive signal
                tiny_boost = self._uncertainty_weighted_delta(lc.score, 0.02)
                lc.score = min(5.0, lc.score + tiny_boost)
                lc.pass_count += 1

    def _uncertainty_weighted_delta(self, score: float, base: float) -> float:
        """
        Higher uncertainty = larger adjustment.
        Score near 0 = high uncertainty → big delta.
        Score near ±5 = high confidence → tiny delta.
        """
        confidence = abs(score) / 5.0
        weight = math.exp(-3.0 * confidence ** 2)
        min_weight = 0.15

        # Catch-up effect: negative scores get extra boost on PASS
        if score < 0:
            catchup = 1.0 + min(1.0, abs(score) / 2.5)
            weight *= catchup

        return base * max(min_weight, weight)

    def _check_graduation(self, lc: LemmaConfidence) -> None:
        """Check if a lemma is ready to advance to the next stage."""
        if lc.stage == ConfidenceStage.BURST:
            if lc.burst_count >= self.burst_size:
                lc.stage = ConfidenceStage.SOAK_IN

        if lc.stage in (ConfidenceStage.BURST, ConfidenceStage.SOAK_IN):
            if (lc.score >= self.graduate_score
                    and lc.appearances >= self.min_appearances
                    and lc.pass_rate >= self.min_pass_rate):
                lc.stage = ConfidenceStage.SRS
                lc.srs_interval = self.srs_initial
                lc.next_due = time.time() + self.srs_initial * 86400

        if lc.stage == ConfidenceStage.SRS:
            if lc.srs_interval >= self.srs_max:
                lc.stage = ConfidenceStage.MATURE

    def process_srs_review(
        self,
        lemma: str,
        passed: bool,
        confidence_map: Dict[str, LemmaConfidence],
    ) -> Optional[float]:
        """Process an SRS review for a single lemma (1T confirmation)."""
        lc = confidence_map.get(lemma)
        if lc is None:
            lc = LemmaConfidence(lemma=lemma)
            confidence_map[lemma] = lc

        now = time.time()

        if passed:
            if lc.srs_interval == 0:
                lc.srs_interval = 1
            elif lc.srs_interval == 1:
                lc.srs_interval = 3
            else:
                lc.srs_interval = min(int(lc.srs_interval * 2.5), self.srs_max)
            lc.score = min(5.0, lc.score + 0.5)
            lc.pass_count += 1
        else:
            lc.srs_interval = 1
            lc.score = max(-5.0, lc.score - 0.5)
            lc.fail_count += 1

        lc.appearances += 1
        lc.next_due = now + lc.srs_interval * 86400
        lc.updated_at = now

        # Check mature
        if lc.srs_interval >= self.srs_max:
            lc.stage = ConfidenceStage.MATURE

        return lc.score
