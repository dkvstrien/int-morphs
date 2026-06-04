"""
IntMorphs TargetPool — manages the ranked list of target lemmas.

The user defines a ranked list of lemmas they want to learn. The first N
(where N = soak_limit) are "active" — cards containing them get priority
in recalc. When an active target graduates (reaches SRS/MATURE stage via
Bayesian inference), the next ungraduated target from the ranked list is
promoted into the active pool.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from .models import LemmaConfidence, ConfidenceStage

log = logging.getLogger(__name__)


class TargetPool:
    """
    Manages a ranked list of target lemmas with auto-promotion.

    The user sets a ranked list (lemma → rank). The first `soak_limit`
    lemmas from this list that haven't graduated yet are "active" —
    the soak mode concentrates on them.

    Graduation = reaching SRS or MATURE stage.
    """

    def __init__(
        self,
        confidence_map: Dict[str, LemmaConfidence],
        soak_limit: int = 10,
        graduate_stage: str = "srs",
    ):
        self._confidence_map = confidence_map
        self.soak_limit = soak_limit
        self._graduate_stage = graduate_stage  # "srs" or "mature"

    # ── Reading state ──────────────────────────────────────────────

    def get_ranked_targets(self) -> List[LemmaConfidence]:
        """All target lemmas sorted by rank (ascending)."""
        targets = [
            lc for lc in self._confidence_map.values()
            if lc.is_target and lc.target_rank is not None
        ]
        targets.sort(key=lambda lc: lc.target_rank)
        return targets

    def get_active_targets(self) -> List[LemmaConfidence]:
        """Active targets = ranked & not yet graduated, limited by soak_limit."""
        ranked = self.get_ranked_targets()
        # First, find active (not graduated) targets from the top of the list
        active: List[LemmaConfidence] = []
        for lc in ranked:
            if lc.is_graduated:
                continue
            active.append(lc)
            if len(active) >= self.soak_limit:
                break
        return active

    def get_queued_targets(self) -> List[LemmaConfidence]:
        """Targets that are ranked but below the soak_limit (waiting)."""
        ranked = self.get_ranked_targets()
        active_lemmas = {lc.lemma for lc in self.get_active_targets()}
        return [lc for lc in ranked if lc.lemma not in active_lemmas and not lc.is_graduated]

    def get_graduated_targets(self) -> List[LemmaConfidence]:
        """Targets that have graduated (reached SRS/MATURE)."""
        return [
            lc for lc in self._confidence_map.values()
            if lc.is_target and lc.is_graduated
        ]

    def is_graduated(self, lc: LemmaConfidence) -> bool:
        """Check if a lemma has graduated per the configured threshold."""
        if self._graduate_stage == "srs":
            return lc.stage in (ConfidenceStage.SRS, ConfidenceStage.MATURE)
        return lc.stage == ConfidenceStage.MATURE

    # ── Mutation ───────────────────────────────────────────────────

    def check_promotion(self) -> List[str]:
        """
        Check if any active targets have graduated and promote the next in line.

        Returns a list of lemmas that were newly promoted (empty if no change).
        """
        ranked = self.get_ranked_targets()
        if not ranked:
            return []

        # Determine current state
        currently_active_lemmas: Set[str] = set()
        next_in_line: Optional[LemmaConfidence] = None

        for lc in ranked:
            if self.is_graduated(lc):
                continue
            if len(currently_active_lemmas) < self.soak_limit:
                currently_active_lemmas.add(lc.lemma)
            elif next_in_line is None:
                next_in_line = lc
                break

        # If there are empty slots and a next-in-line, fill them
        promoted: List[str] = []
        if next_in_line is not None:
            # Check if there's actually a free slot
            active = [lc for lc in ranked if lc.lemma in currently_active_lemmas]
            if len(active) < self.soak_limit:
                # How many slots are free?
                slots_free = self.soak_limit - len(active)
                queued = self.get_queued_targets()
                for qd in queued[:slots_free]:
                    log.info(
                        "[IntMorphs] Promoted target %s (rank %s) into active pool",
                        qd.lemma, qd.target_rank,
                    )
                    promoted.append(qd.lemma)

        return promoted

    def set_ranked_targets(self, ranked_lemmas: List[str]) -> int:
        """
        Set the ranked target list. Overwrites all existing target rankings.

        Args:
            ranked_lemmas: Ordered list of lemma strings (highest priority first).

        Returns:
            Number of lemmas that were newly marked as targets.
        """
        # First, reset all existing target ranks
        for lc in self._confidence_map.values():
            lc.target_rank = None
            lc.is_target = False

        # Set new ranks
        count = 0
        for rank, lemma in enumerate(ranked_lemmas, start=1):
            if lemma not in self._confidence_map:
                self._confidence_map[lemma] = LemmaConfidence(lemma=lemma)
            lc = self._confidence_map[lemma]
            lc.is_target = True
            lc.target_rank = rank
            count += 1

        log.info("[IntMorphs] Set %d ranked targets", count)
        return count

    def get_active_target_lemmas(self) -> Set[str]:
        """Convenience: set of active target lemma strings."""
        return {lc.lemma for lc in self.get_active_targets()}

    def get_target_count(self) -> int:
        """Total number of ranked targets."""
        return len(self.get_ranked_targets())

    def get_active_count(self) -> int:
        """Number of currently active targets."""
        return len(self.get_active_targets())

    def get_graduated_count(self) -> int:
        """Number of graduated targets."""
        return len(self.get_graduated_targets())
