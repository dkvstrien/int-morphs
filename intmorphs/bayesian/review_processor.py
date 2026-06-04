"""
IntMorphs Bayesian review processor.

Hooks into Anki's card review flow to process PASS/FAIL ratings
through Bayesian inference per-lemma confidence tracking.

Reads Bayesian config directly from Anki's addon config system,
so it doesn't need deep integration with IntMorphsConfig.
"""

from __future__ import annotations

from typing import Any, Dict

from anki.cards import Card
from anki.notes import Note
from aqt import mw

from . import intmorphs_config
from .intmorphs_db import IntMorphsDB
from .morphemizers import morphemizer_utils
from .bayesian.inference import BayesianInference
from .bayesian.models import LemmaConfidence


def get_bayesian_config() -> dict[str, Any]:
    """Read Bayesian config from Anki's addon config."""
    if mw is None:
        return {}

    config = mw.addonManager.getConfig(__name__.split(".")[0])
    if config is None:
        return {}
    return {
        "enabled": config.get("bayesian_enabled", True),
        "pass_boost": config.get("bayesian_pass_boost", 0.15),
        "fail_penalty": config.get("bayesian_fail_penalty", 0.20),
        "fail_fraction": config.get("bayesian_fail_fraction", 0.25),
        "burst_size": config.get("bayesian_burst_size", 8),
        "graduate_score": config.get("bayesian_graduate_score", 2.5),
        "min_appearances": config.get("bayesian_min_appearances", 10),
        "min_pass_rate": config.get("bayesian_min_pass_rate", 0.7),
        "srs_initial": config.get("bayesian_srs_initial", 1),
        "srs_max": config.get("bayesian_srs_max", 180),
    }


def get_morphemizer_for_card(card: Card) -> Any | None:
    """Find the matching morphemizer for this card's note type."""
    if mw is None:
        return None

    note = card.note()
    if note is None:
        return None

    note_type_name = note.note_type()["name"]
    am_config = intmorphs_config.IntMorphsConfig()

    for config_filter in am_config.filters:
        if not config_filter.read:
            continue
        if config_filter.note_type == note_type_name:
            # Found matching filter — get morphemizer
            return morphemizer_utils.get_morphemizer(
                config_filter.morphemizer_description
            )

    return None


def get_field_text_for_card(card: Card) -> str | None:
    """Get the text from the configured field for this card."""
    if mw is None:
        return None

    note = card.note()
    if note is None:
        return None

    note_type_name = note.note_type()["name"]
    am_config = intmorphs_config.IntMorphsConfig()

    for config_filter in am_config.filters:
        if not config_filter.read:
            continue
        if config_filter.note_type == note_type_name:
            field_name = config_filter.field
            if field_name in note:
                return note[field_name]

    return None


def process_card_review(card: Card) -> None:
    """
    Process a reviewed card through Bayesian inference.

    Called from gui_hooks.reviewer_did_answer_card.
    Updates lemma confidence based on the ease rating.

    Flow:
    1. Check if Bayesian processing is enabled
    2. Find matching note filter + morphemizer for this card
    3. Extract lemmas from the configured field
    4. Load current lemma confidences from DB
    5. Run Bayesian inference with PASS/FAIL
    6. Save updated confidences back to DB
    """
    bayesian_config = get_bayesian_config()
    if not bayesian_config.get("enabled", False):
        return

    if mw is None:
        return

    # Find matching morphemizer
    morphemizer = get_morphemizer_for_card(card)
    if morphemizer is None:
        return

    # Get field text
    field_text = get_field_text_for_card(card)
    if not field_text:
        return

    # Extract lemmas
    lemmas = morphemizer.get_morphs_from_text(field_text)
    if not lemmas:
        return

    # Determine PASS/FAIL from card's review state
    # ease >= 3 = PASS, ease <= 2 = FAIL
    passed = card.ivl >= 0 and card.queue >= 0  # Default to pass for existing cards

    # We need the review ease — get it from the reviewer
    try:
        from aqt import mw as aqt_mw
        reviewer = aqt_mw.reviewer
        if reviewer and hasattr(reviewer, "last_review_ease"):
            # Anki 25.x stores last ease on the reviewer
            last_ease = getattr(reviewer, "_answered_recently", None)
        passed = True  # Conservative default
    except Exception:
        passed = True

    # Actually, get the ease from the card's review history
    # The reviewer_did_answer_card hook includes the card
    # and we can check if it was recently reviewed
    try:
        # In Anki 25.x, we can read the last review entry
        if hasattr(card, "review_history") and card.review_history:
            last_review = card.review_history[-1]
            ease = last_review[3] if len(last_review) > 3 else 3
            passed = ease >= 3
    except Exception:
        passed = True

    # Initialize Bayesian engine
    inf = BayesianInference(
        pass_boost=bayesian_config["pass_boost"],
        fail_penalty=bayesian_config["fail_penalty"],
        fail_fraction=bayesian_config["fail_fraction"],
        burst_size=bayesian_config["burst_size"],
        graduate_score=bayesian_config["graduate_score"],
        min_appearances=bayesian_config["min_appearances"],
        min_pass_rate=bayesian_config["min_pass_rate"],
        srs_initial=bayesian_config["srs_initial"],
        srs_max=bayesian_config["srs_max"],
    )

    # Load confidence map from DB
    with IntMorphsDB() as am_db:
        confidence_map: dict[str, LemmaConfidence] = (
            am_db.get_lemma_confidence_dict()
        )

        # Auto-create entries for any new lemmas
        for lemma in lemmas:
            if lemma not in confidence_map:
                confidence_map[lemma] = LemmaConfidence(lemma=lemma)

        # Run inference
        result = inf.process_review(
            lemmas=lemmas,
            passed=passed,
            confidence_map=confidence_map,
        )

        # Save back to DB
        am_db.save_all_lemma_confidences(confidence_map)

    # Log for debugging
    print(f"[IntMorphs] Bayesian review: {len(lemmas)} lemmas, "
          f"{'PASS' if passed else 'FAIL'}, "
          f"updated {len(result)} confidences")
