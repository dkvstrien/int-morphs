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

from intmorphs import intmorphs_config
from intmorphs.intmorphs_db import IntMorphsDB
from intmorphs.morphemizers import morphemizer_utils
from intmorphs.bayesian.inference import BayesianInference
from intmorphs.bayesian.models import LemmaConfidence


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
            return morphemizer_utils.get_morphemizer_by_description(
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


def process_bayesian_review(reviewer, card: Card, ease: int) -> None:
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

    # Extract lemmas from the field text
    # get_morphemes returns Iterator[list[Morpheme]] — extract lemma strings
    try:
        morph_lists = list(morphemizer.get_morphemes([field_text]))
        lemmas = []
        if morph_lists:
            for morph in morph_lists[0]:
                lemmas.append(morph.lemma)
    except Exception as exc:
        print(f"[IntMorphs] Morphemizer error: {exc}")
        return

    if not lemmas:
        return

    # Determine PASS/FAIL from ease rating
    # ease 1=Again, 2=Hard → FAIL; 3=Good, 4=Easy → PASS
    passed = ease >= 3

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
