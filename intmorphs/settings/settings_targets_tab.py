"""
IntMorphs Target Words settings tab.

Allows the user to:
- Set a ranked list of target lemmas (paste or type)
- Configure soak limit and graduation stage
- See current status (active, graduated, queued)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from aqt import mw
from aqt.qt import (  # pylint:disable=no-name-in-module
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QComboBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QWidget,
)

from .settings_tab import SettingsTab
from ..intmorphs_config import IntMorphsConfig, RawConfigKeys
from ..ui.settings_dialog_ui import Ui_SettingsDialog
from ..intmorphs_db import IntMorphsDB
from ..bayesian.models import LemmaConfidence
from ..bayesian.target_manager import TargetPool


class TargetWordsTab(SettingsTab):
    """Settings tab for managing the ranked target word list and soak mode."""

    def __init__(
        self,
        parent: QDialog,
        ui: Ui_SettingsDialog,
        config: IntMorphsConfig,
        default_config: IntMorphsConfig,
    ) -> None:
        # We avoid the parent __init__ since we build our own UI independently
        self._parent = parent
        self.ui = ui
        self._config = config
        self._default_config = default_config

        self._raw_config_key_to_radio_button: dict = {}
        self._raw_config_key_to_check_box: dict = {}
        self._raw_config_key_to_spin_box: dict = {}
        self._raw_config_key_to_combo_box: dict = {}
        self._raw_config_key_to_line_edit: dict = {}
        self._raw_config_key_to_key_sequence: dict = {}
        self._previous_state: dict[str, str | int | bool | object] | None = None

        # Build the UI
        self.own_widget = QWidget()
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self.own_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Config group ──────────────────────────────────────────
        config_group = QGroupBox("Soak Mode Configuration")
        config_layout = QVBoxLayout(config_group)

        # Soak limit row
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Active target words at once:"))
        self.soak_limit_spin = QSpinBox()
        self.soak_limit_spin.setRange(1, 200)
        self.soak_limit_spin.setValue(self._config.soak_limit)
        self.soak_limit_spin.setToolTip(
            "How many target words are actively 'soaked' at the same time. "
            "When one graduates, the next from the list is promoted."
        )
        limit_row.addWidget(self.soak_limit_spin)
        limit_row.addStretch()
        config_layout.addLayout(limit_row)

        # Graduation stage row
        grade_row = QHBoxLayout()
        grade_row.addWidget(QLabel("Graduation stage:"))
        self.graduate_stage_combo = QComboBox()
        self.graduate_stage_combo.addItems(["srs", "mature"])
        self.graduate_stage_combo.setCurrentText(self._config.soak_graduate_stage)
        self.graduate_stage_combo.setToolTip(
            "A target word graduates when it reaches this stage. "
            "'srs' = enters SRS interval management (default). "
            "'mature' = only when fully mature."
        )
        grade_row.addWidget(self.graduate_stage_combo)
        grade_row.addStretch()
        config_layout.addLayout(grade_row)

        layout.addWidget(config_group)

        # ── Target list group ─────────────────────────────────────
        list_group = QGroupBox("Target Word List (ranked order, highest priority first)")
        list_layout = QVBoxLayout(list_group)

        # Instructions
        instr = QLabel(
            "Paste or type your target lemmas below, one per line. "
            "The order determines priority — the first N (set above) will be "
            "actively soaked. As each graduates, the next is promoted."
        )
        instr.setWordWrap(True)
        list_layout.addWidget(instr)

        # Text area for pasting
        self.target_list_edit = QPlainTextEdit()
        self.target_list_edit.setPlaceholderText(
            "читать\nписать\nговорить\nслушать\n..."
        )
        self.target_list_edit.setMinimumHeight(150)
        list_layout.addWidget(self.target_list_edit)

        # Buttons row
        btn_row = QHBoxLayout()
        self.apply_targets_btn = QPushButton("Set Target Words")
        self.apply_targets_btn.setToolTip("Write the list above to the database as ranked targets")
        self.apply_targets_btn.clicked.connect(self._on_apply_targets)

        self.refresh_btn = QPushButton("Refresh from Database")
        self.refresh_btn.setToolTip("Reload the current target list from the database")
        self.refresh_btn.clicked.connect(self._refresh_display)

        self.clear_btn = QPushButton("Clear All Targets")
        self.clear_btn.setToolTip("Remove all target word assignments")
        self.clear_btn.clicked.connect(self._on_clear_targets)

        btn_row.addWidget(self.apply_targets_btn)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        list_layout.addLayout(btn_row)

        layout.addWidget(list_group)

        # ── Status table group ────────────────────────────────────
        status_group = QGroupBox("Current Target Status")
        status_layout = QVBoxLayout(status_group)

        # Summary labels
        summary_row = QHBoxLayout()
        self.active_label = QLabel("Active: —")
        self.queued_label = QLabel("Queued: —")
        self.graduated_label = QLabel("Graduated: —")
        self.total_label = QLabel("Total: —")
        for lbl in [self.active_label, self.queued_label, self.graduated_label, self.total_label]:
            lbl.setStyleSheet("font-weight: bold;")
        summary_row.addWidget(self.active_label)
        summary_row.addWidget(self.queued_label)
        summary_row.addWidget(self.graduated_label)
        summary_row.addWidget(self.total_label)
        summary_row.addStretch()
        status_layout.addLayout(summary_row)

        # Table
        self.status_table = QTableWidget(0, 6)
        self.status_table.setHorizontalHeaderLabels(
            ["Rank", "Lemma", "Score", "Stage", "Active", "Graduated"]
        )
        self.status_table.horizontalHeader().setStretchLastSection(True)
        self.status_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.status_table.setAlternatingRowColors(True)
        status_layout.addWidget(self.status_table)

        layout.addWidget(status_group)
        layout.addStretch()

        # Load initial data
        self._refresh_display()

    # ── Data loading ──────────────────────────────────────────────

    def _refresh_display(self) -> None:
        """Reload target data from the database and update the UI."""
        if mw is None:
            return

        try:
            with IntMorphsDB() as am_db:
                confidence_map = am_db.get_lemma_confidence_dict()
                ranked_lemmas = am_db.get_target_ranked_lemmas()

                soak_limit = self.soak_limit_spin.value()
                graduate_stage = self.graduate_stage_combo.currentText()
                pool = TargetPool(
                    confidence_map,
                    soak_limit=soak_limit,
                    graduate_stage=graduate_stage,
                )

                # Populate the text area with the ranked list
                self.target_list_edit.setPlainText("\n".join(ranked_lemmas))

                # Update summary
                active = pool.get_active_targets()
                queued = pool.get_queued_targets()
                graduated = pool.get_graduated_targets()

                self.active_label.setText(f"Active: {len(active)}")
                self.queued_label.setText(f"Queued: {len(queued)}")
                self.graduated_label.setText(f"Graduated: {len(graduated)}")
                self.total_label.setText(f"Total: {len(ranked_lemmas)}")

                # Populate table
                all_targets = pool.get_ranked_targets()
                self.status_table.setRowCount(len(all_targets))
                for row, lc in enumerate(all_targets):
                    active_lemmas = {l.lemma for l in active}
                    graduated_lemmas = {l.lemma for l in graduated}

                    self.status_table.setItem(row, 0, QTableWidgetItem(str(lc.target_rank or "")))
                    self.status_table.setItem(row, 1, QTableWidgetItem(lc.lemma))
                    self.status_table.setItem(row, 2, QTableWidgetItem(f"{lc.score:.2f}"))
                    self.status_table.setItem(row, 3, QTableWidgetItem(lc.stage.value))
                    self.status_table.setItem(
                        row, 4,
                        QTableWidgetItem("✓" if lc.lemma in active_lemmas else ""),
                    )
                    self.status_table.setItem(
                        row, 5,
                        QTableWidgetItem("✓" if lc.lemma in graduated_lemmas else ""),
                    )

        except Exception as exc:
            print(f"[IntMorphs] Error refreshing target display: {exc}")

    def _on_apply_targets(self) -> None:
        """Write the typed/pasted list to the database as ranked targets."""
        if mw is None:
            return

        text = self.target_list_edit.toPlainText().strip()
        if not text:
            self._clear_targets()
            return

        # Parse: one per line, strip whitespace, skip empties
        lemmas = [
            line.strip().lower()
            for line in text.split("\n")
            if line.strip()
        ]

        if not lemmas:
            self._clear_targets()
            return

        with IntMorphsDB() as am_db:
            confidence_map = am_db.get_lemma_confidence_dict()

            pool = TargetPool(confidence_map)
            count = pool.set_ranked_targets(lemmas)

            # Save all back to DB
            am_db.save_all_lemma_confidences(confidence_map)

        from aqt.utils import tooltip
        tooltip(f"Set {count} ranked target words", period=2000)

        self._refresh_display()

    def _on_clear_targets(self) -> None:
        """Clear all target assignments."""
        self._clear_targets()
        self._refresh_display()

    def _clear_targets(self) -> None:
        """Internal: reset targets in DB."""
        with IntMorphsDB() as am_db:
            am_db.clear_all_target_ranks()
        self.target_list_edit.clear()
        from aqt.utils import tooltip
        tooltip("All target words cleared", period=1500)

    # ── SettingsTab interface ─────────────────────────────────────

    def setup_buttons(self) -> None:
        """No-op: our buttons are already connected in _build_ui."""
        pass

    def get_confirmation_text(self) -> str:
        return "Restore default target word settings?"

    def populate(self, use_default_config: bool = False) -> None:
        """Restore UI from config values."""
        source = self._default_config if use_default_config else self._config
        self.soak_limit_spin.setValue(source.soak_limit)
        self.graduate_stage_combo.setCurrentText(source.soak_graduate_stage)
        self._refresh_display()

    def settings_to_dict(self) -> dict[str, str | int | float | bool | object]:
        """Return config values that should be saved."""
        return {
            RawConfigKeys.SOAK_LIMIT: self.soak_limit_spin.value(),
            RawConfigKeys.SOAK_GRADUATE_STAGE: self.graduate_stage_combo.currentText(),
        }

    def restore_defaults(self, skip_confirmation: bool = False) -> None:
        if not skip_confirmation:
            from ..message_box_utils import show_warning_box
            title = "Confirmation"
            text = self.get_confirmation_text()
            confirmed = show_warning_box(title, text, parent=self._parent)
            if not confirmed:
                return
        self.populate(use_default_config=True)

    def restore_to_config_state(self) -> None:
        self.populate()

    def contains_unsaved_changes(self) -> bool:
        """Check if config values differ from saved state."""
        current = self.settings_to_dict()
        saved = {
            RawConfigKeys.SOAK_LIMIT: self._config.soak_limit,
            RawConfigKeys.SOAK_GRADUATE_STAGE: self._config.soak_graduate_stage,
        }
        return current != saved

    def update_previous_state(self) -> None:
        self._previous_state = self.settings_to_dict()
