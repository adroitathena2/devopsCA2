"""
Clinical Copilot - PyQt6 UI
A premium desktop interface for prescription analysis and drug interaction detection.
"""

import sys
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ─── Pre-load PyTorch BEFORE Qt ─────────────────────────────────────────────
# On Windows, Qt modifies the DLL search order which prevents PyTorch's
# c10.dll from finding its dependencies.  Importing torch first avoids
# the "[WinError 1114] DLL initialization routine failed" error.
try:
    import torch           # noqa: F401
    import sentence_transformers  # noqa: F401 — also depends on torch
except ImportError:
    pass  # backend may not be installed; handled later in _init_pipeline

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QFrame, QScrollArea,
    QSizePolicy, QProgressBar, QStackedWidget, QGridLayout,
    QGraphicsDropShadowEffect, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clinical_copilot_ui")

# ─── Colour Palette ─────────────────────────────────────────────────────────
BG_DEEP       = "#0d1117"
BG_PANEL      = "#161b22"
BG_CARD       = "#1f2937"
BG_CARD_ALT   = "#111827"
BORDER        = "#30363d"
ACCENT        = "#3b82f6"
ACCENT_HOVER  = "#2563eb"
ACCENT_GLOW   = "#1d4ed8"
SUCCESS       = "#10b981"
WARNING       = "#f59e0b"
DANGER        = "#ef4444"
DANGER_SOFT   = "#7f1d1d"
TEXT_PRIMARY  = "#f0f6fc"
TEXT_SECONDARY= "#8b949e"
TEXT_MUTED    = "#484f58"

STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: {BG_DEEP};
}}
QWidget {{
    font-family: 'Segoe UI', 'Inter', sans-serif;
    color: {TEXT_PRIMARY};
    background-color: transparent;
}}
QLabel {{
    background-color: transparent;
}}
QPlainTextEdit {{
    background-color: {BG_CARD};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 14px;
    line-height: 1.5;
    selection-background-color: {ACCENT};
}}
QPlainTextEdit:focus {{
    border: 1.5px solid {ACCENT};
}}
QPushButton#analyzeBtn {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
QPushButton#analyzeBtn:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#analyzeBtn:pressed {{
    background-color: {ACCENT_GLOW};
}}
QPushButton#analyzeBtn:disabled {{
    background-color: {TEXT_MUTED};
    color: {TEXT_SECONDARY};
}}
QPushButton#clearBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#clearBtn:hover {{
    border-color: {TEXT_SECONDARY};
    color: {TEXT_PRIMARY};
}}
QPushButton#uploadBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#uploadBtn:hover {{
    border-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background-color: {BG_PANEL};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QProgressBar {{
    background-color: {BG_CARD};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
"""


def make_shadow(color="#000000", blur=20, dx=0, dy=4, alpha=80):
    shadow = QGraphicsDropShadowEffect()
    col = QColor(color)
    col.setAlpha(alpha)
    shadow.setColor(col)
    shadow.setBlurRadius(blur)
    shadow.setOffset(dx, dy)
    return shadow


class Card(QFrame):
    """Reusable card widget with rounded corners and a subtle border."""
    def __init__(self, parent=None, alt=False):
        super().__init__(parent)
        bg = BG_CARD_ALT if alt else BG_CARD
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 0px;
            }}
        """)
        self.setGraphicsEffect(make_shadow(alpha=60))


class SectionHeader(QLabel):
    def __init__(self, text, parent=None, icon=""):
        super().__init__(f"{icon}  {text}" if icon else text, parent)
        self.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                padding: 0px;
                border: none;
                background: transparent;
            }}
        """)


class MedicineTile(QFrame):
    """Displays one identified medicine with its details."""
    def __init__(self, mapping: dict, confidence: float, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
                padding: 0px;
            }}
            QFrame:hover {{
                border-color: {ACCENT};
            }}
        """)

        is_known = bool(mapping.get("generic_names"))
        is_low_conf = confidence < 0.7

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        dot_color = SUCCESS if (is_known and not is_low_conf) else (WARNING if is_low_conf else TEXT_MUTED)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {dot_color}; font-size: 10px; background: transparent; border: none;")
        header_row.addWidget(dot)

        brand_lbl = QLabel(mapping.get("brand_name", "Unknown"))
        brand_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent; border: none;")
        header_row.addWidget(brand_lbl)
        header_row.addStretch()

        conf_lbl = QLabel(f"{confidence:.0%}")
        conf_lbl.setStyleSheet(f"""
            color: {dot_color}; font-size: 11px; font-weight: 600;
            background: transparent; border: none;
        """)
        header_row.addWidget(conf_lbl)
        layout.addLayout(header_row)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        layout.addWidget(sep)

        # Details grid
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 4, 0, 0)

        def add_row(row, key, val):
            k = QLabel(key)
            k.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent; border: none;")
            k.setFixedWidth(110)
            v = QLabel(val or "—")
            v.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent; border: none;")
            v.setWordWrap(True)
            grid.addWidget(k, row, 0)
            grid.addWidget(v, row, 1)

        generics = ", ".join(mapping.get("generic_names", [])) or "Unknown"
        add_row(0, "Generic(s):", generics)
        add_row(1, "Composition:", mapping.get("composition_full", "Unknown"))
        add_row(2, "Price:", mapping.get("price", "—"))
        add_row(3, "Manufacturer:", mapping.get("manufacturer", "—"))
        rx_val = mapping.get("prescription_required")
        add_row(4, "Rx Required:", "Yes" if rx_val is True else ("No" if rx_val is False else "Unknown"))

        layout.addLayout(grid)


class InteractionCard(QFrame):
    """Displays one drug-drug interaction."""
    def __init__(self, interaction: dict, index: int, parent=None):
        super().__init__(parent)

        # Get severity for color coding
        severity = interaction.get("severity_categorical", "Severe")

        # Determine colors based on severity
        if severity == "Severe":
            bg_color = DANGER_SOFT
            border_color = DANGER
            badge_color = DANGER
            badge_icon = "🔴"
        elif severity == "Moderate":
            bg_color = "#7f2d0d"  # Dark orange background
            border_color = "#ea580c"  # Orange border
            badge_color = "#ea580c"
            badge_icon = "🟠"
        else:  # Mild / No Interaction
            bg_color = "#78350f"  # Dark yellow/amber background
            border_color = WARNING
            badge_color = WARNING
            badge_icon = "🟡"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 0px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Drug pair header
        drugs_row = QHBoxLayout()
        num_lbl = QLabel(f"#{index}")
        num_lbl.setStyleSheet(f"""
            background: {badge_color}; color: white; border-radius: 10px;
            font-size: 11px; font-weight: 700; padding: 2px 8px; border: none;
        """)
        drugs_row.addWidget(num_lbl)

        drug_a = QLabel(interaction.get("drug_a", "?"))
        drug_a.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent; border: none;")
        drugs_row.addWidget(drug_a)

        arrow = QLabel("↔")
        arrow.setStyleSheet(f"font-size: 14px; color: {badge_color}; background: transparent; border: none;")
        drugs_row.addWidget(arrow)

        drug_b = QLabel(interaction.get("drug_b", "?"))
        drug_b.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent; border: none;")
        drugs_row.addWidget(drug_b)
        drugs_row.addStretch()

        conf = interaction.get("detection_confidence", 0.0)
        conf_lbl = QLabel(f"Confidence: {conf:.0%}")
        conf_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent; border: none;")
        drugs_row.addWidget(conf_lbl)

        layout.addLayout(drugs_row)

        # Severity Badge Section
        severity_row = QHBoxLayout()
        severity_row.setSpacing(12)

        severity_label = QLabel(f"{badge_icon} Severity:")
        severity_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        severity_row.addWidget(severity_label)

        severity_badge = QLabel(severity)
        severity_badge.setStyleSheet(f"""
            background: {badge_color}; color: white; border-radius: 8px;
            font-size: 12px; font-weight: 700; padding: 4px 12px; border: none;
        """)
        severity_row.addWidget(severity_badge)

        # Show database and Gemini severity if different
        db_severity = interaction.get("database_severity", "N/A")
        gemini_severity = interaction.get("gemini_severity", "N/A")

        if db_severity != "N/A" and gemini_severity != "N/A" and gemini_severity != db_severity:
            sources_lbl = QLabel(f"(Database: {db_severity} | Gemini: {gemini_severity})")
            sources_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; background: transparent; border: none;")
            severity_row.addWidget(sources_lbl)

        severity_row.addStretch()
        layout.addLayout(severity_row)

        # Separator
        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet(f"background: {border_color}; border: none; margin: 4px 0px;")
        layout.addWidget(sep1)

        # Clinical Reasoning Section
        reason_hdr = QLabel("⚕ Clinical Reasoning")
        reason_hdr.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; background: transparent; border: none;")
        layout.addWidget(reason_hdr)

        reasoning_text = interaction.get("reasoning", interaction.get("description", "No reasoning available."))
        reason_lbl = QLabel(reasoning_text)
        reason_lbl.setWordWrap(True)
        reason_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; line-height: 1.5; background: transparent; border: none;")
        layout.addWidget(reason_lbl)

        # DrugBank source description is shown separately from reasoning.
        description_text = interaction.get("description", "No source description available.")
        if description_text:
            sep_src = QFrame()
            sep_src.setFixedHeight(1)
            sep_src.setStyleSheet(f"background: {border_color}; border: none; margin: 4px 0px;")
            layout.addWidget(sep_src)

            src_hdr = QLabel("📚 DrugBank Evidence")
            src_hdr.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; background: transparent; border: none;")
            layout.addWidget(src_hdr)

            src_lbl = QLabel(description_text)
            src_lbl.setWordWrap(True)
            src_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; line-height: 1.45; background: transparent; border: none;")
            layout.addWidget(src_lbl)

        # DrugBank reference
        db_id = interaction.get("drugbank_id")
        if db_id and db_id != "Unknown":
            sep2 = QFrame()
            sep2.setFixedHeight(1)
            sep2.setStyleSheet(f"background: {border_color}; border: none; margin: 4px 0px;")
            layout.addWidget(sep2)

            ref_lbl = QLabel(f"📚 DrugBank Reference: {db_id}")
            ref_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent; border: none;")
            layout.addWidget(ref_lbl)


class ConfidenceGauge(QFrame):
    """Compact metric display for confidence scores."""
    def __init__(self, label: str, value: float, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 0px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(lbl)

        pct = int(value * 100)
        color = SUCCESS if pct >= 80 else (WARNING if pct >= 50 else DANGER)

        val_lbl = QLabel(f"{pct}%")
        val_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 700; background: transparent; border: none;")
        layout.addWidget(val_lbl)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(pct)
        bar.setStyleSheet(f"""
            QProgressBar {{ background: {BG_DEEP}; border-radius: 3px; height: 4px; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}
        """)
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        layout.addWidget(bar)


class AnalysisWorker(QThread):
    """Runs the backend pipeline in a background thread."""
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, text: str, use_gemini: bool = True):
        super().__init__()
        self.pipeline = pipeline
        self.text = text
        self.use_gemini = use_gemini

    def run(self):
        try:
            result = self.pipeline.process_input(self.text, progress=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Analysis Error: {err_msg}")
            self.error.emit(err_msg)


class OCRWorker(QThread):
    """Runs OCR in a background thread and returns extracted text."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, image_path: str):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit("Extracting text from prescription image…")
            result = self.pipeline.extract_text_from_image(self.image_path)
            if result.get('success'):
                self.finished.emit(result.get('text', ''))
            else:
                self.error.emit(result.get('error', 'OCR failed.'))
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Copilot — Drug Interaction Analyzer")
        self.resize(1100, 760)
        self.setMinimumSize(860, 600)
        self.setObjectName("root")

        self.pipeline = None
        self._worker: AnalysisWorker | None = None
        self._ocr_worker: OCRWorker | None = None
        self._status_loader_timer: QTimer | None = None
        self._status_loader_base = ""
        self._status_loader_color = TEXT_SECONDARY

        self._setup_ui()
        self._load_pipeline()

    # ──────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top Bar ──────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setFixedHeight(62)
        top_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_PANEL};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(28, 0, 28, 0)

        title_lbl = QLabel("⚕  Clinical Copilot")
        title_lbl.setStyleSheet(f"""
            font-size: 17px; font-weight: 800;
            color: {TEXT_PRIMARY}; letter-spacing: 0.3px;
        """)
        top_layout.addWidget(title_lbl)
        top_layout.addStretch()

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY};")
        top_layout.addWidget(self.status_lbl)

        main_layout.addWidget(top_bar)

        # ── Body ─────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background-color: {BG_DEEP};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(24, 24, 24, 24)
        body_layout.setSpacing(20)
        main_layout.addWidget(body)

        # ─ Left panel: Input ─────────────────────────────────────────
        left_panel = QWidget()
        left_panel.setMinimumWidth(390)
        left_panel.setMaximumWidth(460)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        inp_hdr = SectionHeader("Prescription Input", icon="📋")
        left_layout.addWidget(inp_hdr)

        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText(
            "Enter prescription text here…\n\nExamples:\n"
            "• Dolo 650 twice daily for fever\n"
            "• Aspirin 75mg + Clopidogrel 75mg + Atorvastatin 10mg\n"
            "• Metformin 500mg twice daily with Glimepiride 2mg"
        )
        self.input_box.setMinimumHeight(280)
        self.input_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.input_box)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setMinimumWidth(88)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self.clear_btn)

        self.upload_btn = QPushButton("📷  Upload Image")
        self.upload_btn.setObjectName("uploadBtn")
        self.upload_btn.setMinimumWidth(146)
        self.upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_btn.clicked.connect(self._upload_prescription_image)
        btn_row.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton("🔍  Analyze")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.setMinimumWidth(128)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self._analyze)
        self.analyze_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row.addWidget(self.analyze_btn)
        btn_row.setStretch(2, 1)

        left_layout.addLayout(btn_row)

        # Gemini summary toggle
        self.gemini_toggle = QCheckBox("✨  Enable Local LLM Summaries (slower)")
        self.gemini_toggle.setChecked(True)
        self.gemini_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.gemini_toggle.setStyleSheet(f"""
            QCheckBox {{
                color: {TEXT_SECONDARY};
                font-size: 12px;
                font-weight: 600;
                spacing: 8px;
                background: transparent;
                border: none;
                padding: 4px 0px;
            }}
            QCheckBox:hover {{
                color: {TEXT_PRIMARY};
            }}
        """)
        left_layout.addWidget(self.gemini_toggle)

        # Tips
        tip_card = Card(alt=True)
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(14, 12, 14, 12)
        tip_layout.setSpacing(4)
        tip_hdr = QLabel("💡  Tips")
        tip_hdr.setStyleSheet(f"color: {ACCENT}; font-size: 12px; font-weight: 700; background: transparent; border: none;")
        tip_layout.addWidget(tip_hdr)
        for tip in [
            "Multiple drugs can be entered together",
            "OCR/scanned text is also supported"
        ]:
            t = QLabel(f"  • {tip}")
            t.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent; border: none;")
            t.setWordWrap(True)
            tip_layout.addWidget(t)
        left_layout.addWidget(tip_card)
        left_layout.addStretch()

        body_layout.addWidget(left_panel)

        # ─ Right panel: Results ───────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        # Stacked widget: empty / loading / results
        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack)

        # Page 0: empty state
        empty_page = QWidget()
        empty_v = QVBoxLayout(empty_page)
        empty_lbl = QLabel("Enter a prescription on the left and click\n\"Analyze\" to see drug interaction results.")
        empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px; line-height: 1.7;")
        empty_v.addStretch()
        empty_v.addWidget(empty_lbl)
        empty_v.addStretch()
        self.stack.addWidget(empty_page)

        # Page 1: loading
        loading_page = QWidget()
        loading_v = QVBoxLayout(loading_page)
        self.loading_lbl = QLabel("Analyzing prescription…")
        self.loading_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px;")
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)  # indeterminate
        self.loading_bar.setFixedHeight(6)
        self.loading_bar.setTextVisible(False)
        loading_v.addStretch()
        loading_v.addWidget(self.loading_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        loading_v.addSpacing(16)
        loading_v.addWidget(self.loading_bar)
        loading_v.addStretch()
        self.stack.addWidget(loading_page)

        # Page 2: results scroll area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(16)
        self.results_layout.setContentsMargins(0, 0, 8, 0)
        self.results_scroll.setWidget(self.results_container)
        self.stack.addWidget(self.results_scroll)

        body_layout.addWidget(right_panel)

    # ──────────────────────────────────────────────────────────────────
    # Backend Loading
    # ──────────────────────────────────────────────────────────────────

    def _load_pipeline(self):
        """Load the backend pipeline asynchronously."""
        self._update_status("Loading AI models...")
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        QTimer.singleShot(200, self._init_pipeline)

    def _init_pipeline(self):
        try:
            # Change working directory to project root so config.yaml and datasets resolve correctly
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            from clinical_copilot_backend import ClinicalCopilotPipeline
            self.pipeline = ClinicalCopilotPipeline()
            self._update_status("✓ Ready", SUCCESS)
        except Exception as e:
            self._update_status(f"Error loading backend: {e}", DANGER)
        finally:
            self.analyze_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)

    # ──────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────

    def _update_status(self, msg: str, color: str = TEXT_SECONDARY):
        """Update the status label and log to console."""
        self._stop_status_loader()
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {color};")
        
        # Log to appropriate level based on color
        if color == DANGER:
            logger.error(msg)
        elif color == WARNING:
            logger.warning(msg)
        elif color == SUCCESS:
            logger.info(f"SUCCESS: {msg}")
        else:
            logger.info(msg)

    def _start_status_loader(self, base_msg: str, color: str = TEXT_SECONDARY):
        """Animated status text for long-running background operations."""
        self._stop_status_loader()
        self._status_loader_base = base_msg
        self._status_loader_color = color

        self._status_loader_timer = QTimer(self)
        self._status_loader_timer.setInterval(280)

        def _tick():
            if self._status_loader_timer is None:
                return
            dots = ((self._status_loader_timer.property("dot_count") or 0) + 1) % 4
            self._status_loader_timer.setProperty("dot_count", dots)
            suffix = "." * dots
            self.status_lbl.setText(f"{self._status_loader_base}{suffix}")
            self.status_lbl.setStyleSheet(f"font-size: 12px; color: {self._status_loader_color};")

        self._status_loader_timer.timeout.connect(_tick)
        self._status_loader_timer.setProperty("dot_count", 0)
        _tick()
        self._status_loader_timer.start()

    def _stop_status_loader(self):
        if self._status_loader_timer is not None:
            self._status_loader_timer.stop()
            self._status_loader_timer.deleteLater()
            self._status_loader_timer = None

    @staticmethod
    def _is_temporary_service_delay(msg: str) -> bool:
        lowered = (msg or "").lower()
        patterns = [
            "resource_exhausted",
            "quota",
            "retry in",
            "rate-limited",
            "disabled for this app session",
            "temporarily",
        ]
        return any(p in lowered for p in patterns)

    def _clear(self):
        self.input_box.clear()
        self.stack.setCurrentIndex(0)
        self._update_status("✓ Ready", SUCCESS)

    def _upload_prescription_image(self):
        if self.pipeline is None:
            self._update_status("Backend not loaded. Please wait.", WARNING)
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prescription Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not image_path:
            return

        self.upload_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self._start_status_loader("Extracting text from image")

        self._ocr_worker = OCRWorker(self.pipeline, image_path)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_result)
        self._ocr_worker.error.connect(self._on_ocr_error)
        self._ocr_worker.start()

    def _analyze(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return

        if self.pipeline is None:
            self._update_status("Backend not loaded. Please wait.", WARNING)
            return

        logger.info(f"Starting analysis for input: {text[:50]}{'...' if len(text) > 50 else ''}")
        self.analyze_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.stack.setCurrentIndex(1)  # loading
        self._start_status_loader("Analyzing prescription")

        self._worker = AnalysisWorker(self.pipeline, text, use_gemini=self.gemini_toggle.isChecked())
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.start()

    def _on_progress(self, msg: str):
        self.loading_lbl.setText(msg)
        # Progress messages are already logged by the worker

    def _on_ocr_progress(self, msg: str):
        self._start_status_loader(msg.replace("…", ""), TEXT_SECONDARY)

    def _on_ocr_result(self, text: str):
        self._ocr_worker = None
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        self.input_box.setPlainText(text)
        self._update_status("✓ OCR complete. Review text and click Analyze.", SUCCESS)

    def _on_ocr_error(self, msg: str):
        self._ocr_worker = None
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        if self._is_temporary_service_delay(msg):
            self._update_status("Image text service is temporarily busy. Please try again shortly.", WARNING)
            return
        self._update_status(f"OCR Error: {msg}", DANGER)

    def _on_result(self, result: dict):
        self._worker = None
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        interactions = result.get("drug_interactions", [])
        n_int = len(interactions)
        has_concern = result.get("metadata", {}).get("has_safety_concerns", False)
        alert = f"⚠ {n_int} interaction(s) found" if has_concern else "✓ No interactions"
        col = DANGER if has_concern else SUCCESS
        
        self._update_status(alert, col)
        
        # Log a summary of the results
        logger.info(f"Analysis complete: {len(result.get('brand_generic_mappings', []))} drugs found, {n_int} interactions detected.")
        
        self._populate_results(result)
        self.stack.setCurrentIndex(2)

    def _on_error(self, msg: str):
        self._worker = None
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        if self._is_temporary_service_delay(msg):
            self._update_status("Some AI enhancements are warming up. Core safety analysis remains available.", WARNING)
        else:
            self._update_status(f"Error: {msg}", DANGER)
        self.stack.setCurrentIndex(0)

    # ──────────────────────────────────────────────────────────────────
    # Result Rendering
    # ──────────────────────────────────────────────────────────────────

    def _clear_results(self):
        """Remove all previously rendered result widgets."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_results(self, result: dict):
        self._clear_results()

        mappings      = result.get("brand_generic_mappings", [])
        conf_scores   = result.get("entity_confidence_scores", [])
        interactions  = result.get("drug_interactions", [])
        conf_metrics  = result.get("confidence_metrics", {})
        metadata      = result.get("metadata", {})

        # ── Safety Alert Banner ───────────────────────────────────────
        if metadata.get("has_safety_concerns"):
            alert_card = QFrame()
            alert_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {DANGER_SOFT};
                    border: 1.5px solid {DANGER};
                    border-radius: 10px;
                    padding: 0px;
                }}
            """)
            alert_layout = QHBoxLayout(alert_card)
            alert_layout.setContentsMargins(18, 12, 18, 12)
            icon_lbl = QLabel("⚠")
            icon_lbl.setStyleSheet("font-size: 20px; background: transparent; border: none;")
            alert_layout.addWidget(icon_lbl)
            msg_lbl = QLabel(
                f"<b>Safety Alert:</b> {len(interactions)} drug-drug interaction(s) detected. "
                f"Review interactions carefully before dispensing."
            )
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet("color: #fecaca; font-size: 13px; background: transparent; border: none;")
            alert_layout.addWidget(msg_lbl, stretch=1)
            self.results_layout.addWidget(alert_card)

        # ── Identified Medicines ──────────────────────────────────────
        med_hdr = SectionHeader(
            f"Identified Medicines ({len(mappings)})", icon="💊"
        )
        self.results_layout.addWidget(med_hdr)

        for i, mapping in enumerate(mappings):
            conf = conf_scores[i] if i < len(conf_scores) else 0.5
            tile = MedicineTile(mapping, conf)
            self.results_layout.addWidget(tile)

        # ── Drug Interactions ─────────────────────────────────────────
        int_hdr = SectionHeader(
            f"Drug-Drug Interactions ({len(interactions)})", icon="⚡"
        )
        self.results_layout.addWidget(int_hdr)

        if interactions:
            for idx, interaction in enumerate(interactions, 1):
                card = InteractionCard(interaction, idx)
                self.results_layout.addWidget(card)
        else:
            no_int = QLabel("✓  No drug-drug interactions detected for this prescription.")
            no_int.setStyleSheet(f"""
                color: {SUCCESS}; font-size: 13px; padding: 16px;
                background: {BG_CARD}; border-radius: 10px;
                border: 1px solid {BORDER};
            """)
            self.results_layout.addWidget(no_int)

        # ── Confidence Metrics ────────────────────────────────────────
        conf_hdr = SectionHeader("Confidence Metrics", icon="📊")
        self.results_layout.addWidget(conf_hdr)

        gauges_row = QHBoxLayout()
        gauges_row.setSpacing(10)

        entity_conf  = conf_metrics.get("entity_extraction", {}).get("confidence", 0.0)
        int_conf     = conf_metrics.get("interaction_detection", 0.0)
        overall_conf = conf_metrics.get("overall_confidence", 0.0)

        for label, val in [
            ("Entity Extraction", entity_conf),
            ("Interaction Detection", int_conf),
            ("Overall Confidence", overall_conf),
        ]:
            gauge = ConfidenceGauge(label, val)
            gauges_row.addWidget(gauge)

        conf_container = QWidget()
        conf_container.setLayout(gauges_row)
        self.results_layout.addWidget(conf_container)

        # ── Summary ───────────────────────────────────────────────────
        summ_hdr = SectionHeader("Summary", icon="📄")
        self.results_layout.addWidget(summ_hdr)

        summ_card = Card()
        summ_inner = QGridLayout(summ_card)
        summ_inner.setContentsMargins(18, 14, 18, 14)
        summ_inner.setSpacing(8)

        def summ_row(row, key, val, color=TEXT_PRIMARY):
            k = QLabel(key)
            k.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent; border: none;")
            v = QLabel(str(val))
            v.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600; background: transparent; border: none;")
            summ_inner.addWidget(k, row, 0)
            summ_inner.addWidget(v, row, 1)

        low_conf = metadata.get("low_confidence_entities", 0)
        needs_review = low_conf > 0 or metadata.get("has_safety_concerns", False)

        summ_row(0, "Total Medicines Identified:", metadata.get("total_medicines", 0))
        summ_row(1, "Total Interactions:", metadata.get("total_interactions", 0),
                 DANGER if metadata.get("has_safety_concerns") else SUCCESS)
        summ_row(2, "Low Confidence Entities:", low_conf,
                 WARNING if low_conf > 0 else SUCCESS)
        summ_row(3, "ML (NER) Enabled:", "Yes" if metadata.get("ml_enabled") else "No")
        summ_row(4, "Semantic Embeddings:", "Yes" if metadata.get("embeddings_enabled") else "No")
        summ_row(5, "Requires Review:", "YES" if needs_review else "NO",
                 DANGER if needs_review else SUCCESS)

        methods = result.get("extraction_methods", [])
        if methods:
            summ_row(6, "Extraction Methods:", ", ".join(methods))

        self.results_layout.addWidget(summ_card)
        self.results_layout.addStretch()


# ─── Entry Point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # Use a clean system font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
