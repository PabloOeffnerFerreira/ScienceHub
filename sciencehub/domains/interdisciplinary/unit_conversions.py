"""
Universal Unit Converter (ScienceHub)
====================================

Flagship, dataset-driven unit converter with:
- Category + unit selection (supports *big* conversion_data dict)
- Full SI-prefix integration (via external si-prefix module)
- Bidirectional live conversion (edit input OR output)
- Temperature offsets (C/K/F)
- Data size (bit/byte) logic
- Smart formatting (scientific / engineering / fixed), sig figs
- History + Favorites (pin), copy buttons, swap
- Magnitude visualization bar (log scale) + "scale sense" readout
- Beautiful modern UI (dark, gradients, micro-animations)

Important:
- This file contains NO conversion tables and NO SI prefix tables.
- It loads conversion tables from sciencehub/data/datasets (like your conversion_data dict) :contentReference[oaicite:0]{index=0}
- It imports SI-prefix logic from your si-prefix.py module.

Author: ScienceHub Team
"""

from __future__ import annotations

import math
import json
import os
import time
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    Qt,
    QTimer,
    QPropertyAnimation,
    QEasingCurve,
    pyqtSignal,
    QSize,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QBrush,
    QLinearGradient,
    QClipboard,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QFrame,
    QSplitter,
    QScrollArea,
    QCheckBox,
    QSlider,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTabWidget,
    QSizePolicy,
    QApplication
)

# ScienceHub base
from sciencehub.ui.components.tool_base import ScienceHubTool


# -----------------------------
# Helpers: module loading
# -----------------------------

def _try_import_module(dotted: str):
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def _import_module_from_path(module_name: str, file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception:
        return None


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------
# Data contract + special cases
# -----------------------------

class ConversionError(Exception):
    pass


@dataclass
class ConversionRequest:
    category: str
    value: float
    from_unit: str
    to_unit: str
    from_prefix: str = ""
    to_prefix: str = ""


@dataclass
class ConversionResult:
    value: float
    base_value_si: float  # value expressed in SI base (as defined by dataset)
    notes: str = ""


# -----------------------------
# SI Prefix Adapter (external)
# -----------------------------

class SIPrefixAdapter:
    """
    Wraps external si-prefix.py behind a stable interface.

    Supports these module patterns:
      - PREFIXES / SI_PREFIXES dict: {key -> factor}
      - PREFIXES / SI_PREFIXES list: [{"name","symbol","factor"}, ...]
      - functions: prefix_factor|factor|get_factor, apply_prefix
    """

    def __init__(self, module: Any):
        self.mod = module
        self._prefix_map = self._build_prefix_map()

    def _build_prefix_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns mapping:
          key -> {"factor": float, "symbol": str, "name": str}
        Key is what we will pass into your module, if possible.
        """
        # 1) List-of-dicts style
        for attr in ("PREFIXES", "SI_PREFIXES", "prefixes", "si_prefixes"):
            obj = getattr(self.mod, attr, None)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                out = {}
                for item in obj:
                    name = str(item.get("name", "")).strip()
                    sym = str(item.get("symbol", "")).strip()
                    factor = float(item.get("factor", 1.0))
                    key = name or sym
                    if not key:
                        continue
                    out[key] = {"factor": factor, "symbol": sym or name, "name": name or sym}
                if "" not in out:
                    out[""] = {"factor": 1.0, "symbol": "", "name": ""}
                return out

        # 2) Dict style
        for attr in ("PREFIXES", "SI_PREFIXES", "prefixes", "si_prefixes"):
            d = getattr(self.mod, attr, None)
            if isinstance(d, dict) and d:
                out = {}
                for k, v in d.items():
                    key = str(k)
                    out[key] = {"factor": float(v), "symbol": key, "name": key}

                out.setdefault("", {"factor": 1.0, "symbol": "", "name": ""})

                # micro compatibility across Unicode variants and ASCII fallback:
                # "μ" (Greek mu) vs "µ" (micro sign) vs "u"
                if "μ" in out:
                    out.setdefault("µ", {"factor": out["μ"]["factor"], "symbol": "µ", "name": "micro"})
                    out.setdefault("u", {"factor": out["μ"]["factor"], "symbol": "u", "name": "micro"})
                elif "µ" in out:
                    out.setdefault("μ", {"factor": out["µ"]["factor"], "symbol": "μ", "name": "micro"})
                    out.setdefault("u", {"factor": out["µ"]["factor"], "symbol": "u", "name": "micro"})
                elif "u" in out:
                    out.setdefault("μ", {"factor": out["u"]["factor"], "symbol": "μ", "name": "micro"})
                    out.setdefault("µ", {"factor": out["u"]["factor"], "symbol": "µ", "name": "micro"})

                return out

        # 3) Fallback (still no hardcoding SI tables; just minimal)
        return {"": {"factor": 1.0, "symbol": "", "name": ""}}

    def prefix_factor(self, key: str) -> float:
        key = key or ""

        # Try module functions first (if they exist)
        for fn_name in ("prefix_factor", "factor", "get_factor"):
            fn = getattr(self.mod, fn_name, None)
            if callable(fn):
                try:
                    return float(fn(key))
                except Exception:
                    pass

        # Then our discovered map
        if key in self._prefix_map:
            return float(self._prefix_map[key]["factor"])

        # Micro alias fallback
        if key == "µ" and "u" in self._prefix_map:
            return float(self._prefix_map["u"]["factor"])
        if key == "u" and "µ" in self._prefix_map:
            return float(self._prefix_map["µ"]["factor"])

        raise ConversionError(f"Unknown SI prefix key '{key}'")

    def apply_prefix(self, value: float, key: str) -> float:
        key = key or ""

        fn = getattr(self.mod, "apply_prefix", None)
        if callable(fn):
            try:
                return float(fn(value, key))
            except Exception:
                pass

        return float(value) * self.prefix_factor(key)

    def prefix_options(self) -> List[Tuple[str, str]]:
        """
        Returns list of (display, key) pairs.
        display is what user sees; key is what we pass to module.
        """
        items = list(self._prefix_map.items())

        # Sort by factor if possible
        try:
            items.sort(key=lambda kv: float(kv[1].get("factor", 1.0)))
        except Exception:
            items.sort(key=lambda kv: kv[0])

        options = []
        for key, meta in items:
            sym = str(meta.get("symbol", key))
            if key == "":
                options.append(("—", ""))  # none
            else:
                # show symbol if it looks like one, otherwise show the key
                options.append((sym if sym else key, key))
        return options

# -----------------------------
# Conversion Controller (no tables here)
# -----------------------------

class ConversionController:
    """
    Core logic glue:
    - Uses conversion_data loaded externally (dict with categories/units/SI list)
    - Uses external si-prefix adapter
    - Handles special categories (Temperature, Data Size)
    """

    def __init__(self, conversion_data: Dict[str, Any], si_prefix: SIPrefixAdapter):
        self.data = conversion_data
        self.si = si_prefix

    def categories(self) -> List[str]:
        return sorted(list(self.data.keys()))

    def units_for(self, category: str) -> List[str]:
        try:
            units = list(self.data[category]["units"].keys())
            units.sort()
            return units
        except Exception:
            return []

    def si_units_for(self, category: str) -> List[str]:
        try:
            si_list = self.data[category].get("SI", [])
            return list(si_list) if isinstance(si_list, list) else []
        except Exception:
            return []

    def _is_temperature(self, category: str) -> bool:
        return category.strip().lower() in ("temperature",)

    def _is_data_size(self, category: str) -> bool:
        return category.strip().lower() in ("data size", "datasize", "data")

    def supports_prefix(self, category: str, unit: str) -> bool:
        """
        Conservative:
        - allow prefixes if unit is listed as SI (dataset SI list)
        - or unit matches common SI shapes (m, g, s, A, V, W, J, Pa, Hz, N, T, lux...)
        - do NOT allow for Temperature or Data Size by default
        """
        if self._is_temperature(category) or self._is_data_size(category):
            return False

        si_units = set(self.si_units_for(category))
        if unit in si_units:
            return True

        # Some datasets might list SI as ["m³","L"] etc. We'll allow prefix for those too.
        # If it's an SI-looking unit string, allow.
        suspicious_non_si = {"mile", "yard", "foot", "inch", "lb", "oz", "psi", "atm", "mmHg", "torr", "cal", "kcal", "Wh", "kWh", "BTU", "hp", "rpm"}
        if unit in suspicious_non_si:
            return False

        # allow if it contains SI tokens and is short-ish
        if any(unit.startswith(x) for x in ("m", "g", "s", "A", "V", "W", "J", "Pa", "Hz", "N", "T", "lux", "cd", "rad")):
            return True

        return False

    def convert(self, req: ConversionRequest) -> ConversionResult:
        if req.category not in self.data:
            raise ConversionError(f"Unknown category '{req.category}'")

        if self._is_temperature(req.category):
            out, base = self._convert_temperature(req.value, req.from_unit, req.to_unit)
            return ConversionResult(value=out, base_value_si=base, notes="Temperature conversion uses offsets.")

        if self._is_data_size(req.category):
            out, base = self._convert_data_size(req.value, req.from_unit, req.to_unit)
            return ConversionResult(value=out, base_value_si=base, notes="Data size conversion uses bit↔byte ×8 rule.")

        units_map = self.data[req.category].get("units", {})
        if req.from_unit not in units_map or req.to_unit not in units_map:
            raise ConversionError("Invalid unit selection")

        # Dataset convention: factor is "unit in SI base units"
        # Example: Length units map to meters, Mass maps to kg (your Mass uses kg=1, g=0.001)
        factor_from = float(units_map[req.from_unit])
        factor_to = float(units_map[req.to_unit])

        # Convert to dataset SI base
        base_value = req.value * factor_from

        # Apply prefix on the *from* side if enabled
        if req.from_prefix and self.supports_prefix(req.category, req.from_unit):
            base_value = self.si.apply_prefix(base_value, req.from_prefix)

        # Convert to target unit
        out_value = base_value / factor_to

        # Apply target prefix (divide) if enabled
        if req.to_prefix and self.supports_prefix(req.category, req.to_unit):
            out_value = out_value / self.si.prefix_factor(req.to_prefix)

        return ConversionResult(value=out_value, base_value_si=base_value)

    # ---- Special conversions ----

    def _convert_temperature(self, value: float, from_u: str, to_u: str) -> Tuple[float, float]:
        from_u = from_u.strip()
        to_u = to_u.strip()
        valid = {"C", "K", "F"}
        if from_u not in valid or to_u not in valid:
            raise ConversionError("Temperature units must be C, K, or F")

        # Convert to K as base
        if from_u == "K":
            k = value
        elif from_u == "C":
            k = value + 273.15
        else:  # F
            k = (value - 32.0) * (5.0 / 9.0) + 273.15

        # Convert K to target
        if to_u == "K":
            out = k
        elif to_u == "C":
            out = k - 273.15
        else:  # F
            out = (k - 273.15) * (9.0 / 5.0) + 32.0

        return out, k

    def _convert_data_size(self, value: float, from_u: str, to_u: str) -> Tuple[float, float]:
        from_u = from_u.strip()
        to_u = to_u.strip()
        valid = {"bit", "byte"}
        if from_u not in valid or to_u not in valid:
            raise ConversionError("Data Size units must be 'bit' or 'byte'")

        # base: bit
        if from_u == "bit":
            bits = value
        else:
            bits = value * 8.0

        if to_u == "bit":
            out = bits
        else:
            out = bits / 8.0

        return out, bits


# -----------------------------
# Formatting + magnitude sense
# -----------------------------

class NumberFormatter:
    def __init__(self):
        self.mode = "smart"  # smart | fixed | scientific | engineering
        self.sig_figs = 6
        self.decimals = 6

    def format(self, x: float) -> str:
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "∞" if x > 0 else "-∞"

        mode = self.mode

        if mode == "fixed":
            return f"{x:.{self.decimals}f}"
        if mode == "scientific":
            return f"{x:.{self.sig_figs}e}"
        if mode == "engineering":
            return self._format_engineering(x, self.sig_figs)

        # smart
        ax = abs(x)
        if ax != 0 and (ax >= 1e6 or ax < 1e-4):
            return f"{x:.{self.sig_figs}e}"
        # show fewer decimals if "nice"
        if ax >= 1000:
            return f"{x:,.{min(2, self.decimals)}f}"
        if ax >= 1:
            return f"{x:.{min(4, self.decimals)}f}"
        return f"{x:.{self.decimals}f}"

    def _format_engineering(self, x: float, sig: int) -> str:
        if x == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
        mant = x / (10 ** exp)
        return f"{mant:.{sig}f}e{exp:+d}"


def _log10_safe(x: float) -> float:
    x = abs(x)
    if x <= 0:
        return -float("inf")
    return math.log10(x)


def magnitude_label(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "zero"
    e = math.floor(math.log10(ax))
    # human-ish buckets
    if e <= -18:
        return "attoscale"
    if e <= -12:
        return "picoscale"
    if e <= -9:
        return "nanoscale"
    if e <= -6:
        return "microscale"
    if e <= -3:
        return "milliscale"
    if e <= 2:
        return "human scale"
    if e <= 6:
        return "macroscale"
    if e <= 12:
        return "megastructure"
    return "astronomical"


# -----------------------------
# UI: Magnitude Bar Widget
# -----------------------------

class MagnitudeBar(QWidget):
    """
    Shows a log-scale bar with a moving marker indicating magnitude of value.
    Purely aesthetic + intuition builder.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self._log_value = 0.0
        self._label = "human scale"

    def set_value(self, value: float):
        lv = _log10_safe(value)
        if not math.isfinite(lv):
            lv = -12.0
        self._log_value = max(-24.0, min(24.0, lv))
        self._label = magnitude_label(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(10, 10, -10, -10)

        # background
        bg = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        bg.setColorAt(0.0, QColor("#121827"))
        bg.setColorAt(1.0, QColor("#0b1220"))
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 14, 14)

        # scale line
        line_y = rect.center().y() + 8
        painter.setPen(QPen(QColor("#2b3a5a"), 3))
        painter.drawLine(rect.left() + 18, line_y, rect.right() - 18, line_y)

        # ticks
        painter.setPen(QPen(QColor("#2b3a5a"), 2))
        for e in range(-24, 25, 6):
            x = self._map_exp_to_x(rect, e)
            painter.drawLine(int(x), line_y - 10, int(x), line_y + 10)

        # marker
        x_marker = self._map_exp_to_x(rect, self._log_value)
        painter.setPen(QPen(QColor("#7cdbff"), 3))
        painter.setBrush(QBrush(QColor("#7cdbff")))
        painter.drawEllipse(int(x_marker) - 7, line_y - 22, 14, 14)

        # label
        painter.setPen(QPen(QColor("#dbe7ff"), 1))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(rect.left() + 18, rect.top() + 20, "Magnitude (log₁₀)")
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QPen(QColor("#9fb3d9"), 1))
        painter.drawText(rect.left() + 18, rect.top() + 42, f"Scale feel: {self._label}")

        painter.end()

    def _map_exp_to_x(self, rect, exp: float) -> float:
        # map exp in [-24, 24] → [left,right]
        left = rect.left() + 18
        right = rect.right() - 18
        t = (exp + 24.0) / 48.0
        return left + t * (right - left)


# -----------------------------
# UI: Fancy pill button
# -----------------------------

class PillButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)


# -----------------------------
# History items
# -----------------------------

@dataclass
class HistoryItem:
    ts: float
    category: str
    value_in: float
    from_unit: str
    from_prefix: str
    value_out: float
    to_unit: str
    to_prefix: str

    def title(self) -> str:
        fp = self.from_prefix or ""
        tp = self.to_prefix or ""
        return f"{self.category}: {self.value_in:g} {fp}{self.from_unit} → {self.value_out:g} {tp}{self.to_unit}"

    def subtitle(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))


# -----------------------------
# Main Tool UI
# -----------------------------

class UnitConverterTool(ScienceHubTool):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._loading_ok = False
        self._busy = False
        self._last_direction = "in_to_out"  # or out_to_in

        self.formatter = NumberFormatter()
        self.history: List[HistoryItem] = []
        self.favorites: List[HistoryItem] = []
        self._last_history_sig: Optional[tuple] = None

        # Load external dataset + SI module
        try:
            conversion_data = self._load_conversion_data()
            si_adapter = self._load_si_prefix_adapter()
            self.controller = ConversionController(conversion_data, si_adapter)
            self._loading_ok = True
        except Exception as e:
            self._loading_ok = False
            self.controller = None  # type: ignore
            QMessageBox.warning(self, "Unit Converter", f"Failed to load datasets/modules:\n{e}")

        self._build_ui()

        if self._loading_ok:
            self._populate_categories()
            QTimer.singleShot(0, self._initial_select)

        # small entrance animation
        QTimer.singleShot(50, self._animate_header)

    # -----------------------------
    # Loaders (NO tables here)
    # -----------------------------

    def _load_conversion_data(self) -> Dict[str, Any]:
        """
        Looks for conversion data in sciencehub/data/datasets.
        Supports:
          - python file exporting `conversion_data` dict
          - json file exporting equivalent structure
        """
        base = os.path.join("sciencehub", "data", "datasets")

        candidates = [
            os.path.join(base, "conversion_data.py"),
            os.path.join(base, "unit_conversions.py"),
            os.path.join(base, "conversion_data.json"),
            os.path.join(base, "unit_conversions.json"),
        ]

        # 1) python module import by dotted path
        mod = _try_import_module("sciencehub.data.datasets.conversion_data")
        if mod and hasattr(mod, "conversion_data"):
            data = getattr(mod, "conversion_data")
            if isinstance(data, dict):
                return data

        # 2) try load from file path
        for path in candidates:
            if path.endswith(".py"):
                m = _import_module_from_path("_sciencehub_conversion_data", path)
                if m and hasattr(m, "conversion_data") and isinstance(getattr(m, "conversion_data"), dict):
                    return getattr(m, "conversion_data")
            elif path.endswith(".json"):
                j = _read_json(path)
                if isinstance(j, dict):
                    # if file wraps
                    if "conversion_data" in j and isinstance(j["conversion_data"], dict):
                        return j["conversion_data"]
                    return j

        raise FileNotFoundError(
            "Could not find conversion data. Expected one of:\n"
            "sciencehub/data/datasets/conversion_data.py (with conversion_data dict)\n"
            "sciencehub/data/datasets/conversion_data.json"
        )

    def _load_si_prefix_adapter(self) -> SIPrefixAdapter:
        """
        Tries multiple conventions for your si-prefix.py.
        """
        # dotted guesses
        for dotted in (
            "sciencehub.utils.si_prefix",
            "sciencehub.utils.si_prefixes",
            "sciencehub.data.datasets.si_prefix",
            "sciencehub.data.datasets.si_prefixes",
        ):
            mod = _try_import_module(dotted)
            if mod:
                return SIPrefixAdapter(mod)

        # file guesses
        base_utils = os.path.join("sciencehub", "utils")
        base_data = os.path.join("sciencehub", "data", "datasets")
        candidates = [
            os.path.join(base_utils, "si-prefix.py"),
            os.path.join(base_utils, "si_prefix.py"),
            os.path.join(base_utils, "si_prefixes.py"),
            os.path.join(base_data, "si-prefix.py"),
            os.path.join(base_data, "si_prefix.py"),
            os.path.join(base_data, "si_prefixes.py"),
        ]
        for path in candidates:
            mod = _import_module_from_path("_sciencehub_si_prefix", path)
            if mod:
                return SIPrefixAdapter(mod)

        raise FileNotFoundError(
            "Could not find SI prefix module. Expected one of:\n"
            "sciencehub/utils/si-prefix.py or si_prefix.py\n"
            "or an importable module sciencehub.utils.si_prefix"
        )

    # -----------------------------
    # UI construction
    # -----------------------------

    def _build_ui(self):
        root = QVBoxLayout()
        root.setSpacing(14)
        self.root_layout.addLayout(root)

        # Header
        header = QFrame()
        header.setObjectName("ucHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 16)

        self.title_lbl = QLabel("Universal Unit Converter")
        self.title_lbl.setObjectName("ucTitle")
        self.subtitle_lbl = QLabel("Convert anything. Feel the scale.")
        self.subtitle_lbl.setObjectName("ucSubtitle")

        title_wrap = QVBoxLayout()
        title_wrap.addWidget(self.title_lbl)
        title_wrap.addWidget(self.subtitle_lbl)

        header_layout.addLayout(title_wrap)
        header_layout.addStretch()

        self.live_checkbox = QCheckBox("Live")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.stateChanged.connect(self._on_any_change)

        self.swap_btn = PillButton("Swap")
        self.swap_btn.clicked.connect(self._swap_units)

        self.copy_btn = PillButton("Copy Output")
        self.copy_btn.clicked.connect(self._copy_output)

        header_layout.addWidget(self.live_checkbox)
        header_layout.addWidget(self.swap_btn)
        header_layout.addWidget(self.copy_btn)

        root.addWidget(header)

        # Main splitter: left controls, right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizes([380, 820])
        root.addWidget(splitter, 1)

        # Left: category + history/favorites
        left = self._build_left_panel()
        splitter.addWidget(left)

        # Right: converter + visuals
        right = self._build_right_panel()
        splitter.addWidget(right)

        # Style
        self._apply_style()

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ucLeftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Category card
        cat_group = QGroupBox("Category")
        cat_group.setObjectName("toolCard")
        cat_layout = QVBoxLayout(cat_group)

        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.category_combo)

        self.si_hint = QLabel("SI base: —")
        self.si_hint.setObjectName("ucHint")
        cat_layout.addWidget(self.si_hint)

        layout.addWidget(cat_group)

        # Formatting card
        fmt_group = QGroupBox("Formatting")
        fmt_group.setObjectName("toolCard")
        fmt_layout = QGridLayout(fmt_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["smart", "fixed", "scientific", "engineering"])
        self.mode_combo.currentTextChanged.connect(self._on_format_changed)

        self.sig_slider = QSlider(Qt.Orientation.Horizontal)
        self.sig_slider.setRange(2, 12)
        self.sig_slider.setValue(self.formatter.sig_figs)
        self.sig_slider.valueChanged.connect(self._on_format_changed)

        self.dec_slider = QSlider(Qt.Orientation.Horizontal)
        self.dec_slider.setRange(0, 12)
        self.dec_slider.setValue(self.formatter.decimals)
        self.dec_slider.valueChanged.connect(self._on_format_changed)

        fmt_layout.addWidget(QLabel("Mode"), 0, 0)
        fmt_layout.addWidget(self.mode_combo, 0, 1)
        fmt_layout.addWidget(QLabel("Sig figs"), 1, 0)
        fmt_layout.addWidget(self.sig_slider, 1, 1)
        fmt_layout.addWidget(QLabel("Decimals"), 2, 0)
        fmt_layout.addWidget(self.dec_slider, 2, 1)

        layout.addWidget(fmt_group)

        # Tabs: History / Favorites
        tabs = QTabWidget()
        tabs.setObjectName("ucTabs")

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._apply_history_item)

        self.fav_list = QListWidget()
        self.fav_list.itemClicked.connect(self._apply_favorite_item)

        tabs.addTab(self.history_list, "History")
        tabs.addTab(self.fav_list, "Favorites")

        layout.addWidget(tabs, 1)

        # Actions
        btn_row = QHBoxLayout()
        self.pin_btn = PillButton("Pin")
        self.pin_btn.clicked.connect(self._pin_current)

        self.clear_hist_btn = PillButton("Clear")
        self.clear_hist_btn.clicked.connect(self._clear_history)

        btn_row.addWidget(self.pin_btn)
        btn_row.addWidget(self.clear_hist_btn)
        layout.addLayout(btn_row)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ucRightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Converter card
        converter = QGroupBox("Convert")
        converter.setObjectName("toolCard")
        grid = QGridLayout(converter)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # Input side
        self.in_value = QLineEdit()
        self.in_value.setPlaceholderText("Enter value…")
        self.in_value.textEdited.connect(self._on_input_value_edited)

        self.in_unit = QComboBox()
        self.in_unit.currentTextChanged.connect(self._on_any_change)

        self.in_prefix = QComboBox()
        self.in_prefix.currentTextChanged.connect(self._on_any_change)

        # Output side
        self.out_value = QLineEdit()
        self.out_value.setPlaceholderText("Result…")
        self.out_value.textEdited.connect(self._on_output_value_edited)

        self.out_unit = QComboBox()
        self.out_unit.currentTextChanged.connect(self._on_any_change)

        self.out_prefix = QComboBox()
        self.out_prefix.currentTextChanged.connect(self._on_any_change)

        # Labels
        grid.addWidget(QLabel("Input"), 0, 0)
        grid.addWidget(QLabel("Output"), 0, 2)

        # Rows
        left_box = QVBoxLayout()
        left_box.addWidget(self.in_value)
        urow1 = QHBoxLayout()
        urow1.addWidget(self.in_prefix)
        urow1.addWidget(self.in_unit)
        left_box.addLayout(urow1)

        right_box = QVBoxLayout()
        right_box.addWidget(self.out_value)
        urow2 = QHBoxLayout()
        urow2.addWidget(self.out_prefix)
        urow2.addWidget(self.out_unit)
        right_box.addLayout(urow2)

        grid.addLayout(left_box, 1, 0)
        eq = QLabel("⟷")
        eq.setObjectName("ucEquals")
        eq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(eq, 1, 1)
        grid.addLayout(right_box, 1, 2)

        # Notes + quick actions
        self.note_lbl = QLabel("Ready.")
        self.note_lbl.setObjectName("ucHint")
        grid.addWidget(self.note_lbl, 2, 0, 1, 3)

        layout.addWidget(converter)

        # Visuals card
        visuals = QGroupBox("Scale Intuition")
        visuals.setObjectName("toolCard")
        vlayout = QVBoxLayout(visuals)

        self.mag_bar = MagnitudeBar()
        vlayout.addWidget(self.mag_bar)

        self.scale_lbl = QLabel("—")
        self.scale_lbl.setObjectName("ucScaleText")
        vlayout.addWidget(self.scale_lbl)

        layout.addWidget(visuals)

        # Footer: action buttons
        action_row = QHBoxLayout()
        self.calc_btn = PillButton("Calculate")
        self.calc_btn.clicked.connect(self.calculate)

        self.random_btn = PillButton("Surprise me")
        self.random_btn.clicked.connect(self._surprise_me)

        self.reset_btn = PillButton("Reset")
        self.reset_btn.clicked.connect(self._reset)

        action_row.addWidget(self.calc_btn)
        action_row.addWidget(self.random_btn)
        action_row.addStretch()
        action_row.addWidget(self.reset_btn)

        layout.addLayout(action_row)

        return panel

    def _apply_style(self):
        # A modern dark theme with gradients + soft borders.
        self.setStyleSheet("""
            #ucHeader {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0b1220, stop:1 #111c33);
                border: 1px solid #1e2a46;
                border-radius: 18px;
            }
            #ucTitle {
                color: #eaf2ff;
                font-size: 22px;
                font-weight: 700;
            }
            #ucSubtitle {
                color: #9fb3d9;
                font-size: 12px;
            }

            #ucLeftPanel, #ucRightPanel {
                background: transparent;
            }

            QGroupBox#toolCard, QGroupBox {
                color: #dbe7ff;
                font-weight: 600;
                border: 1px solid #1e2a46;
                border-radius: 16px;
                margin-top: 12px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0b1220, stop:1 #0f1930);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }

            QLabel {
                color: #dbe7ff;
            }
            QLabel#ucHint {
                color: #9fb3d9;
                font-weight: 400;
            }
            QLabel#ucEquals {
                color: #7cdbff;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#ucScaleText {
                color: #cfe0ff;
                font-weight: 500;
            }

            QLineEdit {
                background: #0a0f1d;
                border: 1px solid #1e2a46;
                border-radius: 12px;
                padding: 10px 12px;
                color: #eaf2ff;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #3d6bff;
            }

            QComboBox {
                background: #0a0f1d;
                border: 1px solid #1e2a46;
                border-radius: 12px;
                padding: 8px 10px;
                color: #eaf2ff;
            }

            QTabWidget::pane {
                border: 1px solid #1e2a46;
                border-radius: 14px;
                background: #0b1220;
            }
            QTabBar::tab {
                background: #0b1220;
                border: 1px solid #1e2a46;
                padding: 8px 10px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 6px;
                color: #9fb3d9;
            }
            QTabBar::tab:selected {
                color: #eaf2ff;
                border-bottom-color: #0b1220;
            }

            QListWidget {
                background: #0a0f1d;
                border: 1px solid #1e2a46;
                border-radius: 12px;
                padding: 8px;
                color: #eaf2ff;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 10px;
            }
            QListWidget::item:selected {
                background: #152447;
            }

            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #1a2a54, stop:1 #0f1b38);
                border: 1px solid #2a3b63;
                color: #eaf2ff;
                padding: 8px 14px;
                border-radius: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #3d6bff;
            }
            QPushButton:pressed {
                background: #0b1220;
            }

            QCheckBox {
                color: #dbe7ff;
            }
            QSlider::groove:horizontal {
                background: #101a32;
                border: 1px solid #1e2a46;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #7cdbff;
                width: 14px;
                margin: -6px 0;
                border-radius: 7px;
            }
        """)

    # -----------------------------
    # Initialization + population
    # -----------------------------

    def _populate_categories(self):
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        for cat in self.controller.categories():
            self.category_combo.addItem(cat)
        self.category_combo.blockSignals(False)

        # Populate prefixes
        options = self.controller.si.prefix_options()
        self.note_lbl.setText(f"Loaded {len(options)} SI prefixes")

        self.in_prefix.clear()
        self.out_prefix.clear()
        for display, key in options:
            self.in_prefix.addItem(display, key)
            self.out_prefix.addItem(display, key)

        self.in_prefix.setCurrentIndex(0)
        self.out_prefix.setCurrentIndex(0)

        self.in_prefix.setCurrentIndex(0)
        self.out_prefix.setCurrentIndex(0)

    def _initial_select(self):
        if self.category_combo.count() == 0:
            return
        # Prefer a friendly default
        for preferred in ("Length", "Mass", "Time", "Speed"):
            idx = self.category_combo.findText(preferred)
            if idx >= 0:
                self.category_combo.setCurrentIndex(idx)
                return
        self.category_combo.setCurrentIndex(0)

    # -----------------------------
    # UI events
    # -----------------------------

    def _on_category_changed(self, category: str):
        if not self._loading_ok:
            return
        units = self.controller.units_for(category)
        if not units:
            return

        self.in_unit.blockSignals(True)
        self.out_unit.blockSignals(True)
        self.in_unit.clear()
        self.out_unit.clear()
        self.in_unit.addItems(units)
        self.out_unit.addItems(units)
        self.in_unit.blockSignals(False)
        self.out_unit.blockSignals(False)

        # Try set common SI base
        si_units = self.controller.si_units_for(category)
        if si_units:
            self.si_hint.setText("SI base: " + ", ".join(si_units))
            # set defaults to first SI if present
            iu = si_units[0]
            ou = si_units[0]
            if iu in units:
                self.in_unit.setCurrentText(iu)
            if ou in units:
                self.out_unit.setCurrentText(ou)
        else:
            self.si_hint.setText("SI base: (special or dataset-defined)")

        # Temperature special: set C->K by default
        if category.strip().lower() == "temperature":
            for u in ("C", "K"):
                if u in units:
                    self.in_unit.setCurrentText("C")
                    self.out_unit.setCurrentText("K")
                    break

        self._sync_prefix_visibility()
        self._soft_pulse(self.note_lbl)
        self.calculate(reverse=(self._last_direction == "out_to_in"), log_history=False)

    def _on_input_value_edited(self, _):
        self._last_direction = "in_to_out"
        if self.live_checkbox.isChecked():
            self.calculate()

    def _on_output_value_edited(self, _):
        self._last_direction = "out_to_in"
        if self.live_checkbox.isChecked():
            self.calculate(reverse=True)

    def _on_any_change(self, *args):
        if self.live_checkbox.isChecked():
            self.calculate(reverse=(self._last_direction == "out_to_in"))

    def _on_format_changed(self, *args):
        self.formatter.mode = self.mode_combo.currentText()
        self.formatter.sig_figs = self.sig_slider.value()
        self.formatter.decimals = self.dec_slider.value()

        self.calculate(reverse=(self._last_direction == "out_to_in"), log_history=False)


    def _sync_prefix_visibility(self):
        if not self._loading_ok:
            return
        cat = self.category_combo.currentText()
        in_u = self.in_unit.currentText()
        out_u = self.out_unit.currentText()

        in_ok = self.controller.supports_prefix(cat, in_u)
        out_ok = self.controller.supports_prefix(cat, out_u)

        self.in_prefix.setEnabled(True)
        self.out_prefix.setEnabled(True)

        # Reset to none if disabled
        if not in_ok:
            self.in_prefix.setCurrentIndex(0)
        if not out_ok:
            self.out_prefix.setCurrentIndex(0)

    # -----------------------------
    # Actions
    # -----------------------------

    def calculate(self, reverse: bool = False, log_history: bool = True):
        if not self._loading_ok or self.controller is None:
            return
        if self._busy:
            return

        self._sync_prefix_visibility()

        cat = self.category_combo.currentText()
        fu = self.in_unit.currentText()
        tu = self.out_unit.currentText()
        fp = self.in_prefix.currentData() or ""
        tp = self.out_prefix.currentData() or ""
        # self.note_lbl.setText(f"prefix in={fp!r} out={tp!r}")  # uncomment for quick debugging
        try:
            self._busy = True

            if reverse:
                value = self._parse_float(self.out_value.text())
                req = ConversionRequest(cat, value, tu, fu, tp, fp)
                res = self.controller.convert(req)

                self.in_value.setText(self.formatter.format(res.value))
                self.note_lbl.setText(res.notes or "Converted (reverse).")
                self._update_visuals(res.value, fu, fp)

                if log_history:
                    sig = (cat, tu, fu, tp, fp, "out_to_in", value)
                    self._upsert_history(sig, value, tu, tp, res.value, fu, fp)

            else:
                value = self._parse_float(self.in_value.text())
                req = ConversionRequest(cat, value, fu, tu, fp, tp)
                res = self.controller.convert(req)

                self.out_value.setText(self.formatter.format(res.value))
                self.note_lbl.setText(res.notes or "Converted.")
                self._update_visuals(res.value, tu, tp)

                if log_history:
                    sig = (cat, fu, tu, fp, tp, "in_to_out", value)
                    self._upsert_history(sig, value, fu, fp, res.value, tu, tp)

            self._soft_pulse(self.out_value)

        except ConversionError as e:
            self.note_lbl.setText(f"⚠ {e}")
        except Exception as e:
            self.note_lbl.setText("⚠ Conversion failed.")
        finally:
            self._busy = False

    def _swap_units(self):
        if not self._loading_ok:
            return
        iu = self.in_unit.currentText()
        ou = self.out_unit.currentText()
        self.in_unit.setCurrentText(ou)
        self.out_unit.setCurrentText(iu)

        # swap prefixes
        ip = self.in_prefix.currentIndex()
        op = self.out_prefix.currentIndex()
        self.in_prefix.setCurrentIndex(op)
        self.out_prefix.setCurrentIndex(ip)

        # swap values
        a = self.in_value.text()
        b = self.out_value.text()
        self.in_value.setText(b)
        self.out_value.setText(a)

        self._soft_pulse(self.swap_btn)
        self._on_any_change()

    def _reset(self):
        self.in_value.setText("")
        self.out_value.setText("")
        self.note_lbl.setText("Ready.")
        self.mag_bar.set_value(1.0)
        self.scale_lbl.setText("—")
        self._soft_pulse(self.reset_btn)

    def _surprise_me(self):
        """
        Fun little feature: pick a dramatic conversion and random magnitude.
        """
        if not self._loading_ok:
            return
        cats = self.controller.categories()
        if not cats:
            return

        # Prefer dramatic categories if present
        preferred = [c for c in ("Length", "Mass", "Energy", "Pressure", "Speed", "Time") if c in cats]
        cat = preferred[int(time.time()) % len(preferred)] if preferred else cats[int(time.time()) % len(cats)]
        self.category_combo.setCurrentText(cat)

        units = self.controller.units_for(cat)
        if len(units) >= 2:
            self.in_unit.setCurrentText(units[0])
            self.out_unit.setCurrentText(units[-1])

        # wild magnitude
        exp = (int(time.time()) % 24) - 12
        value = 10 ** exp
        self.in_value.setText(self.formatter.format(value))

        self._soft_pulse(self.random_btn)
        self.calculate()

    def _copy_output(self):
        text = self.out_value.text().strip()
        if not text:
            return
        cb: QClipboard = QApplication.clipboard()  # type: ignore[name-defined]
        cb.setText(text)
        self.note_lbl.setText("Copied output to clipboard.")
        self._soft_pulse(self.copy_btn)

    def _pin_current(self):
        if not self._loading_ok:
            return
        try:
            cat = self.category_combo.currentText()
            fu = self.in_unit.currentText()
            tu = self.out_unit.currentText()
            fp = self.in_prefix.currentData() or ""
            tp = self.out_prefix.currentData() or ""

            vin = self._parse_float(self.in_value.text())
            vout = self._parse_float(self.out_value.text())

            item = HistoryItem(time.time(), cat, vin, fu, fp, vout, tu, tp)
            self.favorites.insert(0, item)
            self._render_favorites()
            self.note_lbl.setText("Pinned to favorites.")
            self._soft_pulse(self.pin_btn)
        except Exception:
            self.note_lbl.setText("⚠ Nothing to pin yet. Convert something first.")

    def _clear_history(self):
        self.history.clear()
        self.history_list.clear()
        self.note_lbl.setText("History cleared.")
        self._soft_pulse(self.clear_hist_btn)

    # -----------------------------
    # History rendering + applying
    # -----------------------------

    def _push_history(self, vin: float, fu: str, fp: str, vout: float, tu: str, tp: str):
        cat = self.category_combo.currentText()
        item = HistoryItem(time.time(), cat, vin, fu, fp, vout, tu, tp)
        self.history.insert(0, item)
        # cap
        if len(self.history) > 80:
            self.history = self.history[:80]
        self._render_history()

    def _render_history(self):
        self.history_list.clear()
        for h in self.history[:80]:
            it = QListWidgetItem(h.title())
            it.setToolTip(h.subtitle())
            it.setData(Qt.ItemDataRole.UserRole, h)
            self.history_list.addItem(it)

    def _render_favorites(self):
        self.fav_list.clear()
        for h in self.favorites[:80]:
            it = QListWidgetItem("⭐ " + h.title())
            it.setToolTip(h.subtitle())
            it.setData(Qt.ItemDataRole.UserRole, h)
            self.fav_list.addItem(it)

    def _apply_history_item(self, item: QListWidgetItem):
        h: HistoryItem = item.data(Qt.ItemDataRole.UserRole)
        self._apply_item(h)

    def _apply_favorite_item(self, item: QListWidgetItem):
        h: HistoryItem = item.data(Qt.ItemDataRole.UserRole)
        self._apply_item(h)

    def _apply_item(self, h: HistoryItem):
        self.category_combo.setCurrentText(h.category)
        self.in_unit.setCurrentText(h.from_unit)
        self.out_unit.setCurrentText(h.to_unit)

        self._set_prefix_combo(self.in_prefix, h.from_prefix)
        self._set_prefix_combo(self.out_prefix, h.to_prefix)

        self.in_value.setText(self.formatter.format(h.value_in))
        self.out_value.setText(self.formatter.format(h.value_out))
        self.note_lbl.setText("Loaded from list.")
        self._update_visuals(h.value_out, h.to_unit, h.to_prefix)

    def _set_prefix_combo(self, combo: QComboBox, prefix: str):
        prefix = prefix or ""
        for i in range(combo.count()):
            if (combo.itemData(i) or "") == prefix:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _upsert_history(self, sig, vin, fu, fp, vout, tu, tp):
        cat = self.category_combo.currentText()
        now = time.time()

        # If the "same conversion" as last time, overwrite last item instead of appending
        if self.history and self._last_history_sig == sig:
            h = self.history[0]
            h.ts = now
            h.category = cat
            h.value_in = vin
            h.from_unit = fu
            h.from_prefix = fp
            h.value_out = vout
            h.to_unit = tu
            h.to_prefix = tp

            # Update just the first list row instead of re-rendering everything
            if self.history_list.count() > 0:
                item = self.history_list.item(0)
                item.setText(h.title())
                item.setToolTip(h.subtitle())
                item.setData(Qt.ItemDataRole.UserRole, h)
            return

        # Otherwise push new
        item = HistoryItem(now, cat, vin, fu, fp, vout, tu, tp)
        self.history.insert(0, item)
        self._last_history_sig = sig

        if len(self.history) > 80:
            self.history = self.history[:80]

        self._render_history()


    # -----------------------------
    # Visual updates
    # -----------------------------

    def _update_visuals(self, value: float, unit: str, prefix: str):
        # Magnitude
        self.mag_bar.set_value(value)

        # Scale text
        p = prefix or ""
        up = (p if p else "") + unit
        label = magnitude_label(value)
        self.scale_lbl.setText(f"Output: {up} • {label} • log₁₀≈{_log10_safe(value):.2f}")

    # -----------------------------
    # Micro-animations
    # -----------------------------

    def _soft_pulse(self, widget: QWidget):
        try:
            anim = QPropertyAnimation(widget, b"minimumHeight")
            anim.setDuration(220)
            anim.setStartValue(widget.minimumHeight())
            anim.setKeyValueAt(0.5, widget.minimumHeight() + 2)
            anim.setEndValue(widget.minimumHeight())
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            anim.start()
            # keep reference
            widget._pulse_anim = anim  # type: ignore[attr-defined]
        except Exception:
            pass

    def _animate_header(self):
        # subtle pulse on title
        self._soft_pulse(self.title_lbl)

    # -----------------------------
    # Parsing
    # -----------------------------

    def _parse_float(self, s: str) -> float:
        s = (s or "").strip()
        if not s:
            return 0.0
        # allow commas
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            # allow scientific with unicode minus?
            s = s.replace("−", "-")
            return float(s)


TOOL_META = {
    "name": "Universal Unit Converter",
    "description": "Flagship, dataset-driven unit converter with SI-prefix integration, bidirectional live conversion, and scale intuition visuals",
    "category": "Utilities",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Dataset-driven conversions (no hardcoded tables)",
        "SI-prefix integration (external si-prefix module)",
        "Bidirectional conversion (edit input or output)",
        "Live update mode",
        "Temperature offsets (C/K/F)",
        "Data Size logic (bit/byte)",
        "Smart formatting (smart/fixed/scientific/engineering)",
        "Significant figures and decimal controls",
        "History timeline",
        "Favorites (pin conversions)",
        "Swap units + copy output",
        "Magnitude (log) visualization for intuition",
        "Modern, animated UI"
    ],
    "educational_value": "Build intuition for scientific magnitudes and unit systems while converting across many domains",
    "keywords": ["unit converter", "SI prefixes", "scientific notation", "engineering notation", "temperature", "bit byte", "scale intuition"]
}


def create_tool(parent=None):
    return UnitConverterTool(parent)
