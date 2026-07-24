"""
SeaTalk AI VHF — 온디바이스 AI 기반 해상 무전기 UI/UX 프로토타입 (PySide6)

실행 방법:
    pip install PySide6 numpy sounddevice --break-system-packages   (Linux/Jetson)
    pip install PySide6 numpy sounddevice                            (Windows/Mac)
    python seatalk_ai_vhf.py

- 화면 해상도 1024x600 고정 (창 크기 변경 불가)
- PTT 버튼을 누르고 있는 동안 실제 마이크 입력을 sounddevice로 캡처하여
  실시간 파형(레벨 미터)을 표시합니다. 마이크가 없거나 권한이 없으면
  자동으로 시뮬레이션 파형으로 대체됩니다.
- 실제 STT/번역/AI 위험 분석 로직은 포함되어 있지 않으며, 더미 데이터와
  타이머 기반 시뮬레이션으로 동작을 재현합니다.
"""

import sys
import random
import threading
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QStackedWidget,
    QScrollArea,
    QFrame,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QAbstractSpinBox,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)

try:
    import numpy as np
    import sounddevice as sd

    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

FRAME_W = 1024
FRAME_H = 600
BAR_COUNT = 26


# ---------------------------------------------------------------------------
# Color tokens (approximating the web prototype's dark theme)
# ---------------------------------------------------------------------------
class C:
    bg = "#020617"  # slate-950
    bar = "#020617"  # top/bottom control bar background
    panel = "#0f172a"  # slate-900
    panel_deep = "#0b1120"  # slate-950-ish
    border = "#1e293b"  # slate-800
    border2 = "#334155"  # slate-700
    text = "#ffffff"
    text_dim = "#94a3b8"  # slate-400
    text_dim2 = "#64748b"  # slate-500
    cyan = "#22d3ee"
    cyan_dark = "#083344"
    emerald = "#34d399"
    emerald_dark = "#052e2b"
    red = "#ef4444"
    red_dark = "#450a0a"
    yellow = "#eab308"


# Night (default) and Day palettes. Toggling swaps C's attributes in place, then the
# UI is rebuilt so every widget re-reads C.* while constructing its stylesheet.
DARK_THEME = dict(
    bg="#020617", bar="#020617", panel="#0f172a", panel_deep="#0b1120", border="#1e293b", border2="#334155",
    text="#ffffff", text_dim="#94a3b8", text_dim2="#64748b", cyan="#22d3ee", cyan_dark="#083344",
    emerald="#34d399", emerald_dark="#052e2b", red="#ef4444", red_dark="#450a0a", yellow="#eab308",
)
DAY_THEME = dict(
    bg="#f4f6f9", bar="#e2e8f0", panel="#ffffff", panel_deep="#e2e8f0", border="#e2e8f0", border2="#cbd5e1",
    text="#1e293b", text_dim="#334155", text_dim2="#475569", cyan="#0e7490", cyan_dark="#cffafe",
    emerald="#047857", emerald_dark="#d1fae5", red="#dc2626", red_dark="#fee2e2", yellow="#a16207",
)


def apply_theme(mode):
    """mode: 'dark' | 'day' | 'auto'. No ambient light sensor is available, so 'auto' uses the dark palette."""
    palette = DAY_THEME if mode == "day" else DARK_THEME
    for key, value in palette.items():
        setattr(C, key, value)


def now_str():
    return datetime.now().strftime("%H:%M:%S")


# Demo foreign-vessel phrases per source language, keyed to the "번역 언어 설정" combo box.
# The Korean translation shown to the operator is the same regardless of source language.
FOREIGN_DEMO = {
    "EN": ("Approaching the harbor, requesting clearance.", "EN"),
    "JA": ("港に接近中です。入港許可を要請します。", "JA"),
    "ZH": ("正在接近港口,请求入境许可。", "ZH"),
}


# ---------------------------------------------------------------------------
# Microphone capture (real when available, simulated fallback otherwise)
# ---------------------------------------------------------------------------
class MicCapture:
    """Captures microphone input and reduces it to BAR_COUNT levels (0-100).
    Falls back to a simulated waveform if no audio device is available."""

    def __init__(self, bar_count=BAR_COUNT):
        self.bar_count = bar_count
        self._lock = threading.Lock()
        self._levels = [6.0] * bar_count
        self.stream = None
        self.active = False
        self.mode = "idle"  # idle | live | simulated

    def start(self):
        self.active = True
        if HAS_AUDIO:
            try:
                self.stream = sd.InputStream(
                    channels=1,
                    samplerate=44100,
                    blocksize=1024,
                    callback=self._audio_callback,
                )
                self.stream.start()
                self.mode = "live"
                return
            except Exception:
                self.stream = None
        self.mode = "simulated"

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.active:
            return
        data = np.abs(indata[:, 0])
        chunk = max(1, len(data) // self.bar_count)
        levels = []
        for i in range(self.bar_count):
            seg = data[i * chunk : i * chunk + chunk]
            if len(seg) == 0:
                levels.append(6.0)
                continue
            avg = float(np.mean(seg))
            level = min(100.0, avg * 900.0)  # tuned gain for typical mic input
            levels.append(max(6.0, level))
        with self._lock:
            self._levels = levels

    def tick_simulated(self):
        """Call periodically from a GUI QTimer when mode == 'simulated'."""
        with self._lock:
            self._levels = [10 + random.random() * 80 for _ in range(self.bar_count)]

    def get_levels(self):
        with self._lock:
            return list(self._levels)

    def stop(self):
        self.active = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.mode = "idle"
        with self._lock:
            self._levels = [6.0] * self.bar_count


# ---------------------------------------------------------------------------
# Live waveform meter widget
# ---------------------------------------------------------------------------
class LiveMeterWidget(QWidget):
    def __init__(self, bar_count=BAR_COUNT, color=C.cyan, parent=None):
        super().__init__(parent)
        self.bar_count = bar_count
        self.levels = [6.0] * bar_count
        self.color = QColor(color)
        self.setMinimumHeight(36)

    def set_levels(self, levels):
        self.levels = levels
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n = self.bar_count
        gap = 3
        bar_w = max(2.0, (w - gap * (n - 1)) / n)
        painter.setPen(Qt.NoPen)
        for i, lvl in enumerate(self.levels):
            bar_h = max(4.0, (lvl / 100.0) * h)
            x = i * (bar_w + gap)
            y = h - bar_h
            color = QColor(self.color)
            color.setAlphaF(0.5 + (lvl / 100.0) * 0.5)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(int(x), int(y), int(bar_w), int(bar_h), bar_w / 2, bar_w / 2)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setStyleSheet(f"background-color:{C.bar}; border-bottom:1px solid {C.border2};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)

        # Channel
        ch_box = QHBoxLayout()
        ch_label = QLabel("CH")
        ch_label.setStyleSheet(f"color:{C.text_dim}; font-weight:bold; font-size:11px; background:transparent; border:none;")
        self.ch_value = QLabel("16")
        self.ch_value.setStyleSheet(
            f"color:{C.cyan}; font-weight:bold; font-size:32px; background:transparent; border:none;"
        )
        ch_box.addWidget(ch_label)
        ch_box.addWidget(self.ch_value)
        layout.addLayout(ch_box)

        layout.addSpacing(18)

        signal_label = QLabel("● 신호 강함")
        signal_label.setStyleSheet(
            f"color:{C.emerald}; font-size:12px; font-weight:600; background:transparent; border:none;"
        )
        layout.addWidget(signal_label)

        layout.addStretch()

        self.ai_status = QLabel()
        layout.addWidget(self.ai_status)
        self.set_ai_status(True)

        layout.addSpacing(18)

        self.clock_label = QLabel("00:00:00")
        self.clock_label.setStyleSheet(
            f"color:{C.text}; font-size:18px; font-family:monospace; background:transparent; border:none;"
        )
        layout.addWidget(self.clock_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)
        self._tick()

    def _tick(self):
        self.clock_label.setText(now_str())

    def set_channel(self, ch):
        self.ch_value.setText(str(ch))

    def set_ai_status(self, ok):
        if ok:
            self.ai_status.setText("●")
            self.ai_status.setToolTip("AI 모니터링 정상 작동 중")
            self.ai_status.setStyleSheet(f"color:{C.emerald}; font-size:9px; background:transparent; border:none;")
        else:
            self.ai_status.setText("⚠ AI 기능 중단됨")
            self.ai_status.setToolTip("AI 기능이 중단되어 기본 무전 기능으로 동작 중입니다")
            self.ai_status.setStyleSheet(
                f"color:{C.yellow}; background-color:rgba(234, 179, 8, 0.15); border:1px solid {C.yellow}; "
                "border-radius:12px; padding:4px 12px; font-size:11px; font-weight:600;"
            )


# ---------------------------------------------------------------------------
# Chat bubble
# ---------------------------------------------------------------------------
class ChatBubble(QWidget):
    def __init__(self, sender, timestamp, original, translated, language, is_self, font_size=18, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        bubble = QFrame()
        bubble.setMaximumWidth(680)
        bg = C.emerald_dark if is_self else C.panel
        border = C.emerald if is_self else C.border2
        radius_corner = "border-top-right-radius:4px;" if is_self else "border-top-left-radius:4px;"
        bubble.setStyleSheet(
            f"QFrame {{ background-color:{bg}; border:1px solid {border}; "
            f"border-radius:16px; {radius_corner} }}"
        )

        inner = QVBoxLayout(bubble)
        inner.setContentsMargins(14, 10, 14, 10)
        inner.setSpacing(4)

        header = QHBoxLayout()
        who = QLabel(sender)
        who.setStyleSheet(f"color:{C.text_dim}; font-weight:bold; font-size:11px; border:none;")
        ts = QLabel(timestamp)
        ts.setStyleSheet(f"color:{C.text_dim2}; font-size:11px; border:none;")
        header.addWidget(who)
        header.addWidget(ts)
        header.addStretch()
        inner.addLayout(header)

        self.main_text = QLabel(translated)
        self.main_text.setWordWrap(True)
        inner.addWidget(self.main_text)

        sub_box = QHBoxLayout()
        sub_box.setSpacing(6)
        lang_badge = QLabel(language)
        lang_badge.setStyleSheet(
            f"color:{C.text_dim}; background-color:{C.border}; border:none; "
            "border-radius:4px; padding:1px 5px; font-size:10px; font-weight:bold;"
        )
        sub_box.addWidget(lang_badge)
        self.orig_text = QLabel(original)
        self.orig_text.setWordWrap(True)
        sub_box.addWidget(self.orig_text, 1)
        inner.addLayout(sub_box)

        self.set_font_size(font_size)

        if is_self:
            outer.addStretch()
            outer.addWidget(bubble)
        else:
            outer.addWidget(bubble)
            outer.addStretch()

    def set_font_size(self, size):
        sub_size = max(10, size - 4)
        self.main_text.setStyleSheet(f"color:{C.text}; font-size:{size}px; font-weight:500; border:none;")
        self.orig_text.setStyleSheet(f"color:{C.text_dim}; font-size:{sub_size}px; border:none;")


class LiveTransmitBubble(QWidget):
    """Bubble shown while PTT is held, containing the live waveform meter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        bubble = QFrame()
        bubble.setFixedWidth(360)
        bubble.setStyleSheet(
            f"QFrame {{ background-color:{C.emerald_dark}; border:1px solid {C.emerald}; "
            "border-radius:16px; border-top-right-radius:4px; }"
        )
        inner = QVBoxLayout(bubble)
        inner.setContentsMargins(14, 10, 14, 10)

        header = QHBoxLayout()
        who = QLabel("MY VESSEL")
        who.setStyleSheet(f"color:{C.emerald}; font-weight:bold; font-size:11px; border:none;")
        status = QLabel("송신 중...")
        status.setStyleSheet(f"color:{C.text_dim2}; font-size:11px; border:none;")
        header.addWidget(who)
        header.addWidget(status)
        header.addStretch()
        inner.addLayout(header)

        self.meter = LiveMeterWidget(bar_count=BAR_COUNT, color=C.emerald)
        self.meter.setFixedHeight(36)
        inner.addWidget(self.meter)

        outer.addStretch()
        outer.addWidget(bubble)

    def set_levels(self, levels):
        self.meter.set_levels(levels)


# ---------------------------------------------------------------------------
# Main communication screen
# ---------------------------------------------------------------------------
class MainScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(f"QScrollArea {{ background-color:{C.bg}; border:none; }}")

        self.content = QWidget()
        self.content.setStyleSheet(f"background-color:{C.bg};")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 24, 20, 20)
        self.content_layout.setSpacing(14)
        self.content_layout.addStretch()

        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        self.live_bubble = None
        self._bubbles = []
        self._font_size = 18
        self.scroll.verticalScrollBar().rangeChanged.connect(self._scroll_to_bottom)

    def _scroll_to_bottom(self, _min=None, _max=None):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def add_message(self, sender, timestamp, original, translated, language, is_self):
        bubble = ChatBubble(sender, timestamp, original, translated, language, is_self, font_size=self._font_size)
        # insert before the trailing stretch
        idx = self.content_layout.count() - 1
        self.content_layout.insertWidget(idx, bubble)
        self._bubbles.append(bubble)
        self._scroll_to_bottom()

    def clear_messages(self):
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._bubbles.clear()

    def set_font_size(self, size):
        self._font_size = size
        for bubble in self._bubbles:
            bubble.set_font_size(size)

    def show_live_bubble(self):
        if self.live_bubble is None:
            self.live_bubble = LiveTransmitBubble()
            idx = self.content_layout.count() - 1
            self.content_layout.insertWidget(idx, self.live_bubble)
            self._scroll_to_bottom()

    def update_live_levels(self, levels):
        if self.live_bubble is not None:
            self.live_bubble.set_levels(levels)

    def hide_live_bubble(self):
        if self.live_bubble is not None:
            self.content_layout.removeWidget(self.live_bubble)
            self.live_bubble.deleteLater()
            self.live_bubble = None


# ---------------------------------------------------------------------------
# Settings screen
# ---------------------------------------------------------------------------
class SettingsScreen(QWidget):
    channelChanged = Signal(str)
    fontSizeChanged = Signal(int)
    languageChanged = Signal(str)
    backRequested = Signal()

    LANG_CODES = ["EN", "JA", "ZH"]
    PRESET_CHANNELS = ["06", "13", "16", "72"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color:{C.bg};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        title_row = QHBoxLayout()
        back_btn = QPushButton("←")
        back_btn.setFixedSize(36, 36)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(
            f"QPushButton {{ background-color:{C.panel}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; font-size:18px; font-weight:bold; }"
            f"QPushButton:hover {{ border-color:{C.cyan}; color:{C.cyan}; }}"
        )
        back_btn.clicked.connect(self.backRequested.emit)
        title_row.addWidget(back_btn)
        title_row.addSpacing(10)
        title = QLabel("설정 (Settings)")
        title.setStyleSheet(f"color:{C.text}; font-size:20px; font-weight:bold; border:none;")
        title_row.addWidget(title)
        title_row.addStretch()
        outer.addLayout(title_row)
        outer.addSpacing(16)

        grid = QGridLayout()
        grid.setSpacing(18)
        outer.addLayout(grid)
        outer.addStretch()

        # Channel card
        ch_card, ch_layout = self._card("VHF 채널 설정")
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 99)
        self.channel_spin.setValue(16)
        self.channel_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.channel_spin.setAlignment(Qt.AlignCenter)
        self.channel_spin.setFixedHeight(90)
        self.channel_spin.setStyleSheet(
            f"QSpinBox {{ background-color:{C.bg}; color:{C.cyan}; border:1px solid {C.border2}; "
            "border-radius:10px; font-size:60px; font-weight:bold; padding:0px; }"
        )
        self.channel_spin.valueChanged.connect(self._on_channel_value_changed)
        ch_layout.addWidget(self.channel_spin)

        self.preset_buttons = {}
        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        for p in self.PRESET_CHANNELS:
            btn = QPushButton(f"CH {p}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(52)
            btn.clicked.connect(lambda checked=False, val=p: self._set_channel(val))
            preset_row.addWidget(btn)
            self.preset_buttons[p] = btn
        ch_layout.addLayout(preset_row)
        grid.addWidget(ch_card, 0, 0)
        self._refresh_preset_styles()

        # Language card
        lang_card, lang_layout = self._card("번역 언어 설정")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["한국어 ↔ 영어 (English)", "한국어 ↔ 일본어 (日本語)", "한국어 ↔ 중국어 (中文)"])
        self.lang_combo.setStyleSheet(
            f"background-color:{C.bg}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; padding:8px; font-size:13px;"
        )
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        grid.addWidget(lang_card, 0, 1)

        # Font size card
        font_card, font_layout = self._card("자막 폰트 크기")
        font_row = QHBoxLayout()
        minus_btn = QPushButton("A-")
        plus_btn = QPushButton("A+")
        self.font_value_label = QLabel("18px")
        self.font_value_label.setAlignment(Qt.AlignCenter)
        self.font_value_label.setStyleSheet(f"color:{C.cyan}; font-size:18px; font-weight:bold; border:none;")
        for b in (minus_btn, plus_btn):
            b.setFixedSize(48, 48)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ background-color:{C.bg}; color:{C.text}; "
                f"border:1px solid {C.border2}; border-radius:8px; font-weight:bold; }}"
                f"QPushButton:hover {{ border-color:{C.cyan}; }}"
            )
        self._font_size = 18
        minus_btn.clicked.connect(lambda: self._bump_font(-2))
        plus_btn.clicked.connect(lambda: self._bump_font(2))
        font_row.addWidget(minus_btn)
        font_row.addWidget(self.font_value_label, 1)
        font_row.addWidget(plus_btn)
        font_layout.addLayout(font_row)
        grid.addWidget(font_card, 1, 0)

        # Volume card
        vol_card, vol_layout = self._card("시스템 볼륨")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(70)
        slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background:{C.border2}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal {{ background:{C.cyan}; width:16px; margin:-6px 0; border-radius:8px; }}"
        )
        vol_layout.addWidget(slider)
        grid.addWidget(vol_card, 1, 1)

    def _card(self, title_text):
        card = QFrame()
        card.setStyleSheet(f"background-color:{C.panel}; border:1px solid {C.border}; border-radius:14px;")
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(15, 23, 42, 40))
        card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        title = QLabel(title_text)
        title.setStyleSheet(f"color:{C.text_dim}; font-size:12px; font-weight:bold; border:none;")
        layout.addWidget(title)
        return card, layout

    def _set_channel(self, val):
        self.channel_spin.setValue(int(val))

    def _on_channel_value_changed(self, v):
        self.channelChanged.emit(str(v))
        self._refresh_preset_styles()

    def _refresh_preset_styles(self):
        current = f"{self.channel_spin.value():02d}"
        for p, btn in self.preset_buttons.items():
            if p == current:
                btn.setStyleSheet(
                    f"QPushButton {{ background-color:{C.cyan}; color:{C.bg}; "
                    f"border:2px solid {C.cyan}; border-radius:10px; padding:10px; "
                    "font-size:15px; font-weight:bold; }"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background-color:{C.bg}; color:{C.text_dim}; "
                    f"border:2px solid {C.border2}; border-radius:10px; padding:10px; "
                    "font-size:15px; font-weight:bold; }"
                    f"QPushButton:hover {{ border-color:{C.cyan}; color:{C.cyan}; }}"
                )

    def _bump_font(self, delta):
        self._font_size = max(12, min(32, self._font_size + delta))
        self.font_value_label.setText(f"{self._font_size}px")
        self.fontSizeChanged.emit(self._font_size)

    def _on_language_changed(self, index):
        self.languageChanged.emit(self.LANG_CODES[index])


# ---------------------------------------------------------------------------
# Log screen
# ---------------------------------------------------------------------------
class LogScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color:{C.bg};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        header = QHBoxLayout()
        title = QLabel("교신 기록 (Logs)")
        title.setStyleSheet(f"color:{C.text}; font-size:20px; font-weight:bold; border:none;")
        header.addWidget(title)
        header.addStretch()

        search = QLineEdit()
        search.setPlaceholderText("선박명, 키워드 검색")
        search.setFixedWidth(220)
        search.setStyleSheet(
            f"background-color:{C.panel}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; padding:6px;"
        )
        header.addWidget(search)
        outer.addLayout(header)
        outer.addSpacing(12)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시간", "선박명", "내용 (번역)"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color:{C.panel}; color:{C.text}; border:1px solid {C.border}; "
            f"border-radius:12px; gridline-color:{C.border}; }}"
            f"QHeaderView::section {{ background-color:{C.panel_deep}; color:{C.text_dim}; "
            "border:none; padding:6px; font-weight:bold; font-size:11px; }"
        )
        outer.addWidget(self.table)

    def add_row(self, timestamp, sender, translated):
        row = 0
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.table.setItem(row, 1, QTableWidgetItem(sender))
        self.table.setItem(row, 2, QTableWidgetItem(translated))

    def clear_rows(self):
        self.table.setRowCount(0)


# ---------------------------------------------------------------------------
# Emergency overlay
# ---------------------------------------------------------------------------
class EmergencyOverlay(QWidget):
    dismissed = Signal()
    viewLogsRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(2, 6, 23, 0.72);")
        self.setAutoFillBackground(True)

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)

        # Single glassmorphism panel: translucent fill + soft border, no nested boxes.
        panel = QFrame()
        panel.setFixedWidth(580)
        panel.setStyleSheet(
            "QFrame { background-color: rgba(15, 23, 42, 0.75); "
            "border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 20px; }"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(10)
        panel_layout.setAlignment(Qt.AlignHCenter)

        logo = QLabel("⚠")
        logo.setFixedSize(58, 58)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "QLabel { background: qradialgradient(cx:0.5, cy:0.4, radius:0.9, fx:0.5, fy:0.4, "
            f"stop:0 #fb923c, stop:1 {C.red}); border-radius:29px; color:white; font-size:24px; border:none; }}"
        )
        panel_layout.addWidget(logo, 0, Qt.AlignCenter)

        title = QLabel("비상 상황 감지")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{C.red}; font-size:24px; font-weight:bold; border:none;")
        panel_layout.addWidget(title)

        subtitle = QLabel("COLLISION RISK DETECTED")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#f8fafc; font-size:12px; font-weight:bold; letter-spacing:3px; border:none;")
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self._divider())
        panel_layout.addSpacing(8)

        vessel = QLabel("감지된 교신 내용  ·  신박: UNKNOWN VESSEL")
        vessel.setAlignment(Qt.AlignCenter)
        vessel.setStyleSheet("color:#fca5a5; font-size:11px; font-weight:bold; border:none;")
        panel_layout.addWidget(vessel)

        msg1 = QLabel('"충돌 위험! 즉시 변침하십시오!"')
        msg1.setAlignment(Qt.AlignCenter)
        msg1.setWordWrap(True)
        msg1.setStyleSheet("color:#ffffff; font-size:18px; font-weight:700; border:none;")
        panel_layout.addWidget(msg1)

        msg2 = QLabel('"Collision imminent! Alter course immediately!"')
        msg2.setAlignment(Qt.AlignCenter)
        msg2.setWordWrap(True)
        msg2.setStyleSheet("color:#cbd5e1; font-size:12px; border:none;")
        panel_layout.addWidget(msg2)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self._divider())
        panel_layout.addSpacing(8)

        guide_title = QLabel("대응 가이드")
        guide_title.setAlignment(Qt.AlignCenter)
        guide_title.setStyleSheet("color:#94a3b8; font-size:11px; font-weight:bold; border:none;")
        panel_layout.addWidget(guide_title)

        guide_items = [
            "주변 선박의 위치와 침로를 즉시 확인하십시오.",
            "VHF CH 16으로 해당 선박과 교신을 시도하십시오.",
        ]
        for text in guide_items:
            row = QLabel(f"•  {text}")
            row.setAlignment(Qt.AlignCenter)
            row.setStyleSheet("color:#e2e8f0; font-size:13px; border:none;")
            panel_layout.addWidget(row)

        panel_layout.addSpacing(14)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        confirm_btn = QPushButton("⚠  확인 / 경고 해제")
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.setStyleSheet(
            "QPushButton { background-color:#f59e0b; color:#1c1917; border:none; "
            "border-radius:10px; padding:12px 28px; font-size:14px; font-weight:bold; }"
            "QPushButton:hover { background-color:#fbbf24; }"
        )
        confirm_btn.clicked.connect(self.dismissed.emit)

        logs_btn = QPushButton("교신 기록 보기")
        logs_btn.setCursor(Qt.PointingHandCursor)
        logs_btn.setStyleSheet(
            "QPushButton { background-color: rgba(255, 255, 255, 0.08); color:white; "
            "border:1px solid rgba(255, 255, 255, 0.25); border-radius:10px; padding:12px 28px; "
            "font-size:14px; font-weight:bold; }"
            "QPushButton:hover { background-color: rgba(255, 255, 255, 0.16); }"
        )
        logs_btn.clicked.connect(self.viewLogsRequested.emit)

        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(logs_btn)
        panel_layout.addLayout(btn_row)

        outer.addWidget(panel, 0, Qt.AlignCenter)

    @staticmethod
    def _divider():
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: rgba(255, 255, 255, 0.12); border:none;")
        return line


# ---------------------------------------------------------------------------
# Boot screen
# ---------------------------------------------------------------------------
class BootScreen(QWidget):
    completed = Signal()

    STEPS = [
        "AI 음성인식(STT) 모델 로딩",
        "다국어 번역 모델 로딩",
        "VHF 무전 보드 연결 확인",
        "마이크·스피커 연결 확인",
        "저장장치 상태 확인",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color:{C.bg};")
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setSpacing(24)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignCenter)
        title_box = QVBoxLayout()
        title = QLabel("SeaTalk AI")
        title.setStyleSheet(f"color:{C.cyan}; font-size:30px; font-weight:bold; border:none;")
        subtitle = QLabel("Smart VHF System v2.1.0")
        subtitle.setStyleSheet(f"color:{C.text_dim}; font-size:13px; border:none;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        outer.addLayout(header)

        card = QFrame()
        card.setFixedWidth(420)
        card.setStyleSheet(f"background-color:{C.panel}; border:1px solid {C.border}; border-radius:14px;")
        self.card_layout = QVBoxLayout(card)
        self.card_layout.setContentsMargins(20, 18, 20, 18)
        self.card_layout.setSpacing(12)

        self.step_labels = []
        self.step_status = []
        for text in self.STEPS:
            row = QHBoxLayout()
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{C.text_dim2}; font-size:13px; border:none;")
            status = QLabel("○")
            status.setStyleSheet(f"color:{C.text_dim2}; font-size:14px; border:none;")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(status)
            self.card_layout.addLayout(row)
            self.step_labels.append(lbl)
            self.step_status.append(status)

        outer.addWidget(card, 0, Qt.AlignCenter)

        footer = QLabel("AI 기능 실패 시에도 기본 무전 기능은 정상 동작합니다")
        footer.setStyleSheet(f"color:{C.text_dim2}; font-size:11px; border:none;")
        outer.addWidget(footer, 0, Qt.AlignCenter)

        self._step = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.timer.start(650)

    def _advance(self):
        if self._step < len(self.STEPS):
            self.step_labels[self._step].setStyleSheet(f"color:{C.cyan}; font-size:13px; border:none;")
            self.step_status[self._step].setText("◐")
            self.step_status[self._step].setStyleSheet(f"color:{C.cyan}; font-size:14px; border:none;")
            QTimer.singleShot(400, lambda i=self._step: self._complete_step(i))
            self._step += 1
        else:
            self.timer.stop()
            QTimer.singleShot(500, self.completed.emit)

    def _complete_step(self, i):
        self.step_labels[i].setStyleSheet(f"color:{C.text_dim}; font-size:13px; border:none;")
        self.step_status[i].setText("✓")
        self.step_status[i].setStyleSheet(f"color:{C.emerald}; font-size:14px; font-weight:bold; border:none;")


# ---------------------------------------------------------------------------
# Bottom navigation (incl. PTT)
# ---------------------------------------------------------------------------
class BottomNav(QWidget):
    navRequested = Signal(str)
    clearRequested = Signal()
    pttStarted = Signal()
    pttEnded = Signal()
    modeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self.setStyleSheet(f"background-color:{C.bar}; border-top:1px solid {C.border2};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)

        left = QHBoxLayout()
        left.setSpacing(10)
        self.nav_buttons = {}
        for key, label in [("settings", "MENU")]:
            btn = self._nav_button(label)
            btn.clicked.connect(lambda checked=False, k=key: self.navRequested.emit(k))
            self.nav_buttons[key] = btn
            left.addWidget(btn)

        clr_btn = self._nav_button("CLR")
        clr_btn.clicked.connect(self.clearRequested.emit)
        left.addWidget(clr_btn)

        logs_btn = self._nav_button("LOG")
        logs_btn.clicked.connect(lambda: self.navRequested.emit("logs"))
        self.nav_buttons["logs"] = logs_btn
        left.addWidget(logs_btn)

        layout.addLayout(left)
        layout.addSpacing(16)

        self._ptt_active = False
        self.ptt_btn = QPushButton("🎙  PTT")
        self.ptt_btn.setFixedHeight(60)
        self.ptt_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ptt_btn.setCursor(Qt.PointingHandCursor)
        self._ptt_style(pressed=False)
        self.ptt_btn.clicked.connect(self._on_ptt_toggle)
        layout.addWidget(self.ptt_btn, 1)

        right = QHBoxLayout()
        self.mode_btn = self._nav_button("MODE")
        self.mode_btn.clicked.connect(self.modeRequested.emit)
        right.addWidget(self.mode_btn)
        layout.addLayout(right)

    def _nav_button(self, label):
        btn = QPushButton(label)
        btn.setFixedSize(76, 60)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background-color:{C.panel}; color:{C.text_dim}; "
            f"border:1px solid {C.border}; border-radius:10px; font-weight:bold; font-size:11px; }}"
            f"QPushButton:hover {{ background-color:#1e2636; }}"
        )
        return btn

    def _ptt_style(self, pressed):
        if pressed:
            self.ptt_btn.setStyleSheet(
                f"QPushButton {{ background-color:{C.red}; color:white; border:none; "
                "border-radius:12px; font-weight:bold; font-size:16px; }"
            )
        else:
            self.ptt_btn.setStyleSheet(
                f"QPushButton {{ background-color:{C.cyan}; color:{C.bg}; border:none; "
                "border-radius:12px; font-weight:bold; font-size:16px; }"
                "QPushButton:hover { background-color:#67e8f9; }"
            )

    def _on_ptt_toggle(self):
        self._ptt_active = not self._ptt_active
        self._ptt_style(pressed=self._ptt_active)
        if self._ptt_active:
            self.pttStarted.emit()
        else:
            self.pttEnded.emit()


# ---------------------------------------------------------------------------
# Main app container (top bar + content stack + bottom nav)
# ---------------------------------------------------------------------------
class AppContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color:{C.bg};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_bar = TopBar()
        layout.addWidget(self.top_bar)

        self.content_stack = QStackedWidget()
        self.main_screen = MainScreen()
        self.settings_screen = SettingsScreen()
        self.log_screen = LogScreen()
        self.content_stack.addWidget(self.main_screen)  # 0
        self.content_stack.addWidget(self.settings_screen)  # 1
        self.content_stack.addWidget(self.log_screen)  # 2
        layout.addWidget(self.content_stack, 1)

        self.bottom_nav = BottomNav()
        layout.addWidget(self.bottom_nav)

        self._screen_index = {"main": 0, "settings": 1, "logs": 2}
        self.bottom_nav.navRequested.connect(self._toggle_screen)
        self.settings_screen.backRequested.connect(lambda: self.content_stack.setCurrentIndex(0))

    def _toggle_screen(self, key):
        target = self._screen_index[key]
        current = self.content_stack.currentIndex()
        self.content_stack.setCurrentIndex(0 if current == target else target)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeaTalk AI VHF")
        self.setFixedSize(FRAME_W, FRAME_H)
        self.setStyleSheet(f"background-color:{C.bg};")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.root_stack = QStackedWidget()
        root_layout.addWidget(self.root_stack)

        self.boot_screen = BootScreen()
        self.app_container = AppContainer()
        self.root_stack.addWidget(self.boot_screen)  # 0
        self.root_stack.addWidget(self.app_container)  # 1

        self.boot_screen.completed.connect(self._on_boot_complete)

        # Emergency overlay floats above everything, manual geometry
        self.overlay = EmergencyOverlay(self)
        self.overlay.setGeometry(0, 0, FRAME_W, FRAME_H)
        self.overlay.hide()
        self.overlay.dismissed.connect(self._hide_emergency)
        self.overlay.viewLogsRequested.connect(self._view_logs_from_overlay)

        # App state
        self.logs = []
        self.channel = "16"
        self.foreign_lang = "EN"
        self.mic = MicCapture()
        self.theme_modes = ["dark", "day", "auto"]
        self.theme_index = 0

        # PTT live meter timer (GUI thread; reads mic levels or drives simulation)
        self.ptt_timer = QTimer(self)
        self.ptt_timer.timeout.connect(self._ptt_tick)

        self._wire_app_container()

        # Background simulated incoming messages
        self.incoming_timer = QTimer(self)
        self.incoming_timer.timeout.connect(self._maybe_incoming_message)

        self._emergency_fired = False

    def _wire_app_container(self):
        self.app_container.settings_screen.channelChanged.connect(self._set_channel)
        self.app_container.settings_screen.fontSizeChanged.connect(self.app_container.main_screen.set_font_size)
        self.app_container.settings_screen.languageChanged.connect(self._set_foreign_language)
        self.app_container.bottom_nav.pttStarted.connect(self._start_ptt)
        self.app_container.bottom_nav.pttEnded.connect(self._end_ptt)
        self.app_container.bottom_nav.clearRequested.connect(self._clear_logs)
        self.app_container.bottom_nav.modeRequested.connect(self._cycle_theme)

    # ---- Boot ----
    def _on_boot_complete(self):
        self.root_stack.setCurrentIndex(1)
        self.incoming_timer.start(4500)
        QTimer.singleShot(18000, self._trigger_emergency)

    # ---- Channel ----
    def _set_channel(self, ch):
        self.channel = ch
        self.app_container.top_bar.set_channel(ch)

    # ---- Language ----
    def _set_foreign_language(self, code):
        self.foreign_lang = code

    # ---- Theme (Day/Night/Auto) ----
    def _cycle_theme(self):
        self.theme_index = (self.theme_index + 1) % len(self.theme_modes)
        mode = self.theme_modes[self.theme_index]
        apply_theme(mode)
        self._rebuild_ui()

    def _rebuild_ui(self):
        # Colors are baked into stylesheets at widget-construction time, so the simplest
        # reliable way to re-theme everything is to recreate the screens against the
        # freshly-updated C palette, then restore whatever state the user had.
        channel = self.channel
        font_size = self.app_container.main_screen._font_size
        lang_index = self.app_container.settings_screen.lang_combo.currentIndex()
        current_screen = self.app_container.content_stack.currentIndex()
        logs_snapshot = list(self.logs)

        # The window's own background is baked in at construction time too - refresh it
        # here, otherwise a stale dark rectangle can show through around/behind the
        # freshly-rebuilt (light) screens.
        self.setStyleSheet(f"background-color:{C.bg};")

        self.root_stack.removeWidget(self.app_container)
        self.app_container.deleteLater()
        self.app_container = AppContainer()
        self.root_stack.insertWidget(1, self.app_container)
        self.root_stack.setCurrentIndex(1)

        self.overlay.setParent(None)
        self.overlay.deleteLater()
        self.overlay = EmergencyOverlay(self)
        self.overlay.setGeometry(0, 0, FRAME_W, FRAME_H)
        self.overlay.hide()
        self.overlay.dismissed.connect(self._hide_emergency)
        self.overlay.viewLogsRequested.connect(self._view_logs_from_overlay)

        self._wire_app_container()

        self.app_container.settings_screen.channel_spin.setValue(int(channel))
        self._set_channel(channel)
        self.app_container.main_screen.set_font_size(font_size)
        self.app_container.settings_screen._font_size = font_size
        self.app_container.settings_screen.font_value_label.setText(f"{font_size}px")
        self.app_container.settings_screen.lang_combo.setCurrentIndex(lang_index)
        self.app_container.content_stack.setCurrentIndex(current_screen)

        self.logs = []
        for entry in logs_snapshot:
            self._add_message(
                entry["sender"], entry["original"], entry["translated"], entry["language"], entry["is_self"]
            )

    # ---- Messages ----
    def _add_message(self, sender, original, translated, language, is_self):
        ts = now_str()
        self.logs.append(
            {
                "timestamp": ts,
                "sender": sender,
                "original": original,
                "translated": translated,
                "language": language,
                "is_self": is_self,
            }
        )
        self.app_container.main_screen.add_message(sender, ts, original, translated, language, is_self)
        self.app_container.log_screen.add_row(ts, sender, translated)

    def _clear_logs(self):
        self.logs.clear()
        self.app_container.main_screen.clear_messages()
        self.app_container.log_screen.clear_rows()

    def _maybe_incoming_message(self):
        if self.app_container.bottom_nav._ptt_active:
            return
        if random.random() > 0.55:
            sender = "OCEAN STAR" if random.random() > 0.5 else "PORT CONTROL"
            original, tag = FOREIGN_DEMO.get(self.foreign_lang, FOREIGN_DEMO["EN"])
            self._add_message(
                sender,
                original,
                "항구에 접근 중입니다. 진입 허가를 요청합니다.",
                tag,
                is_self=False,
            )

    # ---- PTT ----
    def _start_ptt(self):
        self.mic.start()
        self.app_container.top_bar.set_ai_status(self.mic.mode == "live")
        self.app_container.main_screen.show_live_bubble()
        self.ptt_timer.start(60)

    def _ptt_tick(self):
        if self.mic.mode == "simulated":
            self.mic.tick_simulated()
        levels = self.mic.get_levels()
        self.app_container.main_screen.update_live_levels(levels)

    def _end_ptt(self):
        self.ptt_timer.stop()
        self.mic.stop()
        self.app_container.main_screen.hide_live_bubble()
        self._add_message(
            "MY VESSEL",
            "Copy that, standing by.",
            "수신 완료, 대기 중입니다.",
            "EN",
            is_self=True,
        )

    # ---- Emergency ----
    def _trigger_emergency(self):
        if self._emergency_fired:
            return
        self._emergency_fired = True
        self.overlay.setGeometry(0, 0, FRAME_W, FRAME_H)
        self.overlay.show()
        self.overlay.raise_()

    def _hide_emergency(self):
        self.overlay.hide()

    def _view_logs_from_overlay(self):
        self.overlay.hide()
        self.app_container.content_stack.setCurrentIndex(2)

    def closeEvent(self, event):
        self.mic.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic" if sys.platform.startswith("win") else "Noto Sans CJK KR", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
