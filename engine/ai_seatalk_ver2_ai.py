"""
SeaTalk AI VHF — ver2 UI + 실제 AI 엔진 연동판 (파인튜닝 faster-whisper)

원본 ai_seatalk_ver2.py(하드웨어팀)는 그대로 두고, "AI 연동 지점" 스텁만 실제로 채운 사본.
바뀐 것 3가지:
  1) MicCapture가 PTT 동안 오디오를 16kHz로 녹음(버퍼) — 원본은 레벨 미터만 계산
  2) transcribe_audio(): 파인튜닝 faster-whisper 호출 (환경변수 AI_MODEL_DIR로 모델 폴더 지정,
     없으면 ../models/faster-whisper-small-marine → 그것도 없으면 기본 small)
  3) 비상 경보: 부팅 18초 후 무조건 발동하던 데모 타이머 제거 →
     실제 인식 텍스트에서 조난·긴급(메이데이 등) 감지 시 발동 (marine_danger 등급 기반)
  + 가짜 수신 메시지 타이머(4.5초마다 랜덤 문구)도 실연동 확인에 방해되어 비활성화.

실행(젯슨):
  AI_MODEL_DIR=~/models/faster-whisper-small-marine python3 ai_seatalk_ver2_ai.py


실행 방법:
    pip install PySide6 numpy sounddevice --break-system-packages   (Linux/Jetson)
    pip install PySide6 numpy sounddevice                            (Windows/Mac)
    python seatalk_ai_vhf.py

- 기본 실행은 "키오스크 모드": 제목바 없는 프레임리스 창을 실제 화면 해상도에
  맞춰 전체화면으로 띄웁니다. 실기기(젯슨 나노 + 터치스크린)에서 창이 화면
  밖으로 밀리는 문제를 막기 위한 설정입니다.
- PTT 버튼을 누르고 있는 동안 실제 마이크 입력을 sounddevice로 캡처하여
  실시간 파형(레벨 미터)을 표시합니다. 마이크가 없거나 권한이 없으면
  자동으로 시뮬레이션 파형으로 대체됩니다.
- 실제 STT/번역/AI 위험 분석 로직은 포함되어 있지 않으며, 더미 데이터와
  타이머 기반 시뮬레이션으로 동작을 재현합니다.

Jetson Nano 실기기 배포 시 (성능 최적화):
    # 1) 화면 서버에 맞는 Qt 플랫폼 플러그인 지정
    #    - 일반 X11 데스크톱 위에서 창으로 띄울 때: xcb (기본값)
    #    - 부팅 시 자동 실행되는 전용 풀스크린 디스플레이(HDMI 직결 등): eglfs
    export QT_QPA_PLATFORM=xcb        # 또는 eglfs

    # 2) 10W 최대 성능 모드로 전환 + 클럭 고정 (기본 5W 저전력 모드 해제)
    sudo nvpmodel -m 0
    sudo jetson_clocks

    python3 seatalk_ai_vhf.py
"""

import os
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


# ---------------------------------------------------------------------------
# AI integration points — STT (speech-to-text) and translation
# ---------------------------------------------------------------------------
# Everything in this block is a STUB. It exists so the rest of the app can
# call a stable function signature regardless of whether the model behind it
# is a canned demo response or a real Whisper/LLM call. To wire in the real
# models, replace the BODY of transcribe_audio() / translate_text() /
# translate_to_korean() below — no other file needs to change.
#
# Real pipeline this replaces:
#   PTT held   -> mic audio captured (see MicCapture, already real)
#   PTT released -> transcribe_audio(): Whisper STT, audio -> Korean text
#                -> translate_text(): LLM translation, Korean -> foreign_lang
#   Incoming   -> (radio audio ->) transcribe -> translate_to_korean(): LLM
#                translation, foreign_lang -> Korean, for on-screen display

# Demo phrase set: each entry is the *same sentence* in Korean + every
# language selectable in "번역 언어 설정" (EN/JA/ZH). The stub functions look
# a phrase up here instead of calling a model, but always respect whichever
# language is currently selected in Settings — a real model would too.
PHRASE_BANK = [
    {
        "ko": "항구에 접근 중입니다. 진입 허가를 요청합니다.",
        "EN": "Approaching the harbor, requesting clearance.",
        "JA": "港に接近中です。入港許可を要請します。",
        "ZH": "正在接近港口,请求入境许可。",
    },
    {
        "ko": "수신 완료, 대기 중입니다.",
        "EN": "Copy that, standing by.",
        "JA": "了解しました、待機します。",
        "ZH": "收到,待命中。",
    },
    {
        "ko": "현재 침로를 유지하며 항해 중입니다.",
        "EN": "Maintaining current course and heading.",
        "JA": "現在の針路を維持して航行中です。",
        "ZH": "正在保持当前航向航行。",
    },
]


# [AI] 파인튜닝 모델 로드 (지연 로드 — 첫 PTT 때 1회)
_ASR = {"model": None, "hotwords": None}
AI_MODEL_DIR = os.environ.get(
    "AI_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "faster-whisper-small-marine"))

def _load_asr():
    if _ASR["model"] is None:
        from faster_whisper import WhisperModel
        path = AI_MODEL_DIR if os.path.isdir(AI_MODEL_DIR) else "small"
        print(f"[AI] STT 모델 로드: {path}")
        _ASR["model"] = WhisperModel(path, device="auto", compute_type="int8")
        # user_dict.txt의 고유명사를 hotwords로 (교정쌍 "=>" 줄은 제외)
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            terms = []
            with open(os.path.join(here, "user_dict.txt"), encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t and not t.startswith("#") and "=>" not in t:
                        terms.append(t)
            _ASR["hotwords"] = " ".join(terms) if terms else None
        except Exception:
            _ASR["hotwords"] = None
    return _ASR["model"]


def transcribe_audio(mic: "MicCapture") -> str:
    """[AI 연동 — 실구현] PTT 동안 캡처된 16kHz 버퍼를 파인튜닝 Whisper로 전사."""
    audio = getattr(mic, "rec_audio", None)
    if audio is None or len(audio) < 8000:          # 0.5초 미만이면 무시
        return ""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-3:                                  # 무음
        return ""
    if peak < 0.1:                                   # 자동 증폭 (엔진과 동일 로직)
        audio = (audio / peak * 0.7).astype("float32")
    model = _load_asr()
    kwargs = dict(language="ko", beam_size=5,
                  condition_on_previous_text=False,  # 환각 루프(같은 말 반복) 차단
                  no_repeat_ngram_size=3)
    try:
        if _ASR["hotwords"]:
            segs, _ = model.transcribe(audio, hotwords=_ASR["hotwords"], **kwargs)
        else:
            segs, _ = model.transcribe(audio, **kwargs)
    except TypeError:                                # 구버전 faster-whisper: hotwords 미지원
        segs, _ = model.transcribe(audio, **kwargs)
    return " ".join(s.text.strip() for s in segs).strip()


def translate_text(text_ko: str, target_lang: str) -> str:
    """[AI 연동 지점 — 번역] 실제 구현: LLM 기반 번역 모델에 (text_ko, target_lang)을
    넘겨 번역 결과를 반환한다. 지금은 문구뱅크에서 일치하는 한국어 원문을 찾아
    설정된 target_lang(EN/JA/ZH)에 대응하는 번역을 반환하는 시뮬레이션이다."""
    for phrase in PHRASE_BANK:
        if phrase["ko"] == text_ko:
            return phrase.get(target_lang, text_ko)
    return f"[{target_lang} 번역 준비 중] {text_ko}"


def translate_to_korean(text_foreign: str, source_lang: str) -> str:
    """[AI 연동 지점 — 번역] 실제 구현: LLM 기반 번역 모델에 (text_foreign, source_lang)을
    넘겨 한국어 번역 결과를 반환한다. 지금은 문구뱅크에서 일치하는 원문을 찾아
    한국어 번역을 반환하는 시뮬레이션이다."""
    for phrase in PHRASE_BANK:
        if phrase.get(source_lang) == text_foreign:
            return phrase["ko"]
    return f"[번역 준비 중] {text_foreign}"


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
        self._rec = []          # [AI] PTT 동안의 원시 오디오 (STT 입력)
        self.rec_audio = None   # [AI] stop() 후 완성 버퍼 (float32 16kHz)

    def start(self):
        self.active = True
        self._rec = []          # [AI] 새 송신 시작 — 버퍼 리셋
        self.rec_audio = None
        if HAS_AUDIO:
            try:
                self.stream = sd.InputStream(
                    channels=1,
                    samplerate=16000,   # [AI] Whisper 입력 규격(원본 44100) — pipewire가 리샘플
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
        self._rec.append(indata[:, 0].copy())   # [AI] 녹음 버퍼 축적
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
        # [AI] 녹음 버퍼 확정
        if self._rec:
            try:
                self.rec_audio = np.concatenate(self._rec).astype("float32")
            except Exception:
                self.rec_audio = None
        self._rec = []
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
        self.ch_label = QLabel("CH")
        self.ch_label.setStyleSheet(f"color:{C.text_dim}; font-weight:bold; font-size:11px; background:transparent; border:none;")
        self.ch_value = QLabel("16")
        self.ch_value.setStyleSheet(
            f"color:{C.cyan}; font-weight:bold; font-size:32px; background:transparent; border:none;"
        )
        ch_box.addWidget(self.ch_label)
        ch_box.addWidget(self.ch_value)
        layout.addLayout(ch_box)

        layout.addSpacing(18)

        self.signal_label = QLabel("● 신호 강함")
        self.signal_label.setStyleSheet(
            f"color:{C.emerald}; font-size:12px; font-weight:600; background:transparent; border:none;"
        )
        layout.addWidget(self.signal_label)

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

    def retheme(self):
        """Re-apply current C.* palette without destroying/recreating this widget."""
        self.setStyleSheet(f"background-color:{C.bar}; border-bottom:1px solid {C.border2};")
        self.ch_label.setStyleSheet(
            f"color:{C.text_dim}; font-weight:bold; font-size:11px; background:transparent; border:none;"
        )
        self.ch_value.setStyleSheet(
            f"color:{C.cyan}; font-weight:bold; font-size:32px; background:transparent; border:none;"
        )
        self.signal_label.setStyleSheet(
            f"color:{C.emerald}; font-size:12px; font-weight:600; background:transparent; border:none;"
        )
        self.clock_label.setStyleSheet(
            f"color:{C.text}; font-size:18px; font-family:monospace; background:transparent; border:none;"
        )
        self.set_ai_status(self.ai_status.text() == "●")

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
        self.is_self = is_self
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.bubble = QFrame()
        self.bubble.setMaximumWidth(680)
        inner = QVBoxLayout(self.bubble)
        inner.setContentsMargins(14, 10, 14, 10)
        inner.setSpacing(4)

        header = QHBoxLayout()
        self.who = QLabel(sender)
        self.ts = QLabel(timestamp)
        header.addWidget(self.who)
        header.addWidget(self.ts)
        header.addStretch()
        inner.addLayout(header)

        self.main_text = QLabel(translated)
        self.main_text.setWordWrap(True)
        inner.addWidget(self.main_text)

        sub_box = QHBoxLayout()
        sub_box.setSpacing(6)
        self.lang_badge = QLabel(language)
        sub_box.addWidget(self.lang_badge)
        self.orig_text = QLabel(original)
        self.orig_text.setWordWrap(True)
        sub_box.addWidget(self.orig_text, 1)
        inner.addLayout(sub_box)

        self._font_size = font_size
        self.retheme()

        if is_self:
            outer.addStretch()
            outer.addWidget(self.bubble)
        else:
            outer.addWidget(self.bubble)
            outer.addStretch()

    def retheme(self):
        """Re-apply current C.* palette + font size without recreating this bubble."""
        bg = C.emerald_dark if self.is_self else C.panel
        border = C.emerald if self.is_self else C.border2
        radius_corner = "border-top-right-radius:4px;" if self.is_self else "border-top-left-radius:4px;"
        self.bubble.setStyleSheet(
            f"QFrame {{ background-color:{bg}; border:1px solid {border}; "
            f"border-radius:16px; {radius_corner} }}"
        )
        self.who.setStyleSheet(f"color:{C.text_dim}; font-weight:bold; font-size:11px; border:none;")
        self.ts.setStyleSheet(f"color:{C.text_dim2}; font-size:11px; border:none;")
        self.lang_badge.setStyleSheet(
            f"color:{C.text_dim}; background-color:{C.border}; border:none; "
            "border-radius:4px; padding:1px 5px; font-size:10px; font-weight:bold;"
        )
        self.set_font_size(self._font_size)

    def set_font_size(self, size):
        self._font_size = size
        sub_size = max(10, size - 4)
        self.main_text.setStyleSheet(f"color:{C.text}; font-size:{size}px; font-weight:500; border:none;")
        self.orig_text.setStyleSheet(f"color:{C.text_dim}; font-size:{sub_size}px; border:none;")


class LiveTransmitBubble(QWidget):
    """Bubble shown while PTT is held, containing the live waveform meter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.bubble = QFrame()
        self.bubble.setFixedWidth(360)
        inner = QVBoxLayout(self.bubble)
        inner.setContentsMargins(14, 10, 14, 10)

        header = QHBoxLayout()
        self.who = QLabel("MY VESSEL")
        self.status = QLabel("송신 중...")
        header.addWidget(self.who)
        header.addWidget(self.status)
        header.addStretch()
        inner.addLayout(header)

        self.meter = LiveMeterWidget(bar_count=BAR_COUNT, color=C.emerald)
        self.meter.setFixedHeight(36)
        inner.addWidget(self.meter)

        self.retheme()

        outer.addStretch()
        outer.addWidget(self.bubble)

    def retheme(self):
        self.bubble.setStyleSheet(
            f"QFrame {{ background-color:{C.emerald_dark}; border:1px solid {C.emerald}; "
            "border-radius:16px; border-top-right-radius:4px; }"
        )
        self.who.setStyleSheet(f"color:{C.emerald}; font-weight:bold; font-size:11px; border:none;")
        self.status.setStyleSheet(f"color:{C.text_dim2}; font-size:11px; border:none;")
        self.meter.color = QColor(C.emerald)

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

    def retheme(self):
        self.scroll.setStyleSheet(f"QScrollArea {{ background-color:{C.bg}; border:none; }}")
        self.content.setStyleSheet(f"background-color:{C.bg};")
        for bubble in self._bubbles:
            bubble.retheme()
        if self.live_bubble is not None:
            self.live_bubble.retheme()


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

        self._cards = []  # (card_frame, title_label) pairs, filled in by _card()

        title_row = QHBoxLayout()
        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(36, 36)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self.backRequested.emit)
        title_row.addWidget(self.back_btn)
        title_row.addSpacing(10)
        self.title_label = QLabel("설정 (Settings)")
        title_row.addWidget(self.title_label)
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
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        grid.addWidget(lang_card, 0, 1)

        # Font size card
        font_card, font_layout = self._card("자막 폰트 크기")
        font_row = QHBoxLayout()
        self.minus_btn = QPushButton("A-")
        self.plus_btn = QPushButton("A+")
        self.font_value_label = QLabel("18px")
        self.font_value_label.setAlignment(Qt.AlignCenter)
        for b in (self.minus_btn, self.plus_btn):
            b.setFixedSize(48, 48)
            b.setCursor(Qt.PointingHandCursor)
        self._font_size = 18
        self.minus_btn.clicked.connect(lambda: self._bump_font(-2))
        self.plus_btn.clicked.connect(lambda: self._bump_font(2))
        font_row.addWidget(self.minus_btn)
        font_row.addWidget(self.font_value_label, 1)
        font_row.addWidget(self.plus_btn)
        font_layout.addLayout(font_row)
        grid.addWidget(font_card, 1, 0)

        # Volume card
        vol_card, vol_layout = self._card("시스템 볼륨")
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(70)
        vol_layout.addWidget(self.vol_slider)
        grid.addWidget(vol_card, 1, 1)

        self.retheme()

    def _card(self, title_text):
        # NOTE: no QGraphicsDropShadowEffect here — on Jetson Nano's Maxwell GPU,
        # drop-shadow compositing is expensive and visibly drops frame rate on a
        # 1024x600 embedded display. A flat 1px border gives equal visual clarity
        # (card separation) at a fraction of the render cost.
        card = QFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        title = QLabel(title_text)
        layout.addWidget(title)
        self._cards.append((card, title))
        return card, layout

    def retheme(self):
        self.setStyleSheet(f"background-color:{C.bg};")
        self.back_btn.setStyleSheet(
            f"QPushButton {{ background-color:{C.panel}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; font-size:18px; font-weight:bold; }"
            f"QPushButton:hover {{ border-color:{C.cyan}; color:{C.cyan}; }}"
        )
        self.title_label.setStyleSheet(f"color:{C.text}; font-size:20px; font-weight:bold; border:none;")
        for card, title in self._cards:
            card.setStyleSheet(f"background-color:{C.panel}; border:1px solid {C.border}; border-radius:14px;")
            title.setStyleSheet(f"color:{C.text_dim}; font-size:12px; font-weight:bold; border:none;")
        self.channel_spin.setStyleSheet(
            f"QSpinBox {{ background-color:{C.bg}; color:{C.cyan}; border:1px solid {C.border2}; "
            "border-radius:10px; font-size:60px; font-weight:bold; padding:0px; }"
        )
        self._refresh_preset_styles()
        self.lang_combo.setStyleSheet(
            f"background-color:{C.bg}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; padding:8px; font-size:13px;"
        )
        self.font_value_label.setStyleSheet(f"color:{C.cyan}; font-size:18px; font-weight:bold; border:none;")
        for b in (self.minus_btn, self.plus_btn):
            b.setStyleSheet(
                f"QPushButton {{ background-color:{C.bg}; color:{C.text}; "
                f"border:1px solid {C.border2}; border-radius:8px; font-weight:bold; }}"
                f"QPushButton:hover {{ border-color:{C.cyan}; }}"
            )
        self.vol_slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background:{C.border2}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal {{ background:{C.cyan}; width:16px; margin:-6px 0; border-radius:8px; }}"
        )

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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        header = QHBoxLayout()
        self.title_label = QLabel("교신 기록 (Logs)")
        header.addWidget(self.title_label)
        header.addStretch()

        self.search = QLineEdit()
        self.search.setPlaceholderText("선박명, 키워드 검색")
        self.search.setFixedWidth(220)
        header.addWidget(self.search)
        outer.addLayout(header)
        outer.addSpacing(12)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시간", "선박명", "내용 (번역)"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        outer.addWidget(self.table)

        self.retheme()

    def retheme(self):
        self.setStyleSheet(f"background-color:{C.bg};")
        self.title_label.setStyleSheet(f"color:{C.text}; font-size:20px; font-weight:bold; border:none;")
        self.search.setStyleSheet(
            f"background-color:{C.panel}; color:{C.text}; border:1px solid {C.border2}; "
            "border-radius:8px; padding:6px;"
        )
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color:{C.panel}; color:{C.text}; border:1px solid {C.border}; "
            f"border-radius:12px; gridline-color:{C.border}; }}"
            f"QHeaderView::section {{ background-color:{C.panel_deep}; color:{C.text_dim}; "
            "border:none; padding:6px; font-weight:bold; font-size:11px; }"
        )

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

        self.logo = QLabel("⚠")
        self.logo.setFixedSize(58, 58)
        self.logo.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.logo, 0, Qt.AlignCenter)

        self.title_label = QLabel("비상 상황 감지")
        self.title_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.title_label)

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
        self.retheme()

    def retheme(self):
        self.logo.setStyleSheet(
            "QLabel { background: qradialgradient(cx:0.5, cy:0.4, radius:0.9, fx:0.5, fy:0.4, "
            f"stop:0 #fb923c, stop:1 {C.red}); border-radius:29px; color:white; font-size:24px; border:none; }}"
        )
        self.title_label.setStyleSheet(f"color:{C.red}; font-size:24px; font-weight:bold; border:none;")

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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)

        left = QHBoxLayout()
        left.setSpacing(10)
        self.nav_buttons = {}
        self._all_nav_buttons = []  # every plain nav-style button, for retheme()
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
        self.ptt_btn.pressed.connect(self._on_ptt_press)
        self.ptt_btn.released.connect(self._on_ptt_release)
        layout.addWidget(self.ptt_btn, 1)

        right = QHBoxLayout()
        self.mode_btn = self._nav_button("MODE")
        self.mode_btn.clicked.connect(self.modeRequested.emit)
        right.addWidget(self.mode_btn)
        layout.addLayout(right)

        self.retheme()

    def _nav_button(self, label):
        btn = QPushButton(label)
        btn.setFixedSize(76, 60)
        btn.setCursor(Qt.PointingHandCursor)
        self._all_nav_buttons.append(btn)
        return btn

    def retheme(self):
        self.setStyleSheet(f"background-color:{C.bar}; border-top:1px solid {C.border2};")
        for btn in self._all_nav_buttons:
            btn.setStyleSheet(
                f"QPushButton {{ background-color:{C.panel}; color:{C.text_dim}; "
                f"border:1px solid {C.border}; border-radius:10px; font-weight:bold; font-size:11px; }}"
                f"QPushButton:hover {{ background-color:#1e2636; }}"
            )
        self._ptt_style(pressed=self._ptt_active)

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

    def _on_ptt_press(self):
        self._ptt_active = True
        self._ptt_style(pressed=True)
        self.pttStarted.emit()

    def _on_ptt_release(self):
        self._ptt_active = False
        self._ptt_style(pressed=False)
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

    def retheme(self):
        """Restyle every child screen in place — no widget destruction/recreation.
        This is what keeps a theme switch instant on Jetson Nano instead of causing
        a visible freeze / CPU spike from tearing down and rebuilding the whole tree."""
        self.setStyleSheet(f"background-color:{C.bg};")
        self.top_bar.retheme()
        self.main_screen.retheme()
        self.settings_screen.retheme()
        self.log_screen.retheme()
        self.bottom_nav.retheme()

    def _toggle_screen(self, key):
        target = self._screen_index[key]
        current = self.content_stack.currentIndex()
        self.content_stack.setCurrentIndex(0 if current == target else target)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self, kiosk=True):
        super().__init__()
        self.setWindowTitle("SeaTalk AI VHF")
        self.kiosk = kiosk
        if kiosk:
            # Kiosk/embedded-touchscreen mode: no OS title bar or border.
            # Without this, the window manager adds decoration on top of our
            # fixed 1024x600 canvas, so on a panel that IS exactly 1024x600
            # physical pixels, that extra decoration pushes part of the
            # window off the visible screen — which is exactly the symptom
            # described ("전체적으로 화면 밖으로 밀림").
            self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        else:
            # Desktop dev/testing: keep a normal decorated window fixed at
            # the design resolution.
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

        # Emergency overlay floats above everything, manual geometry — sized
        # to match self, kept in sync by resizeEvent() below rather than the
        # fixed FRAME_W/FRAME_H constants (which may not match the real panel).
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the emergency overlay covering the whole window even if the
        # real screen resolution isn't exactly 1024x600.
        self.overlay.setGeometry(0, 0, self.width(), self.height())


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
        # [AI] 데모용 가짜 수신(4.5초마다 랜덤 문구)·강제 경보(18초) 비활성화 —
        #      실제 STT 연동 확인에 방해됨. 원본 데모 동작이 필요하면 아래 두 줄 복원.
        # self.incoming_timer.start(4500)
        # QTimer.singleShot(18000, self._trigger_emergency)

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
        self._retheme_ui()

    def _retheme_ui(self):
        # IMPORTANT (Jetson Nano perf): earlier this method tore down AppContainer and
        # EmergencyOverlay (deleteLater) and rebuilt the whole widget tree from scratch
        # on every theme switch. On a desktop that's imperceptible; on Jetson Nano's
        # constrained CPU/GPU it caused a 1-2s UI freeze, a momentary CPU spike, and
        # repeated allocation/deallocation churn (risk of RAM fragmentation/leaks over
        # a 24/7 uptime). Every widget that needs re-coloring now exposes its own
        # retheme() method that just re-applies stylesheets to the *existing* widgets,
        # so no state (messages, logs, current screen, etc.) needs to be
        # snapshotted/restored either.
        self.setStyleSheet(f"background-color:{C.bg};")
        self.app_container.retheme()
        self.overlay.retheme()

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
            phrase = random.choice(PHRASE_BANK)
            text_foreign = phrase.get(self.foreign_lang, phrase["EN"])
            # [AI] translation: foreign_lang -> Korean, so the local operator can read it
            text_ko = translate_to_korean(text_foreign, self.foreign_lang)
            self._add_message(
                sender,
                text_foreign,
                text_ko,
                self.foreign_lang,
                is_self=False,
            )

    # ---- PTT ----
    def _start_ptt(self):
        self.mic.start()
        self.app_container.top_bar.set_ai_status(self.mic.mode == "live")
        self.app_container.main_screen.show_live_bubble()
        # 100ms (~10fps) instead of 60ms (~16fps): visually smooth enough for a level
        # meter, but meaningfully lighter on Jetson Nano's CPU during PTT transmission.
        self.ptt_timer.start(100)

    def _ptt_tick(self):
        if self.mic.mode == "simulated":
            self.mic.tick_simulated()
        levels = self.mic.get_levels()
        self.app_container.main_screen.update_live_levels(levels)

    def _end_ptt(self):
        self.ptt_timer.stop()
        self.mic.stop()
        self.app_container.main_screen.hide_live_bubble()

        # [AI] STT: PTT 동안 캡처된 오디오 -> 한국어 텍스트 (파인튜닝 모델)
        text_ko = transcribe_audio(self.mic)
        if not text_ko:
            return                                   # 무음·너무 짧음 — 말풍선 생략
        # [AI] 화자 라벨: "여기는 ○○호" 자기호출이 있으면 선박명으로 표시
        sender = "MY VESSEL"
        try:
            from marine_speaker import SELF_CALL, _norm_name
            m = SELF_CALL.search(text_ko)
            if m:
                sender = _norm_name(m.group(1))
        except Exception:
            pass
        # [AI] 번역: 한국어 -> 설정(Settings)에서 고른 번역 언어 (아직 문구뱅크 스텁)
        text_translated = translate_text(text_ko, self.foreign_lang)

        self._add_message(
            sender,
            text_translated,
            text_ko,
            self.foreign_lang,
            is_self=True,
        )
        # [AI] 위험 감지: 실제 인식 텍스트 기반 경보 (데모 타이머 대체)
        try:
            from marine_danger import DangerAgent
            if not hasattr(self, "_danger_agent"):
                self._danger_agent = DangerAgent()
            rep = self._danger_agent.analyze(text_ko, speaker=sender)
            if rep and rep["level"] in ("DISTRESS", "URGENCY"):
                self._trigger_emergency()
        except Exception:
            # 모듈 없으면 최소한의 키워드 폴백
            low = text_ko.lower()
            if any(k in low for k in ("메이데이", "mayday", "팬팬", "pan-pan", "침수", "화재", "전복")):
                self._trigger_emergency()

    # ---- Emergency ----
    def _trigger_emergency(self):
        if self._emergency_fired:
            return
        self._emergency_fired = True
        self.overlay.setGeometry(0, 0, self.width(), self.height())
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

    # Default: kiosk mode (frameless, snapped to the real screen) — this is what the
    # 1024x600 touchscreen deployment needs. Pass --windowed for desktop dev/testing
    # in a normal decorated window at the design resolution instead.
    kiosk = "--windowed" not in sys.argv
    window = MainWindow(kiosk=kiosk)

    if kiosk:
        # Snap to the ACTUAL physical screen rather than trusting FRAME_W/FRAME_H.
        # This — combined with FramelessWindowHint above — is what keeps a
        # 1024x600-designed UI from being pushed off a 1024x600 (or any other
        # size) real touchscreen panel.
        screen = app.primaryScreen()
        window.setGeometry(screen.geometry())
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()