"""
SeaTalk AI VHF — 실엔진 연동판 (PySide6 UI + 서울팀 STT 엔진)

seatalk_ai_vhf.py 의 화면(위젯)을 그대로 재사용하고,
더미 시뮬레이션 대신 서울팀 STT 엔진 서버(engine/realtime_stt_web.py)에
WebSocket 으로 붙어서 실제 음성인식 결과를 표시하는 버전.

실행 순서 (같은 기기에서 터미널 2개):
    1) 엔진:  python engine/realtime_stt_web.py --model small --translate --no-browser
    2) UI:    python seatalk_ai_vhf_connected.py
       (다른 기기의 엔진에 붙을 때: python seatalk_ai_vhf_connected.py --url ws://<IP>:8765/ws)

동작 방식 (서울-부산 WS 규격):
    UI → 엔진:  {"type":"ptt","state":"down"|"up","speaker":"A"}
    엔진 → UI:  {"type":"status","state":"loading"|"ready"|"processing",...}
                {"type":"utterance","time","speaker","lang","text","translation",
                 "danger":[...], "proc_sec":...}

주의:
    - 마이크 녹음은 엔진(서버)이 직접 한다. UI는 PTT 신호만 보낸다.
    - 그래서 이 연동판의 PTT 레벨미터는 시뮬레이션 파형으로만 그린다.
      (UI와 엔진이 같은 마이크 장치를 동시에 열면 ALSA 충돌이 나기 때문)
    - PTT는 토글식(UI v2): 한 번 누르면 송신 시작, 다시 누르면 종료→인식.
    - MODE(주간/야간) 전환 시 UI가 통째로 재생성되므로, 연동 상태 표시와
      비상 오버레이 참조를 재적용한다 (_rebuild_ui 오버라이드).
    - 나중에 실제 무전기 PTT(GPIO)가 생기면 같은 WS 메시지를 쏘면 된다.
"""

import argparse
import json
import sys

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtWebSockets import QWebSocket

from seatalk_ai_vhf import MainWindow

DEFAULT_URL = "ws://127.0.0.1:8765/ws"
PTT_READY_TEXT = "🎙  PTT"


# ---------------------------------------------------------------------------
# WebSocket client for the Seoul STT engine
# ---------------------------------------------------------------------------
class EngineClient(QObject):
    utterance = Signal(dict)
    status = Signal(dict)
    history = Signal(list)
    connectedChanged = Signal(bool)

    def __init__(self, url=DEFAULT_URL, parent=None):
        super().__init__(parent)
        self.url = QUrl(url)
        self.is_open = False
        self.ws = QWebSocket()
        self.ws.connected.connect(self._on_connected)
        self.ws.disconnected.connect(self._on_disconnected)
        self.ws.textMessageReceived.connect(self._on_message)
        # 자동 재접속 (엔진이 늦게 뜨거나 죽었다 살아나도 UI는 그대로)
        self._retry = QTimer(self)
        self._retry.setInterval(3000)
        self._retry.timeout.connect(self._reconnect)

    def open(self):
        self.ws.open(self.url)
        self._retry.start()

    def _reconnect(self):
        if not self.is_open:
            self.ws.abort()
            self.ws.open(self.url)

    def _on_connected(self):
        self.is_open = True
        self.connectedChanged.emit(True)

    def _on_disconnected(self):
        self.is_open = False
        self.connectedChanged.emit(False)

    def _on_message(self, raw):
        try:
            m = json.loads(raw)
        except (ValueError, TypeError):
            return
        kind = m.get("type")
        if kind == "utterance":
            self.utterance.emit(m)
        elif kind == "status":
            self.status.emit(m)
        elif kind == "history":
            self.history.emit(m.get("items") or [])

    def send_ptt(self, down, speaker="A"):
        if self.is_open:
            self.ws.sendTextMessage(json.dumps(
                {"type": "ptt", "state": "down" if down else "up",
                 "speaker": speaker}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# MainWindow subclass: simulation off, real engine on
# ---------------------------------------------------------------------------
class ConnectedMainWindow(MainWindow):
    def __init__(self, url=DEFAULT_URL):
        super().__init__()
        self.setWindowTitle("SeaTalk AI VHF — LIVE")
        self._engine_ready = False

        self._grab_overlay_labels()

        self.client = EngineClient(url, parent=self)
        self.client.utterance.connect(self._on_utterance)
        self.client.status.connect(self._on_status)
        self.client.history.connect(self._on_history)
        self.client.connectedChanged.connect(self._on_conn_changed)
        self._wire_log_search()
        self._apply_engine_state()
        self.client.open()

    # ---- 비상 오버레이 라벨 참조 (테마 전환으로 오버레이가 재생성돼도 재확보) ----
    def _grab_overlay_labels(self):
        self._ov_vessel = self._ov_msg1 = self._ov_msg2 = self._ov_subtitle = None
        for lb in self.overlay.findChildren(QLabel):
            t = lb.text()
            if t.startswith("감지된 교신 내용"):
                self._ov_vessel = lb
            elif t.startswith('"충돌'):
                self._ov_msg1 = lb
            elif t.startswith('"Collision'):
                self._ov_msg2 = lb
            elif t == "COLLISION RISK DETECTED":
                self._ov_subtitle = lb

    # ---- 엔진 상태를 PTT 버튼에 반영 (v2 UI에는 힌트 라벨이 없어 버튼 자체로 표시) ----
    def _apply_engine_state(self, state_text=None):
        nav = self.app_container.bottom_nav
        if not getattr(self, "client", None) or not self.client.is_open:
            nav.ptt_btn.setEnabled(False)
            nav.ptt_btn.setText("⏳  엔진 연결 대기 중…")
        elif not self._engine_ready:
            nav.ptt_btn.setEnabled(False)
            nav.ptt_btn.setText(state_text or "⏳  엔진 로딩 중…")
        else:
            nav.ptt_btn.setEnabled(True)
            nav.ptt_btn.setText(state_text or PTT_READY_TEXT)

    def _force_ptt_off(self):
        """송신 중 연결이 끊기는 등의 상황에서 토글 상태를 강제로 해제."""
        nav = self.app_container.bottom_nav
        if nav._ptt_active:
            nav._ptt_active = False
            nav._ptt_style(pressed=False)
            self.ptt_timer.stop()
            self.mic.active = False
            self.mic.mode = "idle"
            self.app_container.main_screen.hide_live_bubble()

    # ---- 시뮬레이션 제거: 부팅 후 가짜 수신/가짜 비상 타이머를 시작하지 않음 ----
    def _on_boot_complete(self):
        self.root_stack.setCurrentIndex(1)

    def _maybe_incoming_message(self):
        pass  # 더미 수신 메시지 없음

    # ---- 테마(MODE) 전환: UI가 재생성되므로 연동 상태를 다시 입힌다 ----
    def _rebuild_ui(self):
        super()._rebuild_ui()
        self._grab_overlay_labels()
        self._wire_log_search()
        self._apply_engine_state()

    # ---- PTT (토글): 마이크는 엔진이 잡으므로 UI 미터는 시뮬레이션으로만 ----
    def _start_ptt(self):
        self.mic.active = True
        self.mic.mode = "simulated"  # 실마이크 열지 않음 (엔진과 장치 충돌 방지)
        self.app_container.top_bar.set_ai_status(self.client.is_open)
        self.app_container.main_screen.show_live_bubble()
        self.ptt_timer.start(60)
        self.client.send_ptt(True)

    def _end_ptt(self):
        self.ptt_timer.stop()
        self.mic.active = False
        self.mic.mode = "idle"
        self.app_container.main_screen.hide_live_bubble()
        self.client.send_ptt(False)
        # 결과는 엔진의 utterance 메시지로 도착 → _on_utterance 가 화면에 추가

    # ---- 과거 기록 재수신 (UI 재시작·재접속 시 엔진이 이번 세션 기록을 보내줌) ----
    def _on_history(self, items):
        self._clear_logs()  # 중복 방지: 표를 비우고 전체 재구성
        for m in items:
            speaker = m.get("speaker", "A")
            is_self = (speaker == "A")
            sender = "MY VESSEL" if is_self else f"수신 ({speaker})"
            text = m.get("text", "")
            translation = m.get("translation") or text
            lang = (m.get("lang") or "?").upper()
            self._add_message(sender, text, translation, lang, is_self)
        # 과거 기록 재생에서는 비상 오버레이를 띄우지 않는다 (이미 지나간 상황)

    # ---- 교신 기록 검색 (하드웨어팀 UI의 검색창에 필터 기능 연결) ----
    def _wire_log_search(self):
        from PySide6.QtWidgets import QLineEdit
        log_screen = self.app_container.log_screen
        box = log_screen.findChild(QLineEdit)
        if box is not None:
            box.textChanged.connect(lambda q: self._filter_logs(q))

    def _filter_logs(self, query):
        q = query.strip().lower()
        table = self.app_container.log_screen.table
        for r in range(table.rowCount()):
            texts = [table.item(r, c).text().lower() if table.item(r, c) else ""
                     for c in range(table.columnCount())]
            table.setRowHidden(r, bool(q) and not any(q in t for t in texts))

    # ---- 엔진 → UI ----
    def _on_utterance(self, m):
        speaker = m.get("speaker", "A")
        is_self = (speaker == "A")
        sender = "MY VESSEL" if is_self else f"수신 ({speaker})"
        text = m.get("text", "")
        translation = m.get("translation") or text
        lang = (m.get("lang") or "?").upper()
        self._add_message(sender, text, translation, lang, is_self)

        danger = m.get("danger") or []
        if danger:
            self._show_emergency(sender, text, danger)

        self._apply_engine_state()

    def _on_status(self, m):
        st = m.get("state")
        if st == "ready":
            self._engine_ready = True
            self._apply_engine_state()
        elif st == "processing":
            self._apply_engine_state("🌀  음성 인식 중…")
        elif st == "loading":
            self._engine_ready = False
            self._apply_engine_state(m.get("detail", "⏳  엔진 로딩 중…"))

    def _on_conn_changed(self, ok):
        if not ok:
            self._engine_ready = False
            self._force_ptt_off()
        self._apply_engine_state()

    # ---- 긴급 오버레이: 실제 감지 내용 표시 ----
    def _show_emergency(self, sender, text, danger):
        if self._ov_vessel is not None:
            self._ov_vessel.setText(f"감지된 교신 내용  ·  화자: {sender}")
        if self._ov_msg1 is not None:
            self._ov_msg1.setText(f'"{text}"')
        if self._ov_msg2 is not None:
            self._ov_msg2.setText(f'감지 키워드: {", ".join(danger)}')
        if self._ov_subtitle is not None:
            self._ov_subtitle.setText("EMERGENCY KEYWORD DETECTED")
        self.overlay.setGeometry(0, 0, self.width(), self.height())
        self.overlay.show()
        self.overlay.raise_()

    def _trigger_emergency(self):
        pass  # 더미 비상 타이머 무효화 (실감지로만 발동)


def main():
    ap = argparse.ArgumentParser(description="SeaTalk AI VHF — 실엔진 연동판")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"엔진 WebSocket 주소 (기본 {DEFAULT_URL})")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic" if sys.platform.startswith("win")
                      else "Noto Sans CJK KR", 10))
    window = ConnectedMainWindow(args.url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
