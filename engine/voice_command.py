#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_command.py — 음성으로 시스템 조작 (핸즈프리 관제).

PTT로 말한 게 '교신'이 아니라 '시스템 명령'이면(예: "로그 보여줘"),
자막에 띄우지 않고 화면 전환 등 동작을 실행하도록 명령을 인식한다.

맥락(어투) 대응: 명령마다 여러 표현을 등록 + 짧은 발화만 명령으로 간주
                (실제 교신은 길고 명령은 짧다 → 오작동 억제).

반환: 인식되면 {"action": "...", "arg": ...}, 아니면 None
"""
import re

# action: [표현들]  — 어느 하나가 발화에 포함되면 그 명령
_COMMANDS = {
    "view_log":  ["로그", "기록", "교신 기록", "교신기록", "log"],
    "view_main": ["메인", "홈", "처음", "자막", "메인 화면", "메인화면", "본 화면"],
    "view_sys":  ["시스템", "상태", "시스템 상태"],
    "clear":     ["지워", "지우기", "클리어", "삭제", "비워", "clear"],
    "quit":      ["종료", "꺼", "끝내", "끄기", "닫아", "quit", "exit"],
    "trans_on":  ["번역 켜", "번역켜", "번역 온", "translate on"],
    "trans_off": ["번역 꺼", "번역꺼", "번역 오프", "translate off"],
}
# (수신 선박 목록은 음성 대신 SYS 화면에서 확인 — STT 오인식 리스크 제거)
# 이 동작들은 "선박명 목록" 같은 인자를 뽑을 수도 있으나 지금은 단순 화면전환만.

def parse(text, max_len=16):
    """발화 텍스트 → 명령 dict 또는 None.
    - 공백 제거 길이가 max_len 이하일 때만 명령 후보로 봄(긴 교신 오인 방지).
    - '이상' 같은 무전 종결어는 떼고 판단."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"[,\.。~!\?]+", " ", t)
    t = re.sub(r"\b이상\b|\b오버\b|\bover\b", "", t, flags=re.IGNORECASE)
    compact = "".join(t.split())
    if len(compact) > max_len:          # 길면 교신으로 간주 → 명령 아님
        return None
    low = t.lower()
    # 번역 on/off는 '켜/꺼'가 붙은 것부터 먼저(부분매칭 충돌 방지)
    order = ["quit", "trans_off", "trans_on", "view_log", "view_sys", "view_main", "clear"]
    for action in order:
        for kw in _COMMANDS[action]:
            if kw.replace(" ", "") in compact or kw in low:
                return {"action": action}
    return None


if __name__ == "__main__":
    tests = [
        "로그 보여줘", "기록 좀 불러와", "메인 화면으로 가줘", "시스템 상태",
        "화면 지워", "종료해", "번역 꺼줘", "번역 켜",
        "부산 VTS 여기는 태양호 입항 예정입니다 이상",   # 교신 → None
        "귀선 침로 삼공공도 확인 이상",                    # 교신 → None
    ]
    for t in tests:
        print(f"{parse(t)}   ←  {t}")
