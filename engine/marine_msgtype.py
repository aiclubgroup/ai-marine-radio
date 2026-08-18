#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marine_msgtype.py — 교신 유형 자동 분류.

근거: IMO 표준해사통신용어(SMCP)의 우선순위 분류 +  한국 VTS 실무.
  · 조난(Distress, MAYDAY)   : 침수·화재·충돌·좌초·전복·침몰·익수·퇴선
  · 긴급(Urgency, PAN-PAN)   : 기관고장·조타불능·표류
  · 안전(Safety, SECURITE)   : 기상·항행 경보(주의보·풍랑·안개)
  · 관제(Routine 입출항)      : VTS 입·출항 보고, 도선, 정박, 침로/속력 보고
  · 복창(Readback)           : SMCP 표준 응답 — "…확인"
  · 사담(Chatter)            : 위 어디에도 안 걸리는 일반 잡담(어선 등)
자막 앞에 [유형] 태그로 붙여 관제 흐름을 한눈에 보이게 한다.
"""
import re

_RULES = [
    ("조난", ["메이데이", "mayday", "조난", "구조 요청", "침수", "화재", "폭발",
              "충돌", "좌초", "전복", "침몰", "익수", "퇴선", "man overboard",
              "sinking", "flooding", "on fire"]),
    ("긴급", ["팬팬", "판판", "pan-pan", "pan pan", "기관 고장", "기관고장",
              "조타 불능", "조타불능", "표류", "adrift", "not under command"]),
    ("안전", ["세큐리테", "securite", "sécurité", "기상", "주의보", "경보",
              "풍랑", "안개", "항행 경보", "항행경보", "시정"]),
]
_KWANJE = ["vts", "브이티에스", "입항", "출항", "관제", "도선", "정박", "투묘",
           "예인", "침로", "속력", "감도", "채널", "묘박", "접안", "이안",
           "해경", "상황실", "말씀", "위치", "상황실", "응답", "말씀하십시오",
           "여기는", "여기", "감도 양호", "감도 있"]

def classify(text):
    """교신 텍스트 → 유형 태그 문자열. 우선순위: 조난>긴급>안전>복창>관제>사담."""
    if not text:
        return "사담"
    low = text.lower()
    # 1) 조난/긴급/안전 (SMCP 신호·상황어)
    for tag, kws in _RULES:
        if any(k in low for k in kws):
            return tag
    # 2) 복창 — SMCP 표준 응답. 받은 지시를 되읽어 "…확인/유지/변침하겠습니다".
    #    (조난/긴급/안전이 아닌 발화에서 '확인'·'복창' 계열 응답어가 있으면 복창)
    if ("확인" in text or "복창" in text
            or re.search(r"(유지|변침|감속|증속|착용|출동)\s*하겠습니다", text)):
        return "복창"
    # 3) 관제/입출항 (VTS 실무 키워드)
    if any(k in low for k in _KWANJE):
        return "관제"
    # 4) 그 외 = 사담
    return "사담"


if __name__ == "__main__":
    tests = [
        "메이데이 여기는 제칠해성호 침수 발생",
        "팬팬 본선 기관 고장으로 표류 중",
        "세큐리테 오륙도 부근 짙은 안개 항행 주의",
        "부산 VTS 여기는 태양호 입항 예정입니다 이상",
        "귀선 침로 삼공공도 확인. 이상.",
        "야 오늘 조업 어땠어 밥은 먹었냐",
    ]
    for t in tests:
        print(f"[{classify(t)}]  {t}")
