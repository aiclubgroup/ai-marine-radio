# -*- coding: utf-8 -*-
"""
marine_danger.py — AI Agent 위험분석 2단 (8월 필수목표 "AI Agent 1차 프로토타입")

구조 (확장 스코프 리서치의 2단 하이브리드 그대로):
  Stage1 (상시·즉시): 키워드 감지 → 있으면 무조건 경보 (재현율 우선, 기존 check_danger)
  Stage2 (이번 구현): 감지된 발화 + 최근 문맥 → 위험 등급 + 상황 요약 + SMCP 기반 대응 권고
     - 규칙 엔진이 기본 (LLM 없이 항상 동작 — 안전필수라 LLM 다운 시에도 등급은 나옴)
     - 소형 LLM 훅은 옵션 (있으면 요약문만 보강, 등급·권고는 규칙이 최종 결정 —
       안전필수 판단에 LLM 단독 결정 금지 원칙)

등급 (IMO 조난·긴급·안전 통신 체계 준용):
  DISTRESS (조난)  — MAYDAY, 침몰·전복·화재·익수자 등 인명·선박 즉시 위험
  URGENCY  (긴급)  — PAN-PAN, 기관고장·조타불능·의료 등 잠재 위험
  SAFETY   (안전)  — SECURITE, 기상·항행 경보
  WATCH    (주의)  — 호출어 없이 위험 어휘만 감지 (충돌 위험, 좌초 언급 등)

사용:
    agent = DangerAgent()
    report = agent.analyze(text, speaker="동해호")
    # report = {"level":"DISTRESS","score":95,"situation":"침수",
    #           "keywords":[...],"advice":[...], "speaker":"동해호"}
"""
import re

# ── 등급 결정 규칙 ──
CALLS_DISTRESS = ["메이데이", "mayday"]
CALLS_URGENCY = ["팬팬", "pan-pan", "판판"]
CALLS_SAFETY = ["세큐리테", "securite", "시큐리티"]

SITUATIONS = {  # 상황명: (키워드, 기본등급, 대응 권고)
    "침수":  (["침수", "물이 들어", "flooding"], "DISTRESS",
             ["침수 구획·정도 확인 요청", "펌프 가동 여부 확인", "인근 선박 구조 협조 준비", "해경(VHF 16) 전파"]),
    "화재":  (["화재", "불이", "fire", "연기"], "DISTRESS",
             ["화재 위치·규모 확인", "퇴선 준비 여부 확인", "풍상측 접근 금지 전파", "해경(VHF 16) 전파"]),
    "전복":  (["전복", "capsize", "기울"], "DISTRESS",
             ["선체 상태·경사 확인", "구명동의 착용 지시", "익수자 발생 대비 감시", "해경(VHF 16) 전파"]),
    "익수자": (["익수자", "사람이 빠", "man overboard", "실종"], "DISTRESS",
             ["최종 목격 위치·시각 확인", "윌리엄슨 턴 등 회수 기동", "인근 선박 수색 협조", "해경(VHF 16) 전파"]),
    "좌초":  (["좌초", "얹혔", "aground"], "DISTRESS",
             ["선저 손상·침수 여부 확인", "조석 확인 후 이초 판단", "주변 선박 접근 주의 전파"]),
    "충돌":  (["충돌", "collision", "부딪"], "DISTRESS",
             ["인명 피해·침수 여부 확인", "상대선 상태 확인", "위치 고정 및 전파"]),
    "퇴선":  (["퇴선", "abandon"], "DISTRESS",
             ["인원수·구명뗏목 확인", "EPIRB 작동 확인", "최종 위치 전파"]),
    "기관고장": (["기관 고장", "기관고장", "엔진 고장", "표류"], "URGENCY",
             ["표류 방향·속도 확인", "묘박 가능 수심인지 확인", "예인 필요 여부 판단"]),
    "조타불능": (["조타 불능", "조타불능", "키가"], "URGENCY",
             ["비상 조타 전환 여부 확인", "주변 선박 회피 전파"]),
    "의료":  (["의료", "환자", "부상"], "URGENCY",
             ["환자 상태·의식 확인", "의료 조언 채널 연결", "이송 필요 여부 판단"]),
    "기상":  (["풍랑", "태풍", "주의보", "경보"], "SAFETY",
             ["해당 해역 항행 선박 주의", "피항 여부 판단"]),
}


# ── 핵심 정보 추출 (친구 danger_agent.py에서 이식: 위치·인원·선박명) ──
PLACES = ["가덕도", "오륙도", "태종대", "감천", "영도", "부산항", "광안", "다대포", "송도", "해운대"]

def _find_vessel(text):
    m = re.search(r'([가-힣0-9]{2,8}호)', text)
    return m.group(1) if m else None

def _find_position(text):
    for p in PLACES:
        if p in text:
            m = re.search(p + r'\s*[가-힣]{0,3}방?\s*[영공일이삼사오육칠팔구십0-9]+\s*해리', text)
            return m.group(0) if m else p
    m = re.search(r'[영공일이삼사오육칠팔구십0-9]+\s*해리', text)
    return m.group(0) if m else None

def _find_persons(text):
    m = re.search(r'(선원|인원|승선|탑승)?\s*([영공일이삼사오육칠팔구십0-9]+)\s*명', text)
    return m.group(0).strip() if m else None

LEVEL_ORDER = ["WATCH", "SAFETY", "URGENCY", "DISTRESS"]
LEVEL_SCORE = {"DISTRESS": 95, "URGENCY": 70, "SAFETY": 40, "WATCH": 20}


class DangerAgent:
    def __init__(self, llm_summarize=None):
        """llm_summarize: 옵션 콜러블 (문맥 리스트)->요약문. 없으면 규칙 요약."""
        self.llm = llm_summarize
        self.context = []          # 최근 발화 (speaker, text)
        self.active = None         # 진행 중인 위험 상황 (등급 유지용)
        self.active_speaker = None # 그 상황의 당사자 (같은 화자에만 sticky 적용)

    def analyze(self, text, speaker=None):
        self.context.append((speaker or "?", text))
        self.context = self.context[-6:]
        low = text.lower()

        # 1) 호출어 → 등급 하한 확정. 명시적 호출어는 "새 상황 선언"이므로
        #    이전 상황의 등급 유지(sticky)를 리셋한다.
        level = None
        explicit_call = False
        if any(c in low for c in CALLS_DISTRESS):
            level, explicit_call = "DISTRESS", True
        elif any(c in low for c in CALLS_URGENCY):
            level, explicit_call = "URGENCY", True
        elif any(c in low for c in CALLS_SAFETY):
            level, explicit_call = "SAFETY", True
        if explicit_call:
            self.active = None

        # 2) 상황 어휘 → 상황명 + 등급 상향 + 권고
        hits, situations, advice = [], [], []
        for name, (kws, lv, adv) in SITUATIONS.items():
            found = [k for k in kws if k in low]
            if found:
                hits += found
                situations.append(name)
                advice += [a for a in adv if a not in advice]
                if level is None or LEVEL_ORDER.index(lv) > LEVEL_ORDER.index(level):
                    level = lv
        if level is None:
            if not hits:
                return None                    # 위험 신호 없음
            level = "WATCH"

        # 3) 같은 화자의 진행 중 상황이면 등급 유지 (경보 널뛰기 방지 — 후속 교신이
        #    호출어 없이 이어져도 등급이 떨어지지 않게). 다른 화자에는 전파하지 않는다.
        if (self.active and speaker == self.active_speaker
                and LEVEL_ORDER.index(self.active) > LEVEL_ORDER.index(level)):
            level = self.active
        self.active, self.active_speaker = level, speaker

        # 4) 요약: LLM 있으면 보강, 없으면 규칙 문장
        situation = "·".join(situations) if situations else "호출어 감지"
        summary = f"[{speaker or '미상'}] {situation} — \"{text[:40]}\""
        if self.llm:
            try:
                summary = self.llm(self.context) or summary
            except Exception:
                pass                            # LLM 실패해도 규칙 결과로 진행

        # 핵심 정보: 선박명·위치·인원 (있으면 요약에도 반영)
        vessel = _find_vessel(text)
        position = _find_position(text)
        persons = _find_persons(text)
        extra = " ".join(x for x in [vessel, f"위치 {position}" if position else None, persons] if x)
        if extra:
            summary = f"{summary} | {extra}"
        return {"level": level, "score": LEVEL_SCORE[level], "situation": situation,
                "keywords": hits, "advice": advice[:4], "speaker": speaker,
                "vessel": vessel, "position": position, "persons": persons,
                "summary": summary}

    def clear(self):
        self.active = None
        self.active_speaker = None
        self.context = []


if __name__ == "__main__":
    agent = DangerAgent()
    tests = [
        ("동해호", "나 지금 가덕도 서방 삼 해리다. 새벽호 지나가니까 항로 비켜라. 이상."),
        ("제칠해성호", "메이데이, 메이데이, 메이데이. 여기는 어선 제칠 해성호. 이상."),
        ("제칠해성호", "기관실에 침수가 발생했습니다. 배가 기울고 있습니다. 이상."),
        ("태평호", "팬팬, 팬팬, 팬팬. 여기는 화물선 태평호. 기관 고장으로 표류 중입니다. 이상."),
        ("부산VTS", "남해 동부 해상 풍랑주의보 발효. 항해에 주의 바랍니다. 이상."),
    ]
    for sp, tx in tests:
        r = agent.analyze(tx, sp)
        if r is None:
            print(f"[  -  ] {tx[:36]}")
        else:
            print(f"[{r['level']:<8}] {r['situation']:<10} 권고 {len(r['advice'])}건 | {tx[:32]}")
        if sp == "제칠해성호" and r and "침수" in r["situation"]:
            print("        권고:", " / ".join(r["advice"]))
    agent.clear()
