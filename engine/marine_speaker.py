# -*- coding: utf-8 -*-
"""
marine_speaker.py — 무전 화자분리 (모델 없이 구조로 푸는 3단)

무전은 PTT 방식이라 동시발화가 없다. 그래서 무거운 화자분리 모델(EEND 등) 없이:
  ①역할: PTT 신호로 [본선] / [수신] 구분  (엔진 WS 규격의 speaker 필드 — 이미 있음)
  ②턴:   수신 음성은 스퀠치 무음으로 "한 번의 송신 = 한 턴" 분리 (엔진의 발화 단위와 동일)
  ③이름: 무전 규격상 화자가 자기 배를 스스로 밝힌다("여기는 ○○호") → STT 텍스트에서
        자기호출 패턴을 뽑아 턴에 선박명 라벨을 붙이고, 이후 무라벨 턴은 직전 대화
        상대 추정으로 채운다 (호출 구조: "A호, 여기는 B호" → 다음 응답자는 A호일 확률 높음)

사용:
    tracker = SpeakerTracker()
    label = tracker.assign(text, is_self=False)   # 턴마다 호출
    # label 예: "동해호", "부산VTS", "본선", "상대선(미상)"
"""
import re

# "여기는 ○○(호)" — 자기호출. 관제/도선 호출부호도 허용
SELF_CALL = re.compile(
    r"여기는\s*(?:여객선|화물선|어선|유조선|예인선)?\s*"
    r"([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,12}?(?:호|브이티에스|VTS|도선|해경))")  # 이름 내 공백 허용("제칠 해성호")
# 문두 호출 "○○호," — 수신자(상대) 이름
ADDRESSEE = re.compile(
    r"^\s*([가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{0,12}?(?:호|브이티에스|VTS|도선|해경))[\s,]")

def _norm_name(n):
    return n.replace("브이티에스", "VTS").replace(" ", "")

class SpeakerTracker:
    """턴 단위 화자 라벨러. STT 텍스트만으로 동작 (임베딩 불필요)."""

    def __init__(self):
        self.last_speaker = None     # 직전 턴 화자
        self.last_addressee = None   # 직전 턴이 부른 상대
        self.roster = []             # 등장한 화자 목록 (UI 표시용)

    def assign(self, text, is_self=False):
        """턴 텍스트 → 화자 라벨. is_self=True(PTT 송신)면 '본선' 고정."""
        if is_self:
            self._remember("본선", self._addressee_of(text))
            return "본선"

        m = SELF_CALL.search(text)
        if m:  # ③ 자기호출 — 가장 확실
            name = _norm_name(m.group(1))
            self._remember(name, self._addressee_of(text))
            return name

        # 자기호출이 없으면: 직전 턴이 부른 상대가 응답 중일 확률이 높다
        if self.last_addressee:
            name = self.last_addressee
            self._remember(name, self._addressee_of(text))
            return name
        return "상대선(미상)"

    def _addressee_of(self, text):
        # 문두 감탄사("어 동해호") 제거 후 매칭 — 상대명 오염 방지
        t = re.sub(r"^\s*(어|아|네|예|응|야)\s+", "", text)
        m = ADDRESSEE.match(t)
        return _norm_name(m.group(1)) if m else None

    def _remember(self, speaker, addressee):
        self.last_speaker = speaker
        self.last_addressee = addressee
        if speaker not in self.roster and speaker != "상대선(미상)":
            self.roster.append(speaker)


if __name__ == "__main__":
    # 세션12 (다화자: 동해호·남성호·새벽호) 시나리오 자가 테스트
    turns = [
        "남성호, 여기는 동해호. 감도 있나. 이상.",
        "어 동해호, 잘 들린다. 지금 어디냐. 이상.",          # 자기호출 없음 → 직전 호출 상대=남성호
        "나 지금 가덕도 서방 삼 해리다. 새벽호 지나가니까 항로 비켜라. 이상.",
        "여기는 여객선 새벽호. 항로 유지하겠습니다. 어선들 주의 바랍니다. 이상.",
        "새벽호 확인, 본선 우현으로 피하겠습니다. 이상.",
    ]
    expected = ["동해호", "남성호", "동해호", "새벽호", None]
    t = SpeakerTracker()
    for txt, exp in zip(turns, expected):
        got = t.assign(txt)
        mark = "OK" if (exp is None or got == exp) else "FAIL"
        print(f"[{mark}] {got:<10} | {txt[:34]}")
    print("로스터:", t.roster)
