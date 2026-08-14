# -*- coding: utf-8 -*-
"""
marine_translate.py — 다국어 번역 확장 (EN/JA/ZH) + 조난 호출어 보호

기존 엔진(v6~)의 확정 스택을 유지하고 JA/ZH만 NLLB로 확장:
  한→영: Helsinki-NLP/opus-mt-ko-en   (300MB, 실측 검증 완료 — 유지)
  한→일/중, 영→한: facebook/nllb-200-distilled-600M (2.4GB 1개로 3방향 커버)

규칙 (기존 실측 교훈 그대로):
  * 조난 호출어(MAYDAY/PAN-PAN/SECURITE 계열)는 번역 전 보호 후 복원
    (NLLB가 "MAYDAY"를 "5월 5일"로 오역한 실측 사례)
  * NE(선박명) 마스킹 — "정수빈"→"I'm an integer" 사례 방지. 호출부호는 음차 유지
  * Jetson 배포 시 NLLB는 CTranslate2 int8 변환 권장 (fp32는 메모리 예산 초과 위험)

사용:
    tr = MarineTranslator(targets=["EN", "JA", "ZH"])
    out = tr.translate("메이데이, 기관실에 침수가 발생했습니다.", "JA")
"""
import re


# ── 번역 전 전처리: 숫자말 정규화 + 선박명 마스킹 (실측: 이게 없으면 JA/ZH 붕괴) ──
_DIG = {'공':'0','영':'0','일':'1','이':'2','삼':'3','사':'4','오':'5','육':'6','칠':'7','팔':'8','구':'9'}
_NUM_RE = re.compile('[공영일이삼사오육칠팔구]{2,}')   # 2자리 이상 숫자말만 (일상어 오염 방지)

def _num_normalize(text):
    """침로 이칠공도 → 침로 270도. SMCP 낭독식 숫자를 아라비아로."""
    return _NUM_RE.sub(lambda m: ''.join(_DIG[c] for c in m.group()), text)

def _mask_ships(text, roster):
    """로스터(user_dict+AIS)에 있는 선박명만 정확 매칭으로 VESSEL-n 치환.
    정규식 추측 마스킹은 실측에서 문장을 훼손해('여기는'까지 삼킴) 폐기."""
    names, out = [], text
    for name in sorted(roster, key=len, reverse=True):   # 긴 이름 먼저 (부분 겹침 방지)
        variants = [name, name.replace(" ", ""), name.replace("제7", "제칠")]
        for v in dict.fromkeys(variants):
            if v and v in out:
                if name not in names:
                    names.append(name)
                out = out.replace(v, f"VESSEL-{len(names)}")
    return out, names

def _unmask_ships(text, names):
    """번역기가 VESSEL-n을 변형한 사례까지 복원 (VESSELL-1, ウェッセル1号, 韦塞尔1 등)."""
    for i, n in enumerate(names):
        pat = re.compile(
            rf"(?:[Vv][EeSsLl]+[-\s]?{i+1}|ウェッセル\s?{i+1}\s?号?|ヴェッセル\s?{i+1}\s?号?|韦塞尔\s?{i+1})")
        text = pat.sub(n.replace(" ", ""), text)
    return text

NLLB_CODE = {"EN": "eng_Latn", "JA": "jpn_Jpan", "ZH": "zho_Hans", "KO": "kor_Hang"}

# 번역 전 보호(치환) → 번역 후 복원. 좌측은 정규식.
PROTECT = [
    (re.compile(r"메이\s?데이", re.I), "MAYDAY"),
    (re.compile(r"팬\s?팬|판\s?판", re.I), "PAN-PAN"),
    (re.compile(r"세큐리테|시큐리티", re.I), "SECURITE"),
]


class MarineTranslator:
    def __init__(self, targets=("EN",), device="cpu", roster=None):
        self.targets = [t.upper() for t in targets]
        self.device = device
        self.roster = list(roster or [])   # 선박명 목록 (user_dict + AIS)
        self._opus = None      # 한→영 (품질 우선)
        self._nllb = None      # 한→일/중 + 영→한

    # ── 지연 로드 (첫 사용 시에만 메모리 점유) ──
    def _load_opus(self):
        if self._opus is None:
            from transformers import MarianTokenizer, MarianMTModel
            tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            mdl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            self._opus = (tok, mdl)
        return self._opus

    def _load_nllb(self):
        if self._nllb is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self._nllb = (tok, mdl)
        return self._nllb

    # ── 조난 호출어 보호 ──
    @staticmethod
    def _protect(text):
        for pat, token in PROTECT:
            text = pat.sub(token, text)
        return text

    @staticmethod
    def _restore(text):
        # 번역기가 보호 토큰을 변형했을 흔한 사례 복원
        fixes = {"May Day": "MAYDAY", "Mayday": "MAYDAY", "5월 5일": "MAYDAY",
                 "五月": "MAYDAY", "梅伊德": "MAYDAY", "梅德": "MAYDAY", "メーデー": "MAYDAY", "メイデー": "MAYDAY",
                 "Pan Pan": "PAN-PAN", "pan pan": "PAN-PAN"}
        for k, v in fixes.items():
            text = text.replace(k, v)
        return text

    def translate(self, text_ko, target):
        """한국어 → target(EN/JA/ZH). 실패 시 원문 반환 (자막 공백 방지)."""
        target = target.upper()
        try:
            src = self._protect(_num_normalize(text_ko))
            src, ships = _mask_ships(src, self.roster)
            if target == "EN":
                tok, mdl = self._load_opus()
                ids = tok(src, return_tensors="pt", truncation=True, max_length=256)
                out = mdl.generate(**ids, max_length=256)
                res = tok.batch_decode(out, skip_special_tokens=True)[0]
            elif target in ("JA", "ZH"):
                tok, mdl = self._load_nllb()
                tok.src_lang = NLLB_CODE["KO"]
                # 문장 단위로 나눠 번역 (긴 입력에서 NLLB가 뒷 절을 누락하는 실측 문제 대응)
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", src) if s.strip()]
                outs = []
                for s in sents:
                    ids = tok(s, return_tensors="pt", truncation=True, max_length=256)
                    out = mdl.generate(**ids, max_length=256,
                                       forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODE[target]))
                    outs.append(tok.batch_decode(out, skip_special_tokens=True)[0])
                res = " ".join(outs)
            else:
                return text_ko
            return self._restore(_unmask_ships(res, ships))
        except Exception:
            return text_ko

    def to_korean(self, text, source="EN"):
        """외국어 → 한국어 (수신 자막용). NLLB 단일 모델."""
        try:
            src = self._protect(text)
            tok, mdl = self._load_nllb()
            tok.src_lang = NLLB_CODE.get(source.upper(), "eng_Latn")
            ids = tok(src, return_tensors="pt", truncation=True, max_length=256)
            out = mdl.generate(**ids, max_length=256,
                               forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODE["KO"]))
            return self._restore(tok.batch_decode(out, skip_special_tokens=True)[0])
        except Exception:
            return text
