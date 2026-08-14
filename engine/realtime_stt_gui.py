#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_stt_gui.py — 해상무전기 실시간 STT 데모 (GUI, macOS/Windows/Linux 공통)
화면의 큰 [PTT] 버튼을 마우스/터치로 "누르고 있는 동안" 녹음 → 떼면 인식해 자막 표시.
스페이스바 홀드도 지원(키 리핏 보정 포함). 실제 무전 PTT·터치 UI와 동일한 방식이라
배 시연에 적합하고, macOS에서 별도 권한(sudo)이 필요 없다.
기능:
- PTT 버튼(마우스/터치 홀드) + 스페이스바 홀드로 녹음
- 실시간 자막(원문) 표시, 화자 라벨(A/B/C…) 선택
- 위험 키워드(MAYDAY/침수 등) 감지 시 빨간 경고 배너
- 옵션: 한↔영 번역 자막(--translate)
- 원본 음성 WAV + 교신 로그 CSV 자동 저장 → 배 데이터 수집 겸용
설치(맥 예시):
  brew install portaudio
  pip install faster-whisper sounddevice soundfile numpy
  # (번역까지) pip install ctranslate2 transformers sentencepiece
  # tkinter는 파이썬 기본 포함(맥은 python.org 배포판 권장; pyenv면 tcl-tk 필요)
실행:
  python realtime_stt_gui.py --model small
  python realtime_stt_gui.py --model small --translate
  python realtime_stt_gui.py --model base --device cpu     # 느린 노트북
시연 흐름: 앱 실행 → 모델 로딩 대기 → [PTT] 누른 채 말하기 → 떼면 자막.
화자 바꾸려면 상단 화자 버튼 클릭. 조난 문구("MAYDAY", "침수" 등) 말하면 경고 배너.
"""
import argparse
import csv
import queue
import threading
import time
from collections import deque
import numpy as np
SR = 16000
CH = 1
BLOCK = 1600
DANGER_KW = [
    "mayday", "pan-pan", "pan pan", "메이데이", "팬팬", "판판", "조난", "구조",
    "침수", "화재", "충돌", "좌초", "침몰", "익수", "전복", "퇴선", "man overboard",
    "sinking", "on fire", "flooding", "collision", "require assistance", "aground",
    "capsize", "abandon ship",
]
def _is_repetitive(text, min_reps=4):
    """같은 짧은 구절이 여러 번 반복되면 Whisper 환각으로 보고 True.
    (예: '안녕하십니까'가 수십 번 이어지는 경우)"""
    t = "".join(text.split())
    if len(t) < 6:
        return False
    for w in range(2, 9):                 # 2~8글자 구절 단위로 검사
        unit = t[:w]
        if unit * min_reps in t:          # 같은 구절이 4번 이상 연속
            return True
    # 고유 글자 비율이 극단적으로 낮아도 반복으로 간주
    if len(set(t)) / len(t) < 0.15:
        return True
    return False
def check_danger(text):
    low = text.lower()
    return [k for k in DANGER_KW if k in low]
class STTBackend:
    """Whisper + (옵션)번역. GUI를 막지 않도록 인식은 백그라운드 스레드에서 호출."""
    def __init__(self, model_name, compute_type, device, translate,
                 user_terms=None, corrections=None, correct_model=None):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        # 사용자 사전 (다글로/클로바식): 고유명사 목록 + 오인식 교정 쌍
        self.user_terms = list(user_terms or [])          # 인명·선박명 등
        self.corrections = dict(corrections or {})        # {"오인식": "정답"}
        self.hotwords = " ".join(self.user_terms) if self.user_terms else None
        # 대화 문맥 (직전 발화들 → Whisper initial_prompt로 주입, 자동 일관성 향상)
        self.context = deque(maxlen=3)
        self.use_context = True
        # 도메인 프롬프트: "해상 무전"이라는 맥락을 심어 조난 용어 인식 확률을 높임
        # (사전 치환이 아닌 소프트 바이어싱 — "메이데이"가 "매일"로 붕괴하는 문제 완화.
        #  일반 문장의 "매일"은 그대로 "매일"로 인식됨)
        self.domain_prompt = ("해상 무전 교신 기록. 메이데이, 메이데이, 팬팬, "
                              "침수, 화재, 좌초, 익수자 발생, 구조 요청, 감도 있습니까, 이상.")
        # (옵션) 소형 LLM 문맥 교정기
        self.corrector = None
        if correct_model:
            self._load_corrector(correct_model)
        self.translate = translate
        self.translators = {}
        if translate:
            self._load_translators()
    def _load_translators(self):
        """방향별 독립 로딩 — 한쪽이 실패해도 다른 쪽은 유지.
        한→영: Opus-MT(경량·검증됨) / 영→한: NLLB-600M
        (주의: Helsinki-NLP/opus-mt-en-ko는 존재하지 않고,
               opus-mt-tc-big-en-ko는 HF 변환본이 불량이라 사용 금지)"""
        try:
            from transformers import (MarianMTModel, MarianTokenizer,
                                      AutoModelForSeq2SeqLM, AutoTokenizer)
        except ImportError as e:
            print("[경고] transformers 미설치, 번역 생략:", e)
            print("       해결: pip install transformers sentencepiece torch")
            self.translate = False
            return
        # 한→영 (약 300MB)
        try:
            tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            mdl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            def ko2en(text, _t=tok, _m=mdl):
                b = _t([text], return_tensors="pt", truncation=True)
                return _t.decode(_m.generate(**b, max_length=128)[0],
                                 skip_special_tokens=True)
            self.translators["ko"] = ko2en
            print("[번역기] opus-mt-ko-en 로드 완료 (한→영)")
        except Exception as e:
            print("[경고] 한→영 번역기 로드 실패:", e)
        # 영→한 (NLLB-600M, 약 2.4GB — 첫 실행 시 다운로드 시간 소요)
        try:
            ntok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                                 src_lang="eng_Latn")
            nmdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            kor_id = ntok.convert_tokens_to_ids("kor_Hang")
            def en2ko(text, _t=ntok, _m=nmdl, _k=kor_id):
                b = _t([text], return_tensors="pt", truncation=True)
                return _t.decode(_m.generate(**b, forced_bos_token_id=_k,
                                             max_length=128)[0],
                                 skip_special_tokens=True)
            self.translators["en"] = en2ko
            # 일본어·중국어 → 한국어도 같은 NLLB로 추가 (다국어 번역, UI 약속 기능)
            def make_nllb(src_code, _t=ntok, _m=nmdl, _k=kor_id):
                def fn(text):
                    _t.src_lang = src_code
                    b = _t([text], return_tensors="pt", truncation=True)
                    return _t.decode(_m.generate(**b, forced_bos_token_id=_k, max_length=128)[0],
                                     skip_special_tokens=True)
                return fn
            self.translators["ja"] = make_nllb("jpn_Jpan")   # 일→한
            self.translators["zh"] = make_nllb("zho_Hans")   # 중→한
            print("[번역기] NLLB-600M 로드 완료 (영·일·중→한)")
        except Exception as e:
            print("[경고] NLLB 번역기 로드 실패:", e)
        if not self.translators:
            self.translate = False
    # 주력 언어(우선 언어). --langs 옵션으로 변경 가능, "all"이면 제한 없음.
    # 하드 차단이 아니라 "신뢰도 기반 소프트 제한":
    #   - 감지 언어가 주력 언어면 → 그대로 통과
    #   - 주력 언어 밖인데 확신도가 높으면(>= threshold) → 진짜 외국어로 인정, 통과
    #   - 주력 언어 밖이고 확신도가 낮으면 → 오감지로 판단, 주력 언어 중 확률 높은 쪽으로 재인식
    # (짧은 발화·한국인 억양 영어가 zh/ja로 오감지되는 문제는 대부분 저확신 케이스)
    def set_lang_policy(self, langs=("ko", "en"), threshold=0.80):
        self.preferred_langs = tuple(langs) if langs else None  # None = 제한 없음
        self.lang_threshold = threshold
    def _run_whisper(self, audio, language=None):
        """hotwords(사용자 사전) + 대화 문맥(직전 발화)을 반영해 인식.
        구버전 faster-whisper가 hotwords를 모르면 자동으로 빼고 재시도."""
        kwargs = dict(
            beam_size=5, vad_filter=False,     # 실시간 마이크 입력이 멈추는 문제로 끔(환각은 아래 옵션들로 잡음)
            condition_on_previous_text=False,  # 직전 발화에 '꽂혀서' 같은 말 반복하는 환각 차단
            no_repeat_ngram_size=3,            # 같은 3어절 연속 반복 금지
            compression_ratio_threshold=2.0,   # 반복 심한 결과는 버림
            temperature=0.0,
        )
        if language:
            kwargs["language"] = language
        # initial_prompt = 도메인 프롬프트(항상) + 직전 대화 문맥(있으면)
        parts = []
        if self.domain_prompt:
            parts.append(self.domain_prompt)
        if self.use_context and self.context:
            parts.append(" ".join(self.context)[-150:])
        if parts:
            kwargs["initial_prompt"] = " ".join(parts)
        if self.hotwords:
            try:
                return self.model.transcribe(audio, hotwords=self.hotwords, **kwargs)
            except TypeError:
                pass
        return self.model.transcribe(audio, **kwargs)
    def _load_corrector(self, model_id):
        """소형 LLM 문맥 교정기 로드 (--correct 옵션)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = "mps" if getattr(torch.backends, "mps", None) \
                and torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if dev == "mps" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id)
            try:
                mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev)
            except TypeError:  # 구버전 transformers
                mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(dev)
            self.corrector = (tok, mdl, dev)
            print(f"[AI 보정] {model_id} 로드 완료 ({dev})")
        except Exception as e:
            print("[경고] AI 보정 모델 로드 실패 (보정 없이 계속):", e)
    def llm_correct(self, text, lang):
        """문맥 기반 오류 교정 — 환각 방지 3중 가드 포함.
        (연구 근거: LLM 교정은 효과 크지만 과교정 위험 → 판정·검증·폴백 필수)"""
        if not self.corrector or lang not in ("ko", "en") or len(text) < 4:
            return text
        tok, mdl, dev = self.corrector
        try:
            import difflib
            ctx = " / ".join(self.context) if self.context else "(없음)"
            prompt = (
                "당신은 해상 무전 음성인식(STT) 결과 교정기다.\n"
                f"이전 교신 문맥: {ctx}\n"
                f"인식 문장: {text}\n\n"
                "발음이 비슷한 단어가 문맥상 잘못 인식된 부분만 고쳐라.\n"
                "규칙: 1) 내용 추가·삭제 금지 2) 숫자·좌표·MAYDAY 등 조난용어는 절대 변경 금지 "
                "3) 오류가 없으면 원문을 그대로 출력 4) 교정된 문장만 한 줄로 출력"
            )
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            max_new = max(48, int(len(text) * 1.2))
            if hasattr(enc, "items"):  # 신버전 transformers: dict 반환
                enc = {k: v.to(dev) for k, v in enc.items()}
                in_len = enc["input_ids"].shape[1]
                out = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False)
            else:                       # 구버전: 텐서 반환
                enc = enc.to(dev)
                in_len = enc.shape[1]
                out = mdl.generate(enc, max_new_tokens=max_new, do_sample=False)
            fixed = tok.decode(out[0][in_len:],
                               skip_special_tokens=True).strip().strip('"').strip()
            fixed = fixed.splitlines()[0].strip() if fixed else ""
            # 가드 ①: 빈 출력/과도한 변경(원문과 60% 미만 유사) → 원문 유지
            if not fixed or difflib.SequenceMatcher(None, text, fixed).ratio() < 0.6:
                return text
            # 가드 ②: 원문의 위험 키워드가 교정본에서 사라지면 → 원문 유지
            if set(check_danger(text)) - set(check_danger(fixed)):
                return text
            # 가드 ③: 원문의 숫자가 바뀌면 → 원문 유지
            import re
            if re.findall(r"\d+", text) != re.findall(r"\d+", fixed):
                return text
            if fixed != text:
                print(f"[AI보정] {text} → {fixed}")
            return fixed
        except Exception as e:
            print("[경고] AI 보정 실패(원문 유지):", e)
            return text
    def apply_corrections(self, text):
        """사용자 사전의 교정 쌍(오인식=>정답)을 결정적으로 치환 (다글로 '발음 유사 수정' 방식의 단순판)."""
        for wrong, right in self.corrections.items():
            if wrong in text:
                text = text.replace(wrong, right)
        return text
    def transcribe(self, audio):
        if not hasattr(self, "preferred_langs"):
            self.set_lang_policy()
        segs, info = self._run_whisper(audio)
        segs = list(segs)
        text = " ".join(s.text for s in segs).strip()
        lang = info.language
        if self.preferred_langs and lang not in self.preferred_langs:
            probs = dict(getattr(info, "all_language_probs", None) or [])
            detected_prob = probs.get(lang, 0.0)
            if detected_prob < self.lang_threshold:
                # 확신 낮은 타 언어 감지 → 주력 언어로 재인식
                best = max(self.preferred_langs, key=lambda l: probs.get(l, 0.0)) \
                    if probs else self.preferred_langs[0]
                segs, info = self._run_whisper(audio, language=best)
                segs = list(segs)
                text = " ".join(s.text for s in segs).strip()
                lang = best
            # else: 확신 높은 진짜 외국어 → 그대로 통과 (다국어 확장 대비)
        # 잡음 판정 지표 노출 — rtl_stt 등 호출측이 환각 필터에 사용
        # (ver2_ai RadioBridge와 같은 집계: 신뢰도=세그먼트 평균, no_speech=최대값)
        self.last_avg_logprob = (sum(s.avg_logprob for s in segs) / len(segs)) if segs else 0.0
        self.last_no_speech = max((getattr(s, "no_speech_prob", 0.0) for s in segs), default=1.0)
        if _is_repetitive(text):                 # 반복 환각이면 한 구절만 남김 (화면 도배 방지)
            u = "".join(text.split())
            for w in range(2, 9):
                if u[:w] * 4 in u:
                    text = u[:w]; break
        text = self.apply_corrections(text)      # 1차: 사전 기반 결정적 교정
        text = self.llm_correct(text, lang)      # 2차: (옵션) LLM 문맥 교정
        if text and not _is_repetitive(text):
            self.context.append(text)            # 대화 문맥 축적 → 다음 인식에 반영 (환각 반복은 저장 안 함)
        # 언어 라벨 최종 판정: 모델 감지 대신 결과 텍스트의 문자 구성으로 확정.
        # (파인튜닝 인코더가 언어감지 분포를 틀어 한국어를 en으로 찍는 실측 문제 —
        #  라벨이 틀리면 번역 방향까지 틀어지므로 텍스트 기준이 가장 확실)
        if text:
            import re as _re
            hang = len(_re.findall(r"[가-힣]", text))
            latin = len(_re.findall(r"[A-Za-z]", text))
            if hang >= 5 or (hang >= 3 and hang >= latin):
                lang = "ko"      # 한글 5자↑면 한국어 문장 (MAYDAY 등 라틴 혼재 허용)
            elif latin >= 6 and latin > hang * 2:
                lang = "en"
        return text, lang
    def translate_text(self, text, lang):
        fn = self.translators.get(lang)
        if not self.translate or fn is None:
            return ""
        try:
            import re
            prefix = ""
            # ① SMCP 조난 호출어(MAYDAY 등)는 번역하지 않고 보존
            m = re.match(r"^((?:MAYDAY|PAN[- ]PAN|SECURITE|SÉCURITÉ|메이데이)[,.\s]*)+",
                         text, re.IGNORECASE)
            if m:
                prefix = m.group(0).strip().rstrip(",.") + " — "
                text = text[m.end():].strip()
                if not text:
                    return prefix.rstrip(" —")
            # ② 고유명사(인명·선박명) 플레이스홀더 보호 — 번역 후 원어 복원
            #    ("정수빈입니다" → "I'm an integer" 같은 오역을 구조적으로 차단)
            protected = []
            for i, term in enumerate(self.user_terms):
                if term in text:
                    ph = f"NE{i + 1}"
                    text = text.replace(term, ph)
                    protected.append((ph, term))
            out = fn(text)
            for ph, term in protected:
                out = re.sub(ph, term, out, flags=re.IGNORECASE)
            return (prefix + out).strip()
        except Exception as e:
            print("[경고] 번역 실패:", e)
            return ""
def build_gui(args):
    import tkinter as tk
    from tkinter import scrolledtext
    import sounddevice as sd
    import soundfile as sf
    root = tk.Tk()
    root.title("해상무전기 실시간 STT 데모")
    root.geometry("820x620")
    root.configure(bg="#0b1f33")
    state = {
        "backend": None, "recording": False, "frames": [], "stream": None,
        "speaker": "A", "space_down": False, "release_job": None, "utt": 0,
    }
    # 로그/오디오 저장 준비
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(f"log_{stamp}.csv", "w", newline="", encoding="utf-8-sig")
    logw = csv.writer(logf)
    logw.writerow(["time", "speaker", "lang", "text", "translation", "danger", "wav"])
    logf.flush()
    # ── 상단: 상태 + 화자 선택 ──────────────────────────────────
    top = tk.Frame(root, bg="#0b1f33"); top.pack(fill="x", padx=16, pady=(14, 6))
    status = tk.Label(top, text="모델 로딩 중…", font=("Malgun Gothic", 13, "bold"),
                      fg="#ffd54a", bg="#0b1f33")
    status.pack(side="left")
    spk_frame = tk.Frame(top, bg="#0b1f33"); spk_frame.pack(side="right")
    tk.Label(spk_frame, text="화자:", font=("Malgun Gothic", 12), fg="#cfe0f0", bg="#0b1f33").pack(side="left")
    spk_var = tk.StringVar(value="A")
    # macOS 기본(aqua) 버튼은 bg/fg 색 지정을 무시하므로,
    # 선택 표시는 색이 아니라 텍스트(● 표시)와 눌림(relief)으로 한다 — 전 OS 공통 동작
    def set_speaker(s):
        state["speaker"] = s; spk_var.set(s)
        for b in spk_btns:
            if b._s == s:
                b.configure(text=f"● {b._s}", relief="sunken", default="active")
            else:
                b.configure(text=b._s, relief="raised", default="normal")
    spk_btns = []
    for s in ["A", "B", "C"]:
        b = tk.Button(spk_frame, text=s, width=4, font=("Malgun Gothic", 12, "bold"),
                      command=lambda s=s: set_speaker(s))
        b._s = s; b.pack(side="left", padx=3); spk_btns.append(b)
    set_speaker("A")
    # ── 경고 배너 ───────────────────────────────────────────────
    warn = tk.Label(root, text="", font=("Malgun Gothic", 14, "bold"),
                    fg="white", bg="#0b1f33")
    warn.pack(fill="x", padx=16)
    # ── 자막 영역 ───────────────────────────────────────────────
    subs = scrolledtext.ScrolledText(root, font=("Malgun Gothic", 15), wrap="word",
                                     bg="#12263b", fg="#eaf2fb", relief="flat",
                                     insertbackground="#eaf2fb", height=13)
    subs.pack(fill="both", expand=True, padx=16, pady=10)
    subs.tag_configure("meta", foreground="#7fa8cc", font=("Malgun Gothic", 11))
    subs.tag_configure("trans", foreground="#9fd0a0", font=("Malgun Gothic", 13))
    subs.tag_configure("danger", foreground="#ff6b6b", font=("Malgun Gothic", 13, "bold"))
    subs.configure(state="disabled")
    def add_line(speaker, text, lang, trans, hits):
        subs.configure(state="normal")
        subs.insert("end", f"[{time.strftime('%H:%M:%S')}] 화자 {speaker} ({lang})\n", "meta")
        subs.insert("end", f"  {text}\n")
        if trans:
            subs.insert("end", f"  ↳ {trans}\n", "trans")
        if hits:
            subs.insert("end", f"  ⚠ 위험 감지: {', '.join(hits)}\n", "danger")
        subs.insert("end", "\n")
        subs.see("end"); subs.configure(state="disabled")
    # ── PTT 버튼 ────────────────────────────────────────────────
    ptt = tk.Label(root, text="🎙  PTT — 누르고 말하기", font=("Malgun Gothic", 20, "bold"),
                   fg="white", bg="#2f6f3f", height=2, relief="raised", bd=3)
    ptt.pack(fill="x", padx=16, pady=(0, 8))
    hint = tk.Label(root, text="버튼을 누른 채 말하고 떼세요 (스페이스바 홀드도 가능)",
                    font=("Malgun Gothic", 10), fg="#9fb6cc", bg="#0b1f33")
    hint.pack(pady=(0, 12))
    # ── 구형 Tk(8.5, macOS 시스템 Tk) 호환 모드 ────────────────────────
    # 구형 aqua Tk는 위젯 배경색을 무시해 "흰 바탕 + 흰 글자"가 되어 버림.
    # 감지되면 시스템 기본(밝은) 테마로 자동 전환해 어떤 맥에서도 보이게 한다.
    # (다크테마를 원하면 python.org에서 Python 3.12 설치 후 venv 재생성 — Tk 8.6)
    if tk.TkVersion < 8.6:
        print("[안내] 구형 Tk 감지 → 밝은 테마로 자동 전환 (다크테마는 python.org 파이썬 필요)")
        _tmp = tk.Label(root); _bg = _tmp.cget("bg"); _tmp.destroy()
        for w in (root, top, spk_frame, warn, hint, status):
            try: w.configure(bg=_bg)
            except Exception: pass
        for w in spk_frame.winfo_children():
            try: w.configure(bg=_bg, fg="black")
            except Exception: pass
        hint.configure(fg="#666666")
        warn.configure(fg="#c00000")
        subs.configure(bg="white", fg="black", insertbackground="black")
        subs.tag_configure("meta", foreground="#486e8f")
        subs.tag_configure("trans", foreground="#1f7a2f")
        subs.tag_configure("danger", foreground="#c00000")
        ptt.configure(bg=_bg, fg="black")
        # 이후 코드가 지정하는 밝은 글자색을 밝은 배경용 진한 색으로 자동 변환
        _remap = {"#ffd54a": "#b36b00", "#8fe08f": "#1f8a3d",
                  "#ff6b6b": "#c00000", "white": "black"}
        for _w in (status, ptt, warn):
            _orig_cfg = _w.configure
            def _make(orig):
                def _cfg(**kw):
                    if "fg" in kw:
                        kw["fg"] = _remap.get(kw["fg"], kw["fg"])
                    kw.pop("bg", None)  # 배경색 변경은 무시(구형 Tk에서 깨짐)
                    return orig(**kw)
                return _cfg
            _w.configure = _make(_orig_cfg)
        status.configure(fg="#ffd54a")  # → 자동으로 진한 주황으로 변환됨
    # ── 녹음 제어 ───────────────────────────────────────────────
    def start_record(_evt=None):
        if state["backend"] is None or state["recording"]:
            return
        state["recording"] = True; state["frames"] = []
        ptt.configure(bg="#c0392b", text="● 녹음 중… (떼면 인식)")
        warn.configure(text="", bg="#0b1f33")
        def cb(indata, f, t, s): state["frames"].append(indata[:, 0].copy())
        state["stream"] = sd.InputStream(samplerate=SR, channels=CH, blocksize=BLOCK, callback=cb)
        state["stream"].start()
    def stop_record(_evt=None):
        if not state["recording"]:
            return
        state["recording"] = False
        ptt.configure(bg="#2f6f3f", text="🎙  PTT — 누르고 말하기")
        try:
            state["stream"].stop(); state["stream"].close()
        except Exception:
            pass
        frames = state["frames"]; state["frames"] = []
        if not frames:
            return
        audio = np.concatenate(frames).astype(np.float32)
        if len(audio) < SR * 0.3:
            return
        speaker = state["speaker"]
        state["utt"] += 1
        wavname = f"rec_{stamp}_{state['utt']:03d}_{speaker}.wav"
        sf.write(wavname, audio, SR)  # 원본 음성 저장(배 데이터 수집 겸용)
        status.configure(text="인식 중…", fg="#ffd54a")
        threading.Thread(target=do_transcribe, args=(audio, speaker, wavname), daemon=True).start()
    def do_transcribe(audio, speaker, wavname):
        # (옵션) 노이즈 제거 전처리 — 원본 WAV는 이미 저장됨(데이터 보존), 인식에만 적용
        if getattr(args, "denoise", False):
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=SR).astype(np.float32)
            except Exception as e:
                print("[경고] 노이즈 제거 실패(원음으로 인식):", e)
        text, lang = state["backend"].transcribe(audio)
        trans = state["backend"].translate_text(text, lang) if text else ""
        hits = check_danger(text) if text else []
        def ui():
            status.configure(text="준비 완료 — PTT를 누르세요", fg="#8fe08f")
            if text:
                add_line(speaker, text, lang, trans, hits)
                logw.writerow([time.strftime("%H:%M:%S"), speaker, lang, text, trans,
                               ";".join(hits), wavname]); logf.flush()
                if hits:
                    warn.configure(text=f"⚠ 위험 상황 감지: {', '.join(hits)} — 경고 알림",
                                   bg="#c0392b")
        root.after(0, ui)
    # 마우스/터치 홀드
    ptt.bind("<ButtonPress-1>", start_record)
    ptt.bind("<ButtonRelease-1>", stop_record)
    # 스페이스바 홀드 (키 리핏 보정: 릴리즈 후 40ms 내 재입력이면 홀드 유지로 간주)
    def space_press(_e=None):
        if state["release_job"] is not None:
            root.after_cancel(state["release_job"]); state["release_job"] = None
        if not state["space_down"]:
            state["space_down"] = True; start_record()
    def space_release(_e=None):
        def real_release():
            state["space_down"] = False; state["release_job"] = None; stop_record()
        state["release_job"] = root.after(40, real_release)
    root.bind("<KeyPress-space>", space_press)
    root.bind("<KeyRelease-space>", space_release)
    # ── 백엔드 로딩(백그라운드) ─────────────────────────────────
    def load_backend():
        try:
            terms, corrections = load_user_dict(args.dict)
            be = STTBackend(args.model, args.compute_type, args.device, args.translate,
                            user_terms=terms, corrections=corrections,
                            correct_model=args.correct_model if args.correct else None)
            be.use_context = not args.no_context
            if getattr(args, "no_domain_prompt", False):
                be.domain_prompt = None
            langs = None if args.langs.strip().lower() == "all" \
                else tuple(l.strip() for l in args.langs.split(",") if l.strip())
            be.set_lang_policy(langs, args.lang_threshold)
            def ok():
                state["backend"] = be
                status.configure(text="준비 완료 — PTT를 누르세요", fg="#8fe08f")
            root.after(0, ok)
        except Exception as e:
            root.after(0, lambda: status.configure(text=f"로딩 실패: {e}", fg="#ff6b6b"))
    threading.Thread(target=load_backend, daemon=True).start()
    def on_close():
        try: logf.close()
        except Exception: pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    print(f"[교신 로그] log_{stamp}.csv, 음성 rec_{stamp}_*.wav 저장됩니다.")
    root.mainloop()
def load_user_dict(path):
    """사용자 사전 파일 로드.
    형식: 한 줄에 하나 — 고유명사(인명·선박명) 또는 '오인식=>정답' 교정 쌍. #는 주석.
    반환: (terms 리스트, corrections 딕셔너리)"""
    terms, corrections = [], {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                if "=>" in line:
                    wrong, right = [x.strip() for x in line.split("=>", 1)]
                    if wrong and right:
                        corrections[wrong] = right
                else:
                    terms.append(line)
        print(f"[사용자 사전] {path}: 고유명사 {len(terms)}개, 교정쌍 {len(corrections)}개 로드")
    except FileNotFoundError:
        print(f"[안내] 사용자 사전 없음({path}) — 만들면 고유명사 인식·번역이 좋아집니다")
    return terms, corrections
def main():
    ap = argparse.ArgumentParser(description="해상무전기 실시간 STT 데모 (GUI)")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium (기본 small)")
    ap.add_argument("--compute-type", default="int8", help="int8/float16/float32")
    ap.add_argument("--device", default="auto", help="auto/cuda/cpu")
    ap.add_argument("--translate", action="store_true", help="한↔영 번역 자막 표시")
    ap.add_argument("--langs", default="ko,en",
                    help="주력 언어 목록(쉼표 구분, 기본 ko,en). 'all'이면 제한 없음")
    ap.add_argument("--lang-threshold", type=float, default=0.80,
                    help="타 언어 인정 확신도 임계값(기본 0.80)")
    ap.add_argument("--dict", default="user_dict.txt",
                    help="사용자 사전 파일 (기본 user_dict.txt)")
    ap.add_argument("--correct", action="store_true",
                    help="소형 LLM 문맥 교정 켜기 (첫 실행 시 모델 ~3GB 다운로드)")
    ap.add_argument("--correct-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="교정용 LLM (기본 Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--no-context", action="store_true",
                    help="대화 문맥 주입 끄기")
    ap.add_argument("--no-domain-prompt", action="store_true",
                    help="해상 도메인 프롬프트 끄기 (비교 실험용)")
    ap.add_argument("--denoise", action="store_true",
                    help="노이즈 제거 전처리 켜기 (전후 비교 실험용 — 연구상 오히려 "
                         "성능이 떨어질 수 있음. pip install noisereduce 필요)")
    args = ap.parse_args()
    build_gui(args)
if __name__ == "__main__":
    main()
