#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diarize.py — PTT 발화 단위 화자 자동 판별 (성문 임베딩 기반).

기존 데모는 화자를 UI 버튼(A/B)으로 '사람이' 골랐다 → AI가 목소리를 구분하지 않았다.
이 모듈은 발화(PTT 한 번)마다 음성 임베딩(성문)을 뽑아, 지금까지 등장한 화자들과
코사인 유사도로 비교해 '가장 가까운 화자'로 배정하거나 '새 화자'로 자동 등록한다.

- 임베딩 모델: resemblyzer(가벼움) 우선, 없으면 speechbrain ECAPA, 둘 다 없으면 비활성.
- 비활성 시(모델 미설치)엔 예외 없이 '수동 라벨 그대로'를 돌려줘 데모가 계속 동작.
- 실제 무전은 호출부호("여기는 부산 해경")로 화자가 드러나므로, 전사 힌트로 이름 매핑도 지원.

사용:
    reg = SpeakerRegistry(threshold=0.75)
    label = reg.identify(audio_float32_16k, hint_text="여기는 부산 해경 …")
"""
import numpy as np


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SpeakerRegistry:
    def __init__(self, threshold=0.75, sr=16000, name_hints=True):
        self.threshold = threshold          # 이 이상 유사하면 같은 화자
        self.sr = sr
        self.name_hints = name_hints
        self.speakers = []                   # [{"label":str, "centroid":np.array, "n":int}]
        self.encoder = None
        self._kind = None
        self._load_encoder()

    # ── 임베딩 모델 로드 (가벼운 것부터, 실패해도 데모 안 죽음) ──
    def _load_encoder(self):
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(verbose=False)
            self._kind = "resemblyzer"
            print("[diarize] resemblyzer 성문 임베딩 로드됨")
            return
        except Exception:
            pass
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/spkrec-ecapa")
            self._kind = "speechbrain"
            print("[diarize] speechbrain ECAPA 임베딩 로드됨")
            return
        except Exception as e:
            print(f"[diarize] 임베딩 모델 없음 → 화자분리 비활성(수동 라벨 사용). "
                  f"설치: pip install resemblyzer  ({e})")

    @property
    def enabled(self):
        return self.encoder is not None

    def _embed(self, audio):
        a = np.asarray(audio, dtype=np.float32)
        if self._kind == "resemblyzer":
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(a, source_sr=self.sr)
            return self.encoder.embed_utterance(wav)
        if self._kind == "speechbrain":
            import torch
            with torch.no_grad():
                emb = self.encoder.encode_batch(torch.tensor(a).unsqueeze(0))
            return emb.squeeze().cpu().numpy()
        return None

    # ── 호출부호 힌트로 화자 이름 뽑기 (선택) ──
    @staticmethod
    def _name_from_text(text):
        if not text:
            return None
        import re
        m = re.search(r'여기는\s*([가-힣0-9]{2,10}(?:호|VTS|해경|도선|관제))', text)
        return m.group(1) if m else None

    # ── 핵심: 발화 오디오 → 화자 라벨 ──
    def identify(self, audio, hint_text=None, fallback="A"):
        """오디오로 화자 판별. 모델 없으면 fallback(수동 라벨) 반환."""
        if not self.enabled:
            return fallback
        try:
            emb = self._embed(audio)
        except Exception as e:
            print("[diarize] 임베딩 실패, 수동 라벨 사용:", e)
            return fallback
        # 기존 화자와 비교
        best, best_sim = None, -1.0
        for sp in self.speakers:
            s = _cos(emb, sp["centroid"])
            if s > best_sim:
                best, best_sim = sp, s
        if best is not None and best_sim >= self.threshold:
            # 같은 화자 → centroid 갱신(이동평균)
            best["centroid"] = (best["centroid"] * best["n"] + emb) / (best["n"] + 1)
            best["n"] += 1
            label = best["label"]
        else:
            # 새 화자 등록
            label = f"화자{len(self.speakers) + 1}"
            self.speakers.append({"label": label, "centroid": emb, "n": 1})
        # 호출부호 힌트가 있으면 사람이 읽기 쉬운 이름으로 라벨 보정
        if self.name_hints:
            nm = self._name_from_text(hint_text)
            if nm:
                for sp in self.speakers:
                    if sp["label"] == label:
                        sp["label"] = nm
                label = nm
        return label


if __name__ == "__main__":
    reg = SpeakerRegistry()
    print("화자분리 사용 가능:", reg.enabled)
    # 더미 테스트 (임베딩 모델 있을 때만 의미)
    import numpy as np
    a = np.random.randn(16000).astype(np.float32) * 0.1
    print("판별 결과:", reg.identify(a, hint_text="여기는 부산 해경 상황실", fallback="A"))
