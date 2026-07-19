#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_stt.py — Whisper STT 성능 평가 스크립트 (CER/WER + 키워드 정확도 + RTF)

멘토링 후속 과제 대응:
  - "자체 테스트셋으로 모델 성능을 평가하고 목표 성능 기준을 세워라"
  - 모델 크기별 × 노이즈 조건별 성능 비교표 생성 (중간보고서용)

평가 지표:
  - CER (문자 오류율): 한국어 주지표. OpenAI도 large-v3부터 한국어를 CER로 평가
    (공백·구두점 제거 후 계산 — 한국어 STT 평가 관행)
  - WER (단어 오류율): 영어/혼용 문장 보조 지표
  - 키워드 정확도: MAYDAY, PAN-PAN 등 조난·안전 필수 어휘 인식률 (별도 관리)
  - RTF (Real-Time Factor): 처리시간 ÷ 오디오길이. RTF < 1 이면 실시간보다 빠름

참조 전사 파일 형식 (refs.csv, UTF-8):
  filename,text
  test001.wav,본선은 부산항으로 향하고 있다
  test002.wav,MAYDAY MAYDAY MAYDAY this is fishing vessel Haeundae

사용 예:
  pip install faster-whisper jiwer soundfile
  # (Jetson에서는 사전 빌드 컨테이너/소스 빌드 권장 — 보고서 3장 참고)
  python evaluate_stt.py \
      --audio-dir data/noisy/snr10 \
      --refs refs.csv \
      --model small --compute-type int8 \
      --condition "snr10_denoiseOFF" \
      --out results.csv

여러 조건을 평가한 뒤 results.csv 를 열면 조건별 비교표가 완성된다
(--out 에 같은 파일을 지정하면 행이 계속 추가됨).
"""

import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# 해상 조난·안전 필수 키워드 (SMCP 기반 — 필요시 수정)
DEFAULT_KEYWORDS = [
    "mayday", "pan-pan", "pan pan", "securite", "sécurité",
    "메이데이", "조난", "구조", "침수", "화재", "충돌", "좌초",
]


# ---------------------------------------------------------------- 텍스트 정규화
def normalize_for_cer(text: str) -> str:
    """CER용 정규화: 소문자화, 구두점 제거, '모든 공백 제거'.

    한국어는 띄어쓰기 오류가 의미 전달에 큰 영향이 없어
    공백 제거 후 CER을 계산하는 것이 국내 관행 (nlptutti 등).
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)  # 구두점 → 공백
    text = re.sub(r"\s+", "", text)            # 공백 전부 제거
    return text


def normalize_for_wer(text: str) -> str:
    """WER용 정규화: 소문자화, 구두점 제거, 공백 정리(공백은 유지)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein 편집거리 (jiwer 미설치 시 fallback)."""
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if ref[i - 1] == hyp[j - 1] else 1))
            prev = cur
    return dp[n]


def compute_cer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = normalize_for_cer(ref), normalize_for_cer(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    try:
        import jiwer
        return jiwer.cer(ref_n, hyp_n)
    except ImportError:
        return edit_distance(ref_n, hyp_n) / len(ref_n)


def compute_wer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = normalize_for_wer(ref), normalize_for_wer(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    try:
        import jiwer
        return jiwer.wer(ref_n, hyp_n)
    except ImportError:
        r, h = ref_n.split(), hyp_n.split()
        # 단어 단위 편집거리
        return edit_distance("\x00".join(r), "\x00".join(h)) / max(len(r), 1)


def keyword_hits(ref: str, hyp: str, keywords) -> tuple:
    """참조문에 등장한 키워드 중 가설문에서도 인식된 개수를 센다."""
    ref_l = normalize_for_wer(ref)
    hyp_l = normalize_for_wer(hyp)
    present = [k for k in keywords if k in ref_l]
    hit = sum(1 for k in present if k in hyp_l)
    return hit, len(present)


# ---------------------------------------------------------------- STT 백엔드
def build_transcriber(model_name: str, compute_type: str, device: str, language):
    """faster-whisper 우선, 없으면 openai-whisper 로 fallback."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        def transcribe(path: str) -> str:
            segments, _info = model.transcribe(path, language=language, beam_size=5)
            return " ".join(seg.text for seg in segments)

        return transcribe, f"faster-whisper/{model_name}/{compute_type}"
    except ImportError:
        pass

    import whisper  # openai-whisper
    model = whisper.load_model(model_name)

    def transcribe(path: str) -> str:
        result = model.transcribe(path, language=language)
        return result["text"]

    return transcribe, f"openai-whisper/{model_name}"


def audio_duration_sec(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="Whisper STT 평가 (CER/WER/키워드/RTF)")
    ap.add_argument("--audio-dir", required=True, help="평가할 오디오 폴더")
    ap.add_argument("--refs", required=True, help="참조 전사 CSV (filename,text)")
    ap.add_argument("--model", default="small",
                    help="모델 크기: tiny/base/small/medium/large-v3 (기본 small)")
    ap.add_argument("--compute-type", default="int8",
                    help="faster-whisper 연산 타입: int8/float16/float32 (기본 int8)")
    ap.add_argument("--device", default="auto", help="cuda/cpu/auto")
    ap.add_argument("--language", default=None,
                    help="언어 고정(ko/en). 미지정 시 자동 감지(한영 혼용 권장)")
    ap.add_argument("--condition", default="",
                    help="조건 라벨 (예: snr10_denoiseOFF) — 결과표에 기록됨")
    ap.add_argument("--out", default="results.csv", help="결과 CSV (누적 추가)")
    ap.add_argument("--per-file-out", default=None,
                    help="파일별 상세 결과 CSV (선택)")
    args = ap.parse_args()

    # 참조 전사 로드
    refs = {}
    with open(args.refs, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            refs[row["filename"].strip()] = row["text"].strip()

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS and p.name in refs)
    missing = [p.name for p in audio_dir.iterdir()
               if p.suffix.lower() in AUDIO_EXTS and p.name not in refs]
    if missing:
        print(f"[주의] refs.csv에 없어 건너뜀: {len(missing)}개 (예: {missing[:3]})")
    if not files:
        raise SystemExit("[오류] 평가할 (오디오, 참조전사) 쌍이 없습니다.")

    print(f"모델 로드 중: {args.model} ...")
    transcribe, backend = build_transcriber(
        args.model, args.compute_type, args.device, args.language)
    print(f"백엔드: {backend} | 평가 파일: {len(files)}개 | 조건: {args.condition or '-'}")

    # 워밍업 (첫 추론은 모델 초기화 포함이라 RTF 측정에서 제외)
    transcribe(str(files[0]))

    rows, total_dur, total_time = [], 0.0, 0.0
    cer_list, wer_list, kw_hit, kw_total = [], [], 0, 0

    for i, path in enumerate(files, 1):
        dur = audio_duration_sec(path)
        t0 = time.perf_counter()
        hyp = transcribe(str(path))
        elapsed = time.perf_counter() - t0

        ref = refs[path.name]
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        hit, tot = keyword_hits(ref, hyp, DEFAULT_KEYWORDS)

        cer_list.append(cer)
        wer_list.append(wer)
        kw_hit += hit
        kw_total += tot
        total_dur += dur
        total_time += elapsed
        rows.append({
            "filename": path.name, "ref": ref, "hyp": hyp.strip(),
            "cer": f"{cer:.4f}", "wer": f"{wer:.4f}",
            "audio_sec": f"{dur:.2f}", "proc_sec": f"{elapsed:.2f}",
        })
        print(f"  [{i}/{len(files)}] {path.name}  CER {cer:.3f}  WER {wer:.3f}")

    mean_cer = sum(cer_list) / len(cer_list)
    mean_wer = sum(wer_list) / len(wer_list)
    rtf = total_time / total_dur if total_dur > 0 else float("nan")
    kw_acc = kw_hit / kw_total if kw_total > 0 else float("nan")

    print("\n===== 결과 요약 =====")
    print(f"조건       : {args.condition or '-'}")
    print(f"백엔드     : {backend}")
    print(f"평균 CER   : {mean_cer*100:.2f}%   (한국어 주지표)")
    print(f"평균 WER   : {mean_wer*100:.2f}%")
    print(f"키워드 정확도: {kw_acc*100:.1f}% ({kw_hit}/{kw_total})" if kw_total
          else "키워드 정확도: 해당 키워드 없음")
    print(f"RTF        : {rtf:.3f}  (오디오 {total_dur:.0f}s / 처리 {total_time:.0f}s)")

    # 조건별 요약 누적 저장 → 실험 매트릭스 표 완성용
    out_path = Path(args.out)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["condition", "backend", "model", "n_files",
                        "mean_cer", "mean_wer", "keyword_acc", "rtf"])
        w.writerow([args.condition, backend, args.model, len(files),
                    f"{mean_cer:.4f}", f"{mean_wer:.4f}",
                    f"{kw_acc:.4f}" if kw_total else "", f"{rtf:.4f}"])
    print(f"\n요약 저장(누적): {out_path}")

    if args.per_file_out:
        with open(args.per_file_out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"파일별 상세 저장: {args.per_file_out}")


if __name__ == "__main__":
    main()
