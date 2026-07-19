#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_noisy_testset.py — 해상 노이즈 합성 테스트셋 생성 스크립트

깨끗한 음성(clean) 폴더와 노이즈(파도/바람/선박 엔진) 폴더를 입력으로 받아,
SNR 단계별(기본: 20/10/5/0 dB) 노이즈 합성 오디오를 생성한다.
옵션으로 VHF 무전 채널을 근사하는 협대역 밴드패스 필터(300–3000Hz)도 적용 가능.

멘토링 후속 과제 대응:
  - "노이즈가 있는 음성 vs 노이즈 제거 후 음성" 비교 실험용 데이터 생성
  - SNR 단계별 성능 곡선(SNR–CER curve) 실험 재료 준비

사용 예:
  pip install soundfile scipy numpy
  python make_noisy_testset.py \
      --clean-dir data/clean \
      --noise-dir data/noise_sea \
      --out-dir data/noisy \
      --snr-list 20 10 5 0 \
      --bandpass            # (선택) VHF 협대역 시뮬레이션

출력 구조:
  data/noisy/snr20/xxx.wav
  data/noisy/snr10/xxx.wav
  ...
  data/noisy/manifest.csv   (원본, 노이즈 파일, SNR 기록 — 재현성 확보용)
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TARGET_SR = 16000  # Whisper 입력 표준 샘플레이트


def load_audio_mono16k(path: Path) -> np.ndarray:
    """오디오를 로드해 mono / 16kHz / float32 로 통일한다."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # mono
    if sr != TARGET_SR:
        # 정수비 리샘플링 (예: 48000 -> 16000)
        from math import gcd
        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    return data.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: random.Random) -> np.ndarray:
    """clean 신호에 noise를 목표 SNR(dB)로 섞는다.

    노이즈가 짧으면 반복(loop), 길면 임의 구간을 잘라 사용한다.
    SNR = 20*log10(rms_clean / rms_noise_scaled)
    """
    n_len, c_len = len(noise), len(clean)
    if n_len < c_len:
        reps = int(np.ceil(c_len / n_len))
        noise = np.tile(noise, reps)[:c_len]
    else:
        start = rng.randint(0, n_len - c_len)
        noise = noise[start:start + c_len]

    clean_rms, noise_rms = rms(clean), rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-12))
    mixed = clean + noise_scaled

    # 클리핑 방지 정규화 (피크가 1.0을 넘으면 전체 스케일 다운)
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def vhf_bandpass(x: np.ndarray, sr: int = TARGET_SR,
                 low: float = 300.0, high: float = 3000.0) -> np.ndarray:
    """VHF 무전 채널 근사: 300–3000Hz 협대역 밴드패스 필터."""
    sos = butter(6, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="해상 노이즈 합성 테스트셋 생성")
    ap.add_argument("--clean-dir", required=True, help="깨끗한 음성 폴더")
    ap.add_argument("--noise-dir", required=True, help="노이즈(파도/바람/엔진) 폴더")
    ap.add_argument("--out-dir", required=True, help="출력 폴더")
    ap.add_argument("--snr-list", type=float, nargs="+", default=[20, 10, 5, 0],
                    help="SNR(dB) 목록 (기본: 20 10 5 0)")
    ap.add_argument("--bandpass", action="store_true",
                    help="VHF 협대역(300-3000Hz) 필터 적용")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드(재현성)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clean_dir, noise_dir, out_dir = map(Path, (args.clean_dir, args.noise_dir, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(p for p in clean_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    noise_files = sorted(p for p in noise_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not clean_files:
        raise SystemExit(f"[오류] clean 오디오가 없습니다: {clean_dir}")
    if not noise_files:
        raise SystemExit(f"[오류] noise 오디오가 없습니다: {noise_dir}")

    print(f"clean {len(clean_files)}개 × SNR {args.snr_list} → 합성 시작")
    noises = {p: load_audio_mono16k(p) for p in noise_files}

    manifest_rows = []
    for snr in args.snr_list:
        snr_dir = out_dir / f"snr{int(snr)}"
        snr_dir.mkdir(parents=True, exist_ok=True)
        for cf in clean_files:
            clean = load_audio_mono16k(cf)
            nf = rng.choice(noise_files)
            mixed = mix_at_snr(clean, noises[nf], snr, rng)
            if args.bandpass:
                mixed = vhf_bandpass(mixed)
            out_path = snr_dir / (cf.stem + ".wav")
            sf.write(str(out_path), mixed, TARGET_SR)
            manifest_rows.append({
                "output": str(out_path.relative_to(out_dir)),
                "clean_source": str(cf),
                "noise_source": str(nf),
                "snr_db": snr,
                "bandpass": args.bandpass,
            })
        print(f"  SNR {snr:>4.0f} dB: {len(clean_files)}개 완료 → {snr_dir}")

    # clean 원본도 16kHz mono wav로 복사(동일 조건 평가용)
    clean_out = out_dir / "clean"
    clean_out.mkdir(exist_ok=True)
    for cf in clean_files:
        audio = load_audio_mono16k(cf)
        if args.bandpass:
            audio = vhf_bandpass(audio)
        sf.write(str(clean_out / (cf.stem + ".wav")), audio, TARGET_SR)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n완료! manifest 저장: {manifest_path}")
    print("다음 단계: evaluate_stt.py 로 각 SNR 폴더를 평가하세요.")


if __name__ == "__main__":
    main()
