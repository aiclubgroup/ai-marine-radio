#!/usr/bin/env bash
# =============================================================================
# run_experiment_matrix.sh — 실험 매트릭스 자동 실행 스크립트
#
# 실험 설계 (연구 실행 계획서 5장 참고):
#   모델 3종 (tiny / base / small)
#     × 노이즈 조건 5단계 (clean / 20dB / 10dB / 5dB / 0dB)
#     × 노이즈 제거 OFF / ON  (ON은 노이즈 제거 처리된 폴더를 별도 준비)
#   = 총 30개 조건 → results_matrix.csv 한 파일에 누적 저장
#
# 사전 준비:
#   1) make_noisy_testset.py 로 data/noisy/{clean,snr20,snr10,snr5,snr0} 생성
#   2) (노이즈 제거 ON 실험 시) DeepFilterNet 등으로 처리한 폴더를
#      data/denoised/{clean,snr20,...} 에 준비
#   3) refs.csv (filename,text) 준비
# =============================================================================
set -e

REFS="refs.csv"
OUT="results_matrix.csv"
MODELS=("tiny" "base" "small")
CONDS=("clean" "snr20" "snr10" "snr5" "snr0")

echo "== 노이즈 제거 OFF (원본 노이즈 음성) =="
for MODEL in "${MODELS[@]}"; do
  for COND in "${CONDS[@]}"; do
    python evaluate_stt.py \
      --audio-dir "data/noisy/${COND}" \
      --refs "${REFS}" \
      --model "${MODEL}" --compute-type int8 \
      --condition "${COND}_denoiseOFF_${MODEL}" \
      --out "${OUT}"
  done
done

# 노이즈 제거 ON — data/denoised 폴더가 있을 때만 실행
if [ -d "data/denoised" ]; then
  echo "== 노이즈 제거 ON (전처리 적용 음성) =="
  for MODEL in "${MODELS[@]}"; do
    for COND in "${CONDS[@]}"; do
      python evaluate_stt.py \
        --audio-dir "data/denoised/${COND}" \
        --refs "${REFS}" \
        --model "${MODEL}" --compute-type int8 \
        --condition "${COND}_denoiseON_${MODEL}" \
        --out "${OUT}"
    done
  done
else
  echo "[안내] data/denoised 폴더가 없어 노이즈 제거 ON 실험은 건너뜁니다."
  echo "  DeepFilterNet 예: deepFilter data/noisy/snr10/*.wav -o data/denoised/snr10/"
fi

echo ""
echo "완료! ${OUT} 를 엑셀로 열어 SNR-CER 곡선을 그리세요."
echo "(x축: SNR, y축: CER, 모델×전처리별 선 — 보고서 표준 제시 방식)"
