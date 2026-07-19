# 해상무전기 STT 실험 스크립트

멘토링 후속 과제("자체 테스트셋 평가", "노이즈 제거 전후 비교", "SNR별 성능 곡선")를 수행하기 위한 스크립트 모음입니다. 자세한 배경과 실험 설계는 `연구실행계획서.docx` 5장을 참고하세요.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `make_noisy_testset.py` | 깨끗한 음성 + 해상 노이즈(파도/바람/엔진)를 SNR 단계별로 합성. VHF 협대역 필터 옵션 포함 |
| `evaluate_stt.py` | Whisper로 전사 후 CER/WER/키워드 정확도/RTF 계산. 조건별 결과를 CSV에 누적 |
| `run_experiment_matrix.sh` | 모델 3종 × 노이즈 5조건 × 전처리 ON/OFF = 30개 조건 자동 실행 |

## 빠른 시작 (노트북에서)

```bash
pip install soundfile scipy numpy jiwer faster-whisper

# 1) 폴더 준비
#    data/clean/      : 팀이 녹음한 테스트 문장 (SMCP 영어 + 한국어 해상 교신)
#    data/noise_sea/  : 파도/바람/엔진 노이즈 (ESC-50, BBC, Freesound 등)
#    refs.csv         : filename,text 형식의 정답 전사

# 2) 노이즈 합성 (clean/20/10/5/0 dB 세트 생성)
python make_noisy_testset.py --clean-dir data/clean --noise-dir data/noise_sea \
    --out-dir data/noisy --snr-list 20 10 5 0

# 3) 단일 조건 평가
python evaluate_stt.py --audio-dir data/noisy/snr10 --refs refs.csv \
    --model small --condition snr10_denoiseOFF_small --out results.csv

# 4) 전체 매트릭스 실행
bash run_experiment_matrix.sh
```

## Jetson Orin Nano에서

- pip의 faster-whisper aarch64 휠은 CUDA 라이브러리가 빠져 import 에러가 흔함 → jetson-containers 사전 빌드 Docker 이미지 사용 권장 (계획서 3장 참고)
- 스크립트는 faster-whisper가 없으면 openai-whisper로 자동 fallback
- 메모리 측정: 별도 터미널에서 `sudo tegrastats` 또는 `jtop` 실행 후 피크 RAM 기록

## 주의사항

- RTF 측정 시 첫 추론(워밍업)은 자동으로 제외됨. 공식 결과는 3회 반복 평균 권장
- 한국어 주지표는 **CER**(공백·구두점 제거) — OpenAI도 large-v3부터 한국어를 CER로 평가
- 노이즈 제거(DeepFilterNet 등)는 오히려 성능을 떨어뜨릴 수 있음(아티팩트 문제) → "ON이 항상 좋다"는 가설 없이 양방향으로 측정할 것 (이게 좋은 연구 포인트!)
