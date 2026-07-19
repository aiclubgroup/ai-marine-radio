# SeaTalk AI — 해상 무전 실시간 문자화 AI 엔진 (서울팀)

> 온디바이스 AI 기반 해상 무전 음성 신호 실시간 문자화·화자 분리 시스템의 **AI 파트**입니다.
> 실시간 STT + 한↔영 번역 + 위험 키워드 감지 + 보정 파이프라인을 포함하며,
> 노트북과 Jetson Orin Nano(브라우저) 어디서든 동일하게 동작하는 구조입니다.

---

## 폴더 구조

```
demo/                       # 실시간 데모 앱 (시연·개발용)
 ├─ realtime_stt_web.py     #   ★ 웹 UI판 (시연 주력) — FastAPI + WebSocket 서버
 ├─ seatalk_ui.html         #   ★ SeaTalk AI 화면 (피그마 디자인 구현, 단일 HTML)
 ├─ realtime_stt_gui.py     #   STT 엔진 본체(STTBackend) + 데스크톱 백업 UI
 ├─ user_dict.txt           #   사용자 사전 (인명·선박명 + 오인식 교정쌍) — 자유롭게 수정
 ├─ setup_demo_mac.sh       #   맥 설치 스크립트
 ├─ setup_demo_win.bat      #   윈도우 설치 스크립트
 └─ requirements.txt        #   파이썬 의존성
scripts/                    # 실험 도구 (성능 평가용)
 ├─ make_noisy_testset.py   #   해상 노이즈 SNR별 합성 (+VHF 협대역 필터)
 ├─ evaluate_stt.py         #   CER/WER/키워드 정확도/RTF 측정
 └─ run_experiment_matrix.sh#   모델×노이즈×전처리 실험 매트릭스 자동 실행
```

> 엔진 로직은 전부 `realtime_stt_gui.py`의 `STTBackend` 클래스에 있고,
> 웹판(`realtime_stt_web.py`)은 이를 import해서 화면만 바꾼 것입니다.

---

## 빠른 시작 (웹 UI판)

```bash
# 1) 가상환경 + 설치
python3 -m venv venv
source venv/bin/activate            # 윈도우: venv\Scripts\activate
pip install -r demo/requirements.txt

# 2) 실행 → 브라우저 자동 오픈 (http://localhost:8765)
cd demo
python realtime_stt_web.py --model small --translate
```

- **첫 실행 시 모델 자동 다운로드** (STT ~500MB, 번역 ~3GB) — 인터넷 좋은 곳에서 미리 한 번 실행해 두세요.
- 화면의 **PTT 버튼을 누른 채 말하고 떼면** 자막+번역이 표시됩니다. "메이데이", "침수" 등 조난어를 말하면 상단에 빨간 경고가 뜹니다.
- 맥에서 마이크 권한 팝업이 뜨면 허용해 주세요.

### 실행 옵션

| 옵션 | 설명 |
|---|---|
| `--model small` | Whisper 크기 (tiny/base/small…). 느린 장비는 base |
| `--translate` | 한↔영 번역 자막 켜기 |
| `--denoise` | 노이즈 제거 전처리 켜기 — **전후 비교 실험용** (연구상 기본 OFF가 정답) |
| `--correct` | 소형 LLM 문맥 교정 (발화당 2~4초 지연, 실험적) |
| `--no-domain-prompt` | 해상 도메인 프롬프트 끄기 (비교 실험용) |
| `--dict 경로` | 사용자 사전 파일 지정 (기본 user_dict.txt) |

---

## AI 엔진 파이프라인

```
음성(PTT) ─→ Whisper STT ─→ 보정 ─→ 번역 ─→ 화면/로그
              │                │        │
              │ +도메인 프롬프트  │ 사전 치환 │ 고유명사 NE 보호
              │ +사용자 사전     │ LLM 교정 │ 조난어(MAYDAY) 번역 제외
              │ +대화 문맥      │ (옵션)   │ ko→en: Opus-MT / en→ko: NLLB
```

설계 근거(요약): 노이즈 제거는 기본 미적용(자체 실험 + 최신 연구에서 오히려 성능 저하 확인),
한국어 평가는 CER 기준, 모델은 노이즈 실험 결과로 small 선정. 상세는 팀 폴더의
`프로젝트_진행정리.docx` 참조.

---

## ★ UI ↔ 엔진 통신 규격 (하드웨어팀 연동용)

웹소켓(`ws://host:8765/ws`)으로 JSON을 주고받습니다. **이 형식이 서울-부산 API 규격 초안입니다.**

**UI/HW → 엔진** (PTT 신호 — 실제 제품에선 무전기 PTT GPIO가 이 메시지를 보내면 됩니다):
```json
{"type": "ptt", "state": "down", "speaker": "A"}   // 누름 = 녹음 시작
{"type": "ptt", "state": "up"}                     // 뗌 = 인식 시작
```

**엔진 → UI** (자막 데이터 — 디스플레이가 이걸 그리면 됩니다):
```json
{
  "type": "utterance",
  "time": "10:20:21",          // 발화 시각
  "speaker": "A",              // 화자 라벨
  "lang": "ko",                // 감지 언어 (ko/en)
  "text": "침수가 발생했습니다",   // 원문 자막
  "translation": "We are flooding",  // 번역 자막 (없으면 "")
  "danger": ["침수"],           // 위험 키워드 (비면 정상) → 경고 UI 트리거
  "proc_sec": 1.2              // 처리 소요(초)
}
```

상태 메시지: `{"type":"status","state":"loading"|"ready"|"processing"}`

---

## 실험 도구 (scripts/)

```bash
# 노이즈 합성: clean 음성 + 해상 노이즈 → SNR 20/10/5/0dB 테스트셋
python make_noisy_testset.py --clean-dir data/clean --noise-dir data/noise_sea --out-dir data/noisy

# 평가: CER/WER/조난 키워드 정확도/RTF 측정 (결과 CSV 누적)
python evaluate_stt.py --audio-dir data/noisy/snr10 --refs refs.csv --model small --condition snr10 --out results.csv
```

현재까지의 예비 실험 결과 (한국어 15문장 + 합성 해상 노이즈, CER%):

| 모델 | Clean | 10dB | 0dB |
|---|---|---|---|
| Whisper small (244M) | **8.4** | 11.3 | **29.9** |
| Moonshine tiny-ko (27M) | 4.3 | **5.9** | 40.0 |
| wav2vec2-ko (315M) | 0.2 | 9.9 | 37.9 |

→ 심한 노이즈는 Whisper small, 조용한 환경은 Moonshine 우세. 실해상 데이터로 본실험 예정.

---

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `No supported WebSocket library` 경고, UI 연결 안 됨 | `pip install websockets` |
| 데스크톱판 화면이 하얗게 나옴 (맥) | 구형 Tk 8.5 문제 — 앱이 자동으로 밝은 테마로 전환함. 다크테마를 원하면 python.org 파이썬으로 venv 재생성 |
| 번역이 안 나옴 | `pip install transformers sentencepiece torch` 후 `--translate` |
| 영어가 중국어로 인식됨 | 최신 코드에서 자동 보정됨 (확신도 기반 재인식) |
| "메이데이"가 "매일"로 인식됨 | 최신 코드의 도메인 프롬프트로 완화됨. 또박또박 발음하면 더 정확 |

## 주의사항

- `rec_*.wav`(녹음 원본)와 `log_*.csv`(교신 로그)는 **음성 데이터라 커밋 금지** (.gitignore에 포함됨)
- 번역 모델 NLLB는 **비상업(CC-BY-NC) 라이선스** — 연구·시연용으로만 사용, 상업화 시 교체 예정
- `user_dict.txt`의 이름·선박명은 팀 상황에 맞게 자유롭게 수정

## 로드맵

Jetson 포팅·RTF/메모리 실측 → 실해상 데이터 본실험(모델 최종 선정) → 화자분리 자동화(임베딩) → AI Agent 위험분석(규칙 엔진 MVP) → SMCP 도메인 파인튜닝
