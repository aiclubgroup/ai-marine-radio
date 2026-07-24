# ⚓ 해상 무전 음성 인식 시스템 v2.0
> 2026 스마트해운물류 × ICT 멘토링 프로젝트  
> 온디바이스 AI 기반 해상 무전 음성 신호 실시간 문자화·화자 분리 시스템 — **UI + AI 엔진 통합판**

---

## 🏗 시스템 구조 (v2.0)

```
┌─────────────────────┐   WebSocket    ┌──────────────────────────────┐
│  UI (둘 중 선택)      │ ────────────▶ │  STT 엔진 (engine/)           │
│  · PySide6 연동판    │  ptt down/up  │  · faster-whisper (small)     │
│  · 웹 UI (내장)      │ ◀──────────── │  · 해상 도메인 프롬프트          │
│                     │  utterance/   │  · user_dict + 후처리 보정      │
│                     │  status       │  · 번역 · 긴급키워드 · 로그      │
└─────────────────────┘               └──────────────────────────────┘
```

- **UI와 엔진 분리**: 마이크 녹음·음성인식은 전부 엔진이 담당, UI는 PTT 신호를 보내고 결과(JSON)를 받아 그리기만 함
- **모델 업데이트 = 파일 교체**: 파인튜닝 후에도 엔진의 가중치 파일만 갈아끼우면 됨 (UI 코드 무변경)
- 실제 무전기 PTT(GPIO)가 생기면 화면 버튼 대신 같은 WS 메시지를 보내면 됨

### WS 메시지 규격 (서울-부산 API)

```
UI → 엔진:  {"type":"ptt","state":"down"|"up","speaker":"A"}
엔진 → UI:  {"type":"status","state":"loading"|"ready"|"processing", ...}
            {"type":"utterance","time":"10:20:21","speaker":"A","lang":"ko",
             "text":"원문","translation":"번역","danger":["mayday"],"proc_sec":1.2}
```

---

## 💻 실행 방법

### 공통 — 의존성 설치

```bash
pip install -r requirements.txt            # UI 쪽
pip install -r engine/requirements.txt     # 엔진 쪽 (faster-whisper 등)
```
> Linux/Jetson에서는 뒤에 `--break-system-packages` 추가  
> 최초 실행 시 Whisper 모델 자동 다운로드 (~500MB, 1회만) → 이후 완전 오프라인

### ① 통합 데모 (PySide6 UI + 실엔진) — 시연 주력

```bash
./run_live_demo.sh                 # macOS/Linux: 엔진과 UI를 한 번에 실행
```
```bat
run_live_demo.bat                  :: Windows: 더블클릭 (엔진 창 + UI 창)
```
수동으로 하려면 터미널 2개:
```bash
python3 engine/realtime_stt_web.py --model small --translate --no-browser   # 터미널 1
python3 seatalk_ai_vhf_connected.py                                         # 터미널 2
```
다른 기기의 엔진에 붙을 때는 엔진을 `--host 0.0.0.0`으로 띄우고
UI를 `python3 seatalk_ai_vhf_connected.py --url ws://<엔진IP>:8765/ws`로 실행.

### ② 웹 UI판 (브라우저만 있으면 동작)

```bash
python3 engine/realtime_stt_web.py --model small --translate
# → http://localhost:8765 자동 오픈 (같은 SeaTalk 디자인)
```

### ③ UI 디자인만 보기 (엔진 없이, 더미 데이터)

```bash
python3 seatalk_ai_vhf.py
```

### ④ 참고용 단독 STT (tkinter, 초기 MVP)

```bash
python3 main.py
```

---

## 📁 폴더 구조

```
ai-marine-radio/
├── engine/                        # STT 엔진
│   ├── realtime_stt_gui.py        #   STTBackend (도메인 프롬프트·보정·번역)
│   ├── realtime_stt_web.py        #   WS 서버 + 웹 UI 호스팅
│   ├── seatalk_ui.html            #   웹 UI (피그마 SeaTalk 디자인)
│   ├── user_dict.txt              #   해상 용어 사전
│   └── requirements.txt
├── seatalk_ai_vhf.py              # PySide6 UI 프로토타입 (더미)
├── seatalk_ai_vhf_connected.py    # PySide6 UI 실엔진 연동판 ★
├── run_live_demo.sh               # 통합 데모 실행 스크립트
├── main.py                        # 초기 MVP (tkinter, 참고용)
└── voicetest.py                   # PTT 테스트 스크립트
```

---

## ⚙️ Jetson (Orin Nano 8GB) 메모

- 장비는 **Jetson Orin Nano 8GB** (구형 Jetson Nano 아님)
- pip으로 설치한 faster-whisper(ctranslate2)는 Jetson에서 **CPU 모드만 동작** — `device="cuda"`는 소스 빌드 필요
  → CPU int8로도 small이 실시간 이내(RTF<1). GPU 가속이 필요하면 whisper.cpp CUDA 빌드 경로 사용
- 마이크: `sudo apt install portaudio19-dev` 필요할 수 있음
- 터치스크린 1024×600 기준으로 UI 고정 해상도 설계됨

---

## 🚨 긴급 키워드

엔진의 `user_dict.txt` + 위험 키워드 로직(`engine/realtime_stt_gui.py`의 `check_danger`)이 담당.
감지 시: UI에 빨간 비상 오버레이(실제 감지 문장·키워드 표시) + 로그에 표시.

---

## 📋 로그

엔진이 CSV로 자동 저장 (`engine/log_*.csv`): time, speaker, lang, text, translation, danger, wav
발화별 원본 녹음(wav)도 함께 저장되어 실해상 데이터 수집을 겸함.

---

## 🗺 개발 로드맵

### 1단계 ✅ (완료)
- 한국어/영어 STT + 해상 도메인 프롬프트 + 용어 보정
- SeaTalk UI (웹 + PySide6) & 실엔진 연동
- 긴급 키워드 감지·비상 오버레이, 교신 로그·녹음 저장

### 2단계 🔲 (예정)
- 실해상 노이즈 검증 및 파인튜닝 (모델 파일 교체 배포)
- PTT GPIO 연동, 수신(RX) 오디오 라인 입력
- 화자 분리 고도화 / 번역 품질 개선

### 3단계 🔲 (예정)
- AI Agent 긴급 상황 분석
- Jetson 최적화 (whisper.cpp CUDA / TensorRT)
- 관제 시스템 연동

---

## 👥 팀 정보

| 구분 | 이름 | 역할 |
|------|------|------|
| 멘토 | 이경용 | 멘토링 |
| 팀장 | 강지선 | 프로젝트 총괄 / AI 시스템 개발 |
| 팀원 | 정치훈 | H/W 설계 / 임베디드 |
| 팀원 | 정수빈 | UI/UX / 응용 소프트웨어 |
| 팀원 | 정혜민 | 시스템 연동 / 시제품 제작 |

---

## 📄 라이선스
본 프로젝트는 2026 스마트해운물류 × ICT 멘토링 프로그램 결과물입니다.
