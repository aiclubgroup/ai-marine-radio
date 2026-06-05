# ⚓ 해상 무전 음성 인식 시스템 v1.0
> 2026 스마트해운물류 × ICT 멘토링 프로젝트  
> 온디바이스 AI 기반 해상 무전 음성 신호 실시간 문자화·화자 분리 시스템 — **1차 MVP**

---

## 📸 실행 화면

![실행화면](screenshot.png)

---

## ✅ 1차 MVP 구현 기능

| 기능 | 상태 | 설명 |
|------|------|------|
| 실시간 음성 입력 | ✅ | 마이크 자동 감지, 입력 장치 선택 가능 |
| 묵음 구간 자동 제거 | ✅ | faster-whisper 내장 VAD |
| 한국어 / 영어 STT | ✅ | Whisper small 모델 (신뢰도 표시) |
| 언어 자동 감지 | ✅ | 자동 / 한국어 / English 수동 선택 |
| 실시간 자막 출력 | ✅ | 16pt 대형 자막, 언어·신뢰도 표시 |
| 긴급 키워드 감지 | ✅ | mayday, SOS, 조난, 화재 등 27개 |
| 교신 로그 저장 | ✅ | TXT + CSV 자동 저장 (logs/ 폴더) |
| 오프라인 동작 | ✅ | 모델 1회 다운로드 후 인터넷 불필요 |

---

## 💻 실행 방법 (Windows)

### 1단계 — Python 설치 확인
```
python --version   # 3.10 이상 필요
```
없으면 https://python.org 에서 설치

### 2단계 — 의존성 설치
```cmd
pip install faster-whisper sounddevice numpy scipy
```

### 3단계 — 실행
```cmd
python main.py
```
> 최초 실행 시 Whisper 모델 자동 다운로드 (~500MB, 1회만)  
> 이후 완전 오프라인 동작

---

## 💻 실행 방법 (Linux / Jetson Nano)

```bash
pip install faster-whisper sounddevice numpy scipy --break-system-packages
python3 main.py
```

---

## 📁 폴더 구조

```
maritime-stt/
├── main.py              # 메인 프로그램 (tkinter GUI + STT)
├── requirements.txt     # 의존성 목록
├── README.md            # 이 파일
├── logs/                # 교신 로그 자동 저장
│   ├── log_YYYYMMDD_HHMMSS.txt
│   └── log_YYYYMMDD_HHMMSS.csv
└── models/              # Whisper 모델 캐시 (자동 생성)
```

---

## 📋 로그 형식

### TXT (`logs/log_*.txt`)
```
16:29:45   [교신] [KO] 하나 둘 하나 둘
16:30:12 🚨 [긴급] [EN] mayday mayday engine failure
```

### CSV (`logs/log_*.csv`)
```csv
timestamp,language,confidence,emergency,text
16:29:45,ko,1.00,N,하나 둘 하나 둘
16:30:12,en,0.99,Y,mayday mayday engine failure
```

---

## 🚨 긴급 키워드 목록

| 한국어 | English |
|--------|---------|
| 메이데이, 조난, 긴급, 구조, 화재, 침수, 충돌, SOS, 위험, 사고, 응급, 탈출, 비상 | mayday, SOS, emergency, fire, collision, sinking, man overboard, distress, rescue, abandon, danger, engine failure, help, urgent |

긴급 키워드 감지 시 → 빨간 배너 팝업 + 로그에 🚨 표시

---

## 🗺 개발 로드맵

### 1단계 ✅ (완료)
- 노이즈 제거 (VAD)
- 한국어/영어 STT
- 실시간 자막 출력
- 교신 로그 저장

### 2단계 🔲 (예정)
- 화자 분리 (pyannote.audio)
- 자동 번역 (MarianMT / NLLB-200)
- 해상 용어 특화 튜닝

### 3단계 🔲 (예정)
- AI Agent 긴급 상황 분석
- 다중 채널 무전 분석
- Jetson Nano 최적화 배포
- 관제 시스템 연동

---

## ⚙️ Jetson Nano 최적화

`main.py` STTEngine 부분 수정:
```python
# GPU 사용 (VRAM 여유 시)
WhisperModel("small", device="cuda", compute_type="int8")

# CPU 절약 모드
WhisperModel("base", device="cpu", compute_type="int8")
```

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
