# 엔진 패치 — 2026-08-14 (화자분리·위험분석2단·JA/ZH·환각수정·파인튜닝 모델)

새 모듈 3개는 `engine/` 폴더에 그대로 복사. 기존 파일 수정은 아래 지점만.

## 0. 새 파일 (engine/에 복사)
- `marine_speaker.py`  — 화자분리 (PTT 역할 + "여기는 ○○호" 자기호출 라벨). 자가테스트: `python marine_speaker.py`
- `marine_danger.py`   — 위험분석 2단 (등급 DISTRESS/URGENCY/SAFETY/WATCH + 대응 권고). 자가테스트: `python marine_danger.py`
- `marine_translate.py`— EN/JA/ZH 번역 + 조난어 보호. (NLLB 첫 로드 시 2.4GB 다운로드)

## 1. 환각 루프 수정 (젯슨에서 같은 말 반복되던 문제)
`realtime_stt_gui.py`의 `model.transcribe(...)` 호출 kwargs에 추가 (두 군데 모두):
```python
kwargs["condition_on_previous_text"] = False   # 직전 출력에 갇히는 루프 차단
kwargs["no_repeat_ngram_size"] = 3             # 반복 n-gram 억제
```
※ 대화 문맥 주입(initial_prompt)은 그대로 유지 — 이건 입력 컨텍스트라 환각 루프와 무관.

## 2. 파인튜닝 모델 교체 (젯슨·맥 공통)
```bash
# 파인튜닝/faster-whisper-small-marine.zip 을 젯슨으로 복사 후
unzip faster-whisper-small-marine.zip -d ~/models/
python realtime_stt_web.py --model ~/models/faster-whisper-small-marine
```
모델 인자에 폴더 경로를 주면 끝 — 코드 수정 없음 ("모델 업데이트 = 파일 교체" 규격).

## 3. 화자분리 연결 (realtime_stt_web.py의 do_transcribe 부근)
```python
from marine_speaker import SpeakerTracker
tracker = SpeakerTracker()          # 상태 전역 1개

# 발화 확정 직후:
label = tracker.assign(text, is_self=(speaker == "본선" or ptt_was_down))
# WS utterance 메시지의 "speaker" 필드에 label 사용 → UI가 그대로 표시
```

## 4. 위험분석 2단 연결 (기존 check_danger 다음 줄)
```python
from marine_danger import DangerAgent
danger_agent = DangerAgent()        # 상태 전역 1개

hits = check_danger(text)           # Stage1 (기존 그대로 — 즉시 경보)
report = danger_agent.analyze(text, speaker=label)   # Stage2
# report가 있으면 WS 메시지에 추가: "danger_level": report["level"],
#   "danger_advice": report["advice"], "danger_summary": report["summary"]
# UI: DISTRESS=빨강 오버레이(기존), URGENCY=주황 배너, SAFETY=파랑 배너 권장
```

## 5. JA/ZH 번역 연결 (STTBackend의 번역 부분)
```python
from marine_translate import MarineTranslator
tr = MarineTranslator(targets=["EN", "JA", "ZH"])
translation = tr.translate(text_ko, ui_selected_lang)   # UI의 EN/JA/ZH 선택값
```
⚠ 젯슨에서는 NLLB를 CTranslate2 int8로 변환해서 쓸 것 (fp32 2.4GB는 예산 초과):
```bash
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
    --output_dir nllb-600m-int8 --quantization int8
```

## 젯슨 배포 순서 (오늘 밤)
1. 파인튜닝 모델 zip + engine/ 폴더(패치 반영) USB 복사
2. 모델 압축 해제 → --model 경로 지정 실행
3. 마이크 입력 확인 (pipewire 자동 장치 선택은 7/25 커밋에 이미 있음)
4. 데모 문장 낭독 → ①환각 루프 사라졌는지 ②속도(발화당 처리초) ③화자 라벨 표시 확인
5. 속도가 여전히 느리면: --model tiny 백업 경로 (small 대비 5배 빠름, CER 손해 감수)
"""


---

# 2차 통합 — 2026-08-15 (친구 모듈 병합, 젯슨 배포판)

기준: push된 최신 engine/ + 친구(A역할) 모듈 이식. 한 폴더 = 젯슨 배포 단위.

## 새로 편입 (친구 작성)
- `diarize.py` — 성문 임베딩 화자 자동분리 (resemblyzer→ECAPA 폴백, 미설치 시 자동 비활성)
- `io_backend.py` — 오디오·PTT 소스 추상화 (mic/linein/file × ui/gpio/vad) — 무전기·RTL-SDR 연결 대비
- `log_search.py` — 교신 로그 CSV 검색 CLI

## 병합 내용
- `marine_danger.py` ← 친구 danger_agent의 **선박명·위치·인원 추출** 이식 (report에 vessel/position/persons 필드 추가)
- `realtime_stt_gui.py` = 친구판 채택 (**_is_repetitive 환각 반복 필터** + **일/중→한 수신 번역**) + DANGER_KW 복원(팬팬·전복·퇴선)
- `realtime_stt_web.py` ← `--diarize` 옵션 연결. 화자 라벨 우선순위: 텍스트 자기호출 > 성문 임베딩 > 수동 A/B

## 젯슨 실행 (통합판)
```bash
pip install faster-whisper fastapi uvicorn websockets sounddevice soundfile numpy --break-system-packages
# (선택) 성문 화자분리: pip install resemblyzer
python3 realtime_stt_web.py --model ~/models/faster-whisper-small-marine            # 웹 UI판
AI_MODEL_DIR=~/models/faster-whisper-small-marine python3 ai_seatalk_ver2_ai.py    # PySide6 UI판
```

---

# 3차 — RTL-SDR 무전 → STT 브리지 (2026-08-15 새벽)

하드웨어팀 젯슨 수신 검증(8/14 Notion 로그: Quansheng→RTL-SDR→rtl_fm→스피커)에 STT를 연결.

- `rtl_stt.py` (신규) — rtl_fm 파이프 입력 → VAD 발화 분절 → 파인튜닝 STT → 화자 라벨·위험분석·CSV 로그.
  A(친구)·B 양쪽 초안을 병합: 엔진 STTBackend(사전·프롬프트) 기반 + 친구 제안 `-l 40` 스퀠치·`--rate` 리샘플 채택.
  모의 rtl_fm 스트림(무전화 실음성)으로 E2E 검증: 메이데이→DISTRESS 경보, 일반 교신 정상 인식.
- `io_backend.py` ← StdinSource 추가 (동일 파이프를 웹 UI 엔진에 연결할 때 사용).

젯슨 실행:
```bash
rtl_fm -f 433.575M -M nfm -s 200000 -r 16000 -l 40 | python3 rtl_stt.py --model ~/models/fw-marine
```
해상 VHF(156-162MHz)는 수신 전용 모니터링만 (전파법).
