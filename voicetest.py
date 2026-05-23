# pip install sounddevice numpy
# pip install faster-whisper
# pip install webrtcvad
# pip install keyboard
# =================================
# vscode에서 가상환경을 만들고 
# vscode 터미널에 위의 패키지를 다운로드를 받는다.
"""
webrtcvad는 사람의 음성을 자동으로 인식을 함.
keyboard는 실제 무전기에서 버튼을 실행해보려고 다운
sounddevice는 노트북에 있는 음성 마이크를 가져오기 위해서 다운
faster-whisper를 쓰는 이유는 기존 whisper보다 가볍고 빠르기 때문에 씀
-> 다만 데이터 학습 시킬 때 어려울수도 있음


"""
import queue
import logging
import sounddevice as sd
import numpy as np
# import webrtcvad
from faster_whisper import WhisperModel
import keyboard

# =========================
# 로그 설정
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# faster-whisper 내부 로그 숨김
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# =========================
# 설정
# =========================

MODEL_SIZE = "base"

SAMPLE_RATE = 16000
CHANNELS = 1

BLOCK_DURATION = 1
VAD_MODE = 2

# =========================
# VAD 초기화
# =========================

# vad = webrtcvad.Vad(VAD_MODE)

# =========================
# Whisper 모델 로드
# =========================

logger.info("Whisper 모델 로딩 중...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",       # GPU 있으면 "cuda"
    compute_type="int8"
)

logger.info("Whisper 모델 로딩 완료")

# =========================
# 오디오 큐
# =========================

audio_queue = queue.Queue()

# =========================
# 마이크 콜백
# =========================
# sounddevice는 4개의 인지라를 갖기 때문에 4개를 써줘야 함.

def audio_callback(indata, frames, time_info, status):

    if status:
        logger.warning(status)

    audio_queue.put(indata.copy())

# =========================
# 스트림 생성
# =========================

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    callback=audio_callback
)

# =========================
# 버퍼
# =========================

buffer = np.empty((0, 1), dtype=np.float32)
last_text = ""

logger.info("실시간 음성인식 시작")
logger.info("Ctrl+C 로 종료")

# =========================
# 실행
# =========================

try:

    stream.start()

    while True:

        try:
            data = audio_queue.get()

        except queue.Empty:
            continue

        # =========================
        # Push-To-Talk
        # =========================

        if keyboard.is_pressed("space"):

            # 스페이스 누르는 동안 버퍼 저장
            buffer = np.concatenate((buffer, data), axis=0)

            print("녹음 중...", end="\r")

        else:

            # 스페이스 뗐고 버퍼에 데이터 있으면
            if len(buffer) > 0:

                print("\n음성 처리 중...")

                audio_data = buffer.flatten()

                # =========================
                # float32 -> int16
                # =========================

                int16_audio = (
                    audio_data * 32767
                ).astype(np.int16)

                # =========================
                # Whisper 실행
                # =========================

                segments, info = model.transcribe(
                        audio_data,
                        language="ko"
                    )

                texts = []

                for segment in segments:

                        text = segment.text.strip()

                        if text and text != last_text:

                            last_text = text

                            texts.append(last_text)

                if texts:

                        print("\n====================")
                        print("인식 결과")
                        print("====================")

                        for text in texts:

                            print(text)

                            logger.info(text)

                # =========================
                # 버퍼 초기화
                # =========================

                buffer = np.empty(
                    (0, 1),
                    dtype=np.float32
                )
# =========================
# Ctrl+C 종료
# =========================

except KeyboardInterrupt:

    logger.info("Ctrl+C 종료")

    print("\n프로그램 종료")

# =========================
# 종료 처리
# =========================

finally:

    logger.info("오디오 스트림 종료")

    stream.stop()

    stream.close()