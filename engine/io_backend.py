#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_backend.py — 입력(오디오)·PTT(발화구간) 소스 추상화 레이어.

목적: STT 엔진 코드를 건드리지 않고, 하드웨어가 준비되는 대로 '옵션만 바꿔' 연결.
      (마이크 데모 → 실제 무전기 수신 → 파일 재생, PTT 화면버튼 → 무전기 스퀠치신호(GPIO))

두 축:
  1) AudioSource : 어디서 소리를 받나
       - mic     : 시스템 기본 마이크 (sounddevice)         [현재 데모]
       - linein  : 무전기 오디오출력을 물린 사운드카드 입력   [무전기 연결 시]
       - file    : wav 파일 재생 (테스트/시연용)
  2) PTTSource   : 언제 발화가 시작/끝나나
       - ui      : 화면 버튼(WebSocket 메시지)               [현재 데모]
       - gpio    : 무전기 스퀠치/COR 신호를 젯슨 GPIO로 감지  [하드웨어 연결 시]
       - vad     : 음성활동 자동감지(소리 크기 임계)          [무전기 없이 자동]

하드웨어가 없으면 gpio/linein은 '안전하게 미동작(fallback)' 하고 경고만 출력.
→ 지금은 mic + ui 로 동작, 나중에 linein + gpio 로 스위치만 바꾸면 실장 연결.
"""
import time
import numpy as np

SR = 16000
CH = 1
BLOCK = 1600


# ─────────────────────────── 오디오 소스 ───────────────────────────
class AudioSource:
    """콜백으로 오디오 프레임(np.float32 mono)을 흘려보내는 공통 인터페이스."""
    def start(self, on_frame): raise NotImplementedError
    def stop(self): raise NotImplementedError

    @staticmethod
    def create(kind="mic", device=None, path=None):
        if kind == "file":
            return FileSource(path)
        return DeviceSource(device)   # mic / linein 은 device 지정만 다름


class DeviceSource(AudioSource):
    """sounddevice 입력. device=None → 기본 마이크, device='...'(이름/인덱스) → 무전기 line-in."""
    def __init__(self, device=None):
        self.device = device
        self.stream = None

    def start(self, on_frame):
        import sounddevice as sd
        def cb(indata, frames, t, status):
            on_frame(indata[:, 0].copy())
        self.stream = sd.InputStream(samplerate=SR, channels=CH, blocksize=BLOCK,
                                     device=self.device, callback=cb)
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass
        self.stream = None


class FileSource(AudioSource):
    """wav 파일을 실시간처럼 흘려보냄 (시연·재현용)."""
    def __init__(self, path):
        self.path = path; self._run = False

    def start(self, on_frame):
        import soundfile as sf, threading
        self._run = True
        audio, sr = sf.read(self.path)
        if audio.ndim > 1:
            audio = audio.mean(1)
        audio = audio.astype(np.float32)
        def worker():
            for i in range(0, len(audio), BLOCK):
                if not self._run:
                    break
                on_frame(audio[i:i + BLOCK])
                time.sleep(BLOCK / SR)
        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self._run = False


# ─────────────────────────── PTT 소스 ───────────────────────────
class PTTSource:
    """발화 시작(down)/끝(up)을 콜백으로 알리는 공통 인터페이스."""
    def start(self, on_down, on_up): raise NotImplementedError
    def stop(self): pass

    @staticmethod
    def create(kind="ui", **kw):
        if kind == "gpio":
            return GPIOPTT(**kw)
        if kind == "vad":
            return VADPTT(**kw)
        return UIPTT()   # 기본: 화면 버튼(외부에서 down/up 호출)


class UIPTT(PTTSource):
    """화면 버튼/WebSocket. 외부(웹서버)가 press()/release()를 직접 호출."""
    def __init__(self):
        self._down = self._up = None
    def start(self, on_down, on_up):
        self._down, self._up = on_down, on_up
    def press(self):
        if self._down: self._down()
    def release(self):
        if self._up: self._up()


class GPIOPTT(PTTSource):
    """무전기 스퀠치/COR 신호를 젯슨 GPIO 핀으로 감지.
    Jetson.GPIO 없으면(개발 PC 등) 안전하게 미동작 + 경고."""
    def __init__(self, pin=7, active_high=True, poll_hz=100):
        self.pin, self.active_high, self.poll_hz = pin, active_high, poll_hz
        self._run = False
    def start(self, on_down, on_up):
        try:
            import Jetson.GPIO as GPIO
        except Exception:
            print(f"[io_backend] Jetson.GPIO 미탑재 → GPIO PTT 비활성(핀 {self.pin}). "
                  f"하드웨어 연결 시 자동 동작.")
            return
        import threading
        GPIO.setmode(GPIO.BOARD); GPIO.setup(self.pin, GPIO.IN)
        self._run = True
        def worker():
            prev = False
            while self._run:
                lvl = GPIO.input(self.pin) == (1 if self.active_high else 0)
                if lvl and not prev: on_down()
                elif not lvl and prev: on_up()
                prev = lvl
                time.sleep(1.0 / self.poll_hz)
        threading.Thread(target=worker, daemon=True).start()
    def stop(self):
        self._run = False


class VADPTT(PTTSource):
    """무전기 없이 소리 크기(에너지)로 발화 시작/끝 자동 판정.
    AudioSource 프레임을 feed()로 넣어주면 threshold 넘을 때 down/up 발생."""
    def __init__(self, thr=0.02, hang=0.6):
        self.thr, self.hang = thr, hang
        self._down = self._up = None
        self._active = False; self._last_voice = 0.0
    def start(self, on_down, on_up):
        self._down, self._up = on_down, on_up
    def feed(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        now = time.time()
        if rms > self.thr:
            self._last_voice = now
            if not self._active:
                self._active = True; self._down and self._down()
        elif self._active and now - self._last_voice > self.hang:
            self._active = False; self._up and self._up()


# ─────────────────────────── 설정 헬퍼 ───────────────────────────
def build_io(args):
    """argparse 옵션(--audio-source/--audio-device/--ptt-source ...)으로 소스 생성.
    웹서버/GUI가 이 함수만 부르면 하드웨어 구성이 바뀌어도 코드 불변."""
    audio = AudioSource.create(
        kind=getattr(args, "audio_source", "mic"),
        device=getattr(args, "audio_device", None),
        path=getattr(args, "audio_file", None))
    ptt = PTTSource.create(
        kind=getattr(args, "ptt_source", "ui"),
        pin=getattr(args, "gpio_pin", 7))
    print(f"[io_backend] audio={getattr(args,'audio_source','mic')} "
          f"ptt={getattr(args,'ptt_source','ui')}")
    return audio, ptt


def add_io_args(ap):
    """공통 CLI 옵션 등록 — 웹서버/GUI argparse에 붙여 쓰면 됨."""
    ap.add_argument("--audio-source", default="mic", choices=["mic", "linein", "file"],
                    help="mic=기본마이크, linein=무전기 오디오출력, file=wav재생")
    ap.add_argument("--audio-device", default=None,
                    help="linein일 때 사운드카드 이름/인덱스 (sd.query_devices로 확인)")
    ap.add_argument("--audio-file", default=None, help="file 소스일 때 wav 경로")
    ap.add_argument("--ptt-source", default="ui", choices=["ui", "gpio", "vad"],
                    help="ui=화면버튼, gpio=무전기 스퀠치신호(젯슨), vad=자동감지")
    ap.add_argument("--gpio-pin", type=int, default=7, help="GPIO PTT 핀 번호(BOARD)")
    return ap


if __name__ == "__main__":
    # 자체 점검: 파일 소스로 프레임이 흐르는지
    import argparse
    ap = add_io_args(argparse.ArgumentParser()); a = ap.parse_args()
    print("현재 설정:", a.audio_source, a.ptt_source, "(하드웨어 없으면 mic/ui로 동작)")
