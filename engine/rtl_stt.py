#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtl_stt.py — RTL-SDR 무전 수신 → STT 브리지 (하드웨어팀 파이프라인의 STT 연결판).

하드웨어팀이 젯슨에서 검증한 수신 경로(2026-08-14 로그):
    rtl_fm -f 433.575M -M nfm -s 200000 -r 48000 | aplay -D plughw:1,0 ...  (스피커)
실행은 두 방식 — 어느 쪽이든 결과 동일:
  (1) 한 줄 실행 (권장): rtl_fm을 내부에서 직접 띄움
      python3 rtl_stt.py --freq 433.575M --model ~/models/fw-marine
  (2) 파이프: rtl_fm -f 433.575M -M nfm -s 16000 -r 16000 -l 80 | python3 rtl_stt.py --model ~/models/fw-marine
  ⚠ -s는 반드시 -r과 같게(16000) — 200000이면 협대역 음성이 2.5% 진폭으로 소실 (밤샘 실측)
  ⚠ --model 을 빼면 기본 small(파인튜닝 안 된 모델)이 로드됨 — 반드시 fw-marine 경로 지정
  스퀠치(-l/--squelch 80)는 무신호 잡음을 차단해 VAD 분절을 정확하게 함. 0으로 끄면
  잡음이 통째로 들어가 "MAYDAY" 환각이 생김 (2026-08-15 새벽 실측)

동글 인식 확인 (최초 1회):
    lsusb            # Realtek RTL2838 보이면 연결됨
    rtl_test -t      # Found 1 device(s) 나오면 정상 (PLL not locked 메시지는 무시)

USB 권한 (일반 계정으로 --freq 모드 쓰려면 최초 1회):
    sudo tee /etc/udev/rules.d/20-rtlsdr.rules <<< 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"'
    sudo udevadm control --reload-rules && sudo udevadm trigger   # 후 동글 재연결
    (임시 방편: sudo rtl_fm ... -l 80 | python3 rtl_stt.py --model ... 파이프)

동작:
  * stdin으로 raw S16_LE 16kHz 모노 PCM을 받는다 (rtl_fm 출력 포맷 그대로)
  * PTT 신호가 없으므로 VAD(에너지 기반)로 "한 번의 송신 = 한 발화"를 자동 분리
    — 무전은 스퀠치가 닫히면 무음이라 경계가 뚜렷해 에너지 VAD로 충분
  * 발화마다: STT → 화자 라벨(marine_speaker) → 위험분석(marine_danger) → 출력 + CSV 로그

테스트 (무전기 없이, 녹음 파일로):
    ffmpeg -i 녹음.wav -f s16le -ar 16000 -ac 1 - | python3 rtl_stt.py --model small

주의: 해상 VHF(156–162MHz)는 수신 전용 모니터링만 (전파법 — 송신 금지).
"""
import argparse
import csv
import sys
import time

import numpy as np

from realtime_stt_gui import STTBackend, load_user_dict, check_danger
from marine_speaker import SpeakerTracker
from marine_danger import DangerAgent

SR = 16000
BLOCK = 1600            # 100ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="모델 경로(fw-marine 폴더) 또는 이름")
    ap.add_argument("--freq", default=None,
                    help="수신 주파수 — 주면 rtl_fm을 직접 실행 (예: 433.575M, 156.8M). 없으면 stdin 파이프 모드")
    ap.add_argument("--squelch", type=int, default=80,
                    help="rtl_fm 스퀠치 레벨 (-l). 0=끔(잡음 홍수 — 환각 원인이라 비권장). "
                         "-s 16000 스케일 기준 80~150 사이에서 보정 (80=잡음 유입, 200=전부 차단 실측)")
    ap.add_argument("--gain", default=None, help="rtl_fm 튜너 게인 (기본 자동)")
    ap.add_argument("--ui", default=None,
                    help="웹 UI 주입 주소 (예: http://localhost:8765/inject) — 주면 화면에도 표시")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dict", default=None, help="user_dict.txt 경로 (기본: 엔진 폴더)")
    ap.add_argument("--rate", type=int, default=16000, help="rtl_fm -r 값 (16000 아니면 리샘플)")
    ap.add_argument("--vad-db", type=float, default=-38.0, help="발화 판정 임계(dBFS)")
    ap.add_argument("--min-sil", type=float, default=0.6, help="발화 종료 무음 길이(초)")
    ap.add_argument("--min-utt", type=float, default=0.5, help="최소 발화 길이(초)")
    ap.add_argument("--max-utt", type=float, default=25.0, help="최대 발화 길이(초, 강제 절단)")
    args = ap.parse_args()

    from pathlib import Path
    dict_path = args.dict or str(Path(__file__).parent / "user_dict.txt")
    try:
        terms, corr = load_user_dict(dict_path)
    except Exception:
        terms, corr = [], {}

    print(f"[rtl_stt] 모델 로드 중: {args.model}", flush=True)
    be = STTBackend(args.model, args.compute_type, args.device, translate=False,
                    user_terms=terms, corrections=corr)
    tracker = SpeakerTracker()
    agent = DangerAgent()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(f"log_rtl_{stamp}.csv", "w", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    logw.writerow(["time", "speaker", "text", "danger", "level", "position", "persons"])

    print("[rtl_stt] 수신 대기 — rtl_fm 파이프 입력 (Ctrl+C 종료)", flush=True)

    # VAD 상태
    voiced = []          # 현재 발화 프레임
    sil_frames = 0
    min_sil_frames = int(args.min_sil * SR / BLOCK)
    utt_no = 0

    def flush_utt():
        nonlocal voiced, utt_no
        if not voiced:
            return
        audio = np.concatenate(voiced)
        voiced = []
        if args.rate != SR:      # rtl_fm -r 이 다르면 선형 리샘플
            n = int(len(audio) * SR / args.rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                              np.arange(len(audio)), audio).astype(np.float32)
        if len(audio) < args.min_utt * SR:
            return
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            return
        if peak < 0.1:                    # 작은 수신음 자동 증폭
            audio = (audio / peak * 0.7).astype(np.float32)
        utt_no += 1
        t0 = time.perf_counter()
        text, lang = be.transcribe(audio)
        dt = time.perf_counter() - t0
        if not text:
            return
        # 잡음 환각 필터 (ver2_ai RadioBridge와 동일 기준) — 치직음을 억지로
        # 글자화한 결과("MAYDAY.com", "바사사삭")가 위험감지까지 가는 것 차단.
        #  1) no_speech 낮음(<0.3) = 명백한 사람 말 → 신뢰도 무관 통과
        #  2) no_speech 높음(>0.6) = 무음/잡음 → 폐기
        #  3) 그 사이면 신뢰도 -1.4 미만일 때만 폐기 (무전 채널은 원래 낮게 나옴)
        no_sp = getattr(be, "last_no_speech", 0.0)
        avg_lp = getattr(be, "last_avg_logprob", 0.0)
        if no_sp > 0.6 or (no_sp >= 0.3 and avg_lp < -1.4):
            print(f"[잡음 폐기] conf {avg_lp:.2f}, no_speech {no_sp:.2f}: {text[:30]}", flush=True)
            return
        t = "".join(text.split())
        if len(t) >= 6 and (len(set(t)) / len(t) < 0.25 or any(t[:w] * 3 in t for w in (1, 2, 3))):
            print(f"[반복 환각 폐기] {text[:30]}", flush=True)
            return
        label = tracker.assign(text, is_self=False)
        if label == "상대선(미상)":
            label = f"수신{utt_no}"
        hits = check_danger(text)
        rep = agent.analyze(text, speaker=label) if hits else None
        now = time.strftime("%H:%M:%S")
        mark = f"  ⚠{rep['level']}" if rep else ""
        print(f"[{now}] {label}: {text}  ({dt:.1f}s){mark}", flush=True)
        if rep:
            print(f"         └ {rep['summary']}", flush=True)
            if rep["advice"]:
                print(f"         └ 권고: {' / '.join(rep['advice'])}", flush=True)
        logw.writerow([now, label, text, ";".join(hits),
                       rep["level"] if rep else "", rep["position"] if rep else "",
                       rep["persons"] if rep else ""])
        logf.flush()
        if args.ui:                      # 웹 UI에도 표시 (실패해도 수신은 계속)
            try:
                import json as _json
                from urllib import request as _rq
                body = {"speaker": label, "text": text, "danger": hits, "proc_sec": round(dt, 2)}
                if rep:
                    body["danger_level"] = rep["level"]
                    body["danger_advice"] = rep["advice"]
                    body["danger_summary"] = rep["summary"]
                _rq.urlopen(_rq.Request(args.ui, data=_json.dumps(body).encode(),
                                        headers={"Content-Type": "application/json"}), timeout=2)
            except Exception as e:
                print(f"[ui 전송 실패] {e}", flush=True)

    # 입력 소스: --freq 주면 rtl_fm을 직접 실행, 아니면 stdin 파이프
    proc = None
    if args.freq:
        import subprocess, shutil
        if shutil.which("rtl_fm") is None:
            sys.exit("[오류] rtl_fm 없음 — 설치: sudo apt install rtl-sdr  (설치 후 rtl_test -t로 동글 인식 확인)")
        # -s를 -r과 같게: 협대역 변조 복조 감도 확보 (200000이면 음성이 2.5%로 소실 — 실측)
        cmd = ["rtl_fm", "-f", args.freq, "-M", "nfm", "-s", str(args.rate), "-r", str(args.rate)]
        if args.squelch:
            cmd += ["-l", str(args.squelch)]
        if args.gain:
            cmd += ["-g", str(args.gain)]
        print(f"[rtl_stt] 수신 시작: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)   # stderr는 그대로 노출 (장치 오류 진단용)
        buf = proc.stdout
    else:
        buf = sys.stdin.buffer
    bytes_per_block = BLOCK * 2
    try:
        while True:
            data = buf.read(bytes_per_block)
            if not data:
                if proc is not None and proc.poll() is not None:
                    print(f"[오류] rtl_fm이 즉시 종료됨 (코드 {proc.returncode}) — "
                          "대부분 USB 권한 문제. sudo rtl_fm ... | 파이프로 실행하거나 "
                          "udev 규칙 설치 후 동글 재연결 (파일 상단 주석 참조)", flush=True)
                break
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level_db = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
            if level_db > args.vad_db:
                voiced.append(x)
                sil_frames = 0
                if len(voiced) * BLOCK > args.max_utt * SR:
                    flush_utt()
            elif voiced:
                voiced.append(x)          # 발화 내 짧은 무음 포함
                sil_frames += 1
                if sil_frames >= min_sil_frames:
                    flush_utt()
                    sil_frames = 0
    except KeyboardInterrupt:
        pass
    flush_utt()
    if proc is not None:
        proc.terminate()
    logf.close()
    print("[rtl_stt] 종료", flush=True)


if __name__ == "__main__":
    main()
