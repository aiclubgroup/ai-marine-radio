#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_stt_web.py — SeaTalk AI 웹 UI 데모 서버 (피그마 디자인 구현판)

기존 STT 엔진(realtime_stt_gui.py의 STTBackend)을 그대로 재사용하고,
UI만 부산팀 피그마 디자인(SeaTalk AI)을 구현한 웹 화면으로 교체한 버전.
젯슨 터치스크린(1024×600)에서도 브라우저만 있으면 그대로 동작하는 구조.

설치 (venv 안에서):
  pip install fastapi uvicorn websockets
  (websockets가 없으면 "No supported WebSocket library" 경고와 함께 UI가 연결되지 않음)

실행:
  python realtime_stt_web.py --model small --translate
  → 자동으로 브라우저가 열림 (안 열리면 http://localhost:8765 접속)

옵션은 GUI판과 동일: --model / --translate / --correct / --denoise / --dict / --langs ...

★ UI ↔ 엔진 WebSocket 메시지 형식 (서울-부산 API 규격 초안) ★
  UI → 엔진:  {"type":"ptt","state":"down"|"up","speaker":"A"}
              {"type":"speaker","value":"B"}
              {"type":"play"}   ← 직전 녹음을 스피커로 재생
  엔진 → UI:  {"type":"status","state":"loading"|"ready"|"processing","model":"...","denoise":false}
              {"type":"history","items":[utterance...]}  ← 접속 시 이번 세션 기록 재전송
              {"type":"utterance","time":"10:20:21","speaker":"A","lang":"ko",
               "text":"원문","translation":"번역","danger":["mayday"],"proc_sec":1.2}
  실제 제품에서 PTT는 화면 버튼 대신 무전기 PTT 신호(GPIO)가 같은 메시지를 보내면 됨.
"""

import argparse
import asyncio
import csv
import json
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

from realtime_stt_gui import STTBackend, load_user_dict, check_danger, SR, CH, BLOCK

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    raise SystemExit("설치 필요: pip install fastapi uvicorn websockets")

try:
    import websockets  # noqa: F401 — uvicorn의 WS 지원 확인용
except ImportError:
    try:
        import wsproto  # noqa: F401
    except ImportError:
        raise SystemExit("웹소켓 라이브러리 필요: pip install websockets "
                         "(없으면 UI가 서버에 연결되지 않습니다)")

HERE = Path(__file__).parent
PORT = 8765

app = FastAPI()
state = {
    "backend": None, "ready": False, "args": None,
    "recording": False, "frames": [], "stream": None, "speaker": "A",
    "clients": set(), "loop": None, "utt": 0, "stamp": time.strftime("%Y%m%d_%H%M%S"),
    "logw": None, "logf": None, "history": [],
    "audio_device": None,        # 마이크 입력 장치
    "audio_out_device": None,    # 스피커 출력 장치
    "last_audio": None,          # 직전 녹음 (재생용)
}


def broadcast(obj):
    """백그라운드 스레드에서도 안전하게 전 클라이언트로 전송."""
    loop = state["loop"]
    if loop is None:
        return
    msg = json.dumps(obj, ensure_ascii=False)
    for ws in list(state["clients"]):
        asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)


def load_backend_thread(args):
    try:
        terms, corrections = load_user_dict(args.dict)
        be = STTBackend(args.model, args.compute_type, args.device, args.translate,
                        user_terms=terms, corrections=corrections,
                        correct_model=args.correct_model if args.correct else None)
        be.use_context = not args.no_context
        if getattr(args, "no_domain_prompt", False):
            be.domain_prompt = None
        langs = None if args.langs.strip().lower() == "all" \
            else tuple(l.strip() for l in args.langs.split(",") if l.strip())
        be.set_lang_policy(langs, args.lang_threshold)
        state["backend"] = be
        state["ready"] = True
        print("[준비 완료] 브라우저에서 PTT를 누르세요.")
        broadcast({"type": "status", "state": "ready", "model": args.model,
                   "denoise": bool(args.denoise)})
    except Exception as e:
        print("[오류] 엔진 로딩 실패:", e)
        broadcast({"type": "status", "state": "loading", "detail": f"로딩 실패: {e}"})


def pick_input_device(prefer=None):
    """녹음에 쓸 입력 장치를 고른다.
    - prefer: 사용자가 지정한 값(정수 인덱스 또는 이름 일부). 있으면 우선.
    - 없으면 pipewire → pulse → default 순으로 이름에 매칭되는 장치를 고른다.
      (Ubuntu 24.04는 pipewire가 오디오를 관리하므로 raw ALSA hw 직접 접근은
       'Device or resource busy'로 실패한다. pipewire가 노출하는 장치를 써야 함.)
    반환: sounddevice device 식별자(정수/문자열) 또는 None(시스템 기본)."""
    import sounddevice as sd
    if prefer not in (None, "", "auto"):
        try:
            return int(prefer)          # "27" 같은 인덱스
        except (ValueError, TypeError):
            return prefer               # "pipewire" 같은 이름 일부(부분 매칭)
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    inputs = [(i, d) for i, d in enumerate(devs) if d.get("max_input_channels", 0) > 0]
    for key in ("pipewire", "pulse", "default"):
        for i, d in inputs:
            if key in d["name"].lower():
                return i
    return None                          # 시스템 기본 장치


def pick_output_device(prefer=None):
    """스피커(재생)에 쓸 출력 장치를 고른다. pick_input_device의 출력 버전.
    pipewire 환경에서 raw ALSA hw 직접 재생이 충돌하는 것을 피하려고
    pipewire→pulse→default 순으로 자동 선택한다."""
    import sounddevice as sd
    if prefer not in (None, "", "auto"):
        try:
            return int(prefer)
        except (ValueError, TypeError):
            return prefer
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    outputs = [(i, d) for i, d in enumerate(devs) if d.get("max_output_channels", 0) > 0]
    for key in ("pipewire", "pulse", "default"):
        for i, d in outputs:
            if key in d["name"].lower():
                return i
    return None


def play_audio(audio, sr=None):
    """오디오(numpy float32)를 선택된 출력 장치로 재생. 실패는 로그로 노출."""
    import sounddevice as sd
    if audio is None or len(audio) == 0:
        print("[재생] 재생할 오디오가 없음")
        return False
    sr = sr or SR
    dev = state.get("audio_out_device")
    try:
        sd.play(np.asarray(audio, dtype=np.float32), sr, device=dev)
        sd.wait()
        return True
    except Exception as e:
        try:
            names = sd.query_devices()
        except Exception:
            names = "(장치 목록 조회 실패)"
        print(f"[재생 오류] 출력 장치 열기 실패 (device={dev!r}): {e}")
        print(f"[힌트] --audio-output-device 로 지정하세요. 가용 장치:\n{names}")
        return False


def start_recording():
    import sounddevice as sd
    if state["recording"] or not state["ready"]:
        return
    state["frames"] = []

    def cb(indata, f, t, s):
        state["frames"].append(indata[:, 0].copy())

    dev = state.get("audio_device")
    try:
        state["stream"] = sd.InputStream(samplerate=SR, channels=CH, blocksize=BLOCK,
                                         device=dev, callback=cb)
        state["stream"].start()
        state["recording"] = True
    except Exception as e:
        # 실패를 조용히 삼키지 않는다: 로그 + UI에 상태 전파(무한 "인식중" 방지)
        state["recording"] = False
        state["stream"] = None
        try:
            names = sd.query_devices()
        except Exception:
            names = "(장치 목록 조회 실패)"
        print(f"[마이크 오류] 입력 장치 열기 실패 (device={dev!r}): {e}")
        print(f"[힌트] --audio-device 로 장치를 지정하세요. 가용 장치:\n{names}")
        broadcast({"type": "status", "state": "ready",
                   "detail": f"마이크 열기 실패: {e} — --audio-device 확인"})


def stop_recording_and_process():
    import soundfile as sf
    if not state["recording"]:
        return
    state["recording"] = False
    try:
        state["stream"].stop(); state["stream"].close()
    except Exception:
        pass
    frames = state["frames"]; state["frames"] = []
    if not frames:
        broadcast({"type": "status", "state": "ready"})
        return
    audio = np.concatenate(frames).astype(np.float32)
    state["last_audio"] = audio          # 재생(모니터)용으로 보관
    if len(audio) < SR * 0.3:
        broadcast({"type": "status", "state": "ready"})
        return
    broadcast({"type": "status", "state": "processing"})
    threading.Thread(target=process_audio, args=(audio, state["speaker"]), daemon=True).start()


def process_audio(audio, speaker):
    import soundfile as sf
    args = state["args"]
    be = state["backend"]
    # 원본 저장 (노이즈 제거 전 상태 — 데이터 수집 겸용)
    state["utt"] += 1
    wavname = str(HERE / f"rec_{state['stamp']}_{state['utt']:03d}_{speaker}.wav")
    try:
        sf.write(wavname, audio, SR)
    except Exception:
        wavname = ""
    # 입력 레벨 안전장치: 무음이면 건너뛰고, 너무 작으면 자동 증폭
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-3:
        print(f"[무음 감지] 입력 신호가 거의 없음 (peak={peak:.5f}) — "
              "맥/젯슨의 입력 장치·입력 음량 설정을 확인하세요")
        broadcast({"type": "status", "state": "ready"})
        return
    if peak < 0.1:
        print(f"[자동 증폭] 입력이 작아 증폭함 (peak {peak:.4f} → 0.70)")
        audio = (audio / peak * 0.7).astype(np.float32)
    # (옵션) 모니터: 방금 녹음한 소리를 스피커로 되들려줌 (입력·출력 동시 확인)
    if getattr(args, "monitor", False):
        print("[모니터] 방금 녹음 재생")
        play_audio(audio, SR)
    # (옵션) 노이즈 제거는 인식용 사본에만
    if args.denoise:
        try:
            import noisereduce as nr
            audio = nr.reduce_noise(y=audio, sr=SR).astype(np.float32)
        except Exception as e:
            print("[경고] 노이즈 제거 실패(원음 사용):", e)
    t0 = time.perf_counter()
    text, lang = be.transcribe(audio)
    trans = be.translate_text(text, lang) if text else ""
    proc = time.perf_counter() - t0
    hits = check_danger(text) if text else []
    now = time.strftime("%H:%M:%S")
    if text:
        if state["logw"]:
            state["logw"].writerow([now, speaker, lang, text, trans, ";".join(hits), wavname])
            state["logf"].flush()
        msg = {"type": "utterance", "time": now, "speaker": speaker, "lang": lang,
               "text": text, "translation": trans, "danger": hits,
               "proc_sec": round(proc, 2)}
        state["history"].append(msg)
        del state["history"][:-200]   # 최근 200건만 유지
        broadcast(msg)
    else:
        broadcast({"type": "status", "state": "ready"})


@app.get("/")
async def index():
    return HTMLResponse((HERE / "seatalk_ui.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state["clients"].add(ws)
    state["loop"] = asyncio.get_running_loop()
    if state["ready"]:
        await ws.send_text(json.dumps({"type": "status", "state": "ready",
                                       "model": state["args"].model,
                                       "denoise": bool(state["args"].denoise)},
                                      ensure_ascii=False))
    else:
        await ws.send_text(json.dumps({"type": "status", "state": "loading",
                                       "detail": "엔진 로딩 중… (첫 실행은 모델 다운로드로 오래 걸릴 수 있음)"},
                                      ensure_ascii=False))
    if state["history"]:
        await ws.send_text(json.dumps({"type": "history", "items": state["history"]},
                                      ensure_ascii=False))
    try:
        while True:
            m = json.loads(await ws.receive_text())
            if m.get("type") == "ptt":
                if m.get("state") == "down":
                    state["speaker"] = m.get("speaker", state["speaker"])
                    start_recording()
                else:
                    stop_recording_and_process()
            elif m.get("type") == "speaker":
                state["speaker"] = m.get("value", "A")
            elif m.get("type") == "play":
                # UI에서 직전 녹음을 스피커로 재생 요청
                threading.Thread(target=play_audio,
                                 args=(state.get("last_audio"), SR), daemon=True).start()
    except WebSocketDisconnect:
        state["clients"].discard(ws)
    except Exception:
        state["clients"].discard(ws)


def run_selftest(seconds=3):
    """UI/STT 없이 오디오 경로만 점검: 장치목록 → N초 녹음 → 되재생.
    입력·출력이 각각 되는지 귀와 로그로 바로 확인하는 용도."""
    import sounddevice as sd
    print("\n===== 오디오 자가진단 =====")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"[장치목록] 조회 실패: {e}")
    idev, odev = state.get("audio_device"), state.get("audio_out_device")
    print(f"\n[입력] device={idev} / [출력] device={odev}")
    print(f"\n{seconds}초간 녹음합니다. 마이크에 대고 말하세요…")
    try:
        rec = sd.rec(int(SR * seconds), samplerate=SR, channels=CH, device=idev, dtype="float32")
        sd.wait()
    except Exception as e:
        print(f"[녹음 실패] {e}\n→ --audio-device 로 입력 장치를 지정해 보세요.")
        return
    rec = rec[:, 0] if rec.ndim > 1 else rec
    peak = float(np.max(np.abs(rec))) if len(rec) else 0.0
    print(f"[녹음 완료] 최대 레벨 peak={peak:.4f} "
          f"({'무음에 가까움 — 입력 음량/장치 확인 필요' if peak < 1e-3 else '신호 정상'})")
    try:
        import soundfile as sf
        out = HERE / "selftest.wav"
        sf.write(str(out), rec, SR)
        print(f"[저장] {out}")
    except Exception as e:
        print(f"[저장 건너뜀] {e}")
    print("\n녹음한 소리를 스피커로 재생합니다…")
    ok = play_audio(rec, SR)
    print("[재생 완료]" if ok else "[재생 실패] → --audio-output-device 로 출력 장치를 지정해 보세요.")
    print("===== 자가진단 끝 =====\n")


def main():
    ap = argparse.ArgumentParser(description="SeaTalk AI 웹 UI 데모 서버")
    ap.add_argument("--model", default="small")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--langs", default="ko,en")
    ap.add_argument("--lang-threshold", type=float, default=0.80)
    ap.add_argument("--dict", default=str(HERE / "user_dict.txt"))
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--correct-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-domain-prompt", action="store_true")
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="다른 기기에서 UI가 붙으면 0.0.0.0")
    ap.add_argument("--audio-device", default="auto",
                    help="녹음 입력 장치(인덱스 또는 이름 일부). 기본 auto=pipewire/pulse/default 자동. "
                         "Ubuntu24.04(pipewire)에서 raw ALSA hw 충돌 시 'pipewire'나 인덱스 지정")
    ap.add_argument("--audio-output-device", default="auto",
                    help="스피커 출력 장치(인덱스 또는 이름 일부). 기본 auto. 재생/모니터에 사용")
    ap.add_argument("--monitor", action="store_true",
                    help="PTT로 녹음할 때마다 그 소리를 스피커로 되들려줌(입력·출력 동시 확인)")
    ap.add_argument("--selftest", action="store_true",
                    help="UI 없이 마이크→스피커 루프만 테스트: 장치목록 출력 + 3초 녹음 + 재생 후 종료")
    args = ap.parse_args()
    state["args"] = args

    # 오디오 장치 결정 (pipewire 우회 문제 대응) — 입력·출력 모두
    import sounddevice as sd
    try:
        state["audio_device"] = pick_input_device(args.audio_device)
        pi = state["audio_device"]
        ni = sd.query_devices(pi)["name"] if pi is not None else "시스템 기본"
        print(f"[마이크] 입력 장치: {ni}  (device={pi})")
    except Exception as e:
        state["audio_device"] = None
        print(f"[마이크] 입력 장치 자동선택 실패({e}) → 시스템 기본. 문제 시 --audio-device 지정")
    try:
        state["audio_out_device"] = pick_output_device(args.audio_output_device)
        po = state["audio_out_device"]
        no = sd.query_devices(po)["name"] if po is not None else "시스템 기본"
        print(f"[스피커] 출력 장치: {no}  (device={po})")
    except Exception as e:
        state["audio_out_device"] = None
        print(f"[스피커] 출력 장치 자동선택 실패({e}) → 시스템 기본. 문제 시 --audio-output-device 지정")

    # 자가진단 모드: UI/STT 없이 마이크→스피커 루프만 확인하고 종료
    if args.selftest:
        run_selftest()
        return

    logpath = HERE / f"log_{state['stamp']}.csv"
    state["logf"] = open(logpath, "w", newline="", encoding="utf-8-sig")
    state["logw"] = csv.writer(state["logf"])
    state["logw"].writerow(["time", "speaker", "lang", "text", "translation", "danger", "wav"])
    print(f"[교신 로그] {logpath}")

    threading.Thread(target=load_backend_thread, args=(args,), daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    print(f"[SeaTalk AI] http://localhost:{args.port} 에서 UI가 열립니다.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
