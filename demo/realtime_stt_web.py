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
  엔진 → UI:  {"type":"status","state":"loading"|"ready"|"processing","model":"...","denoise":false}
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
    "logw": None, "logf": None,
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


def start_recording():
    import sounddevice as sd
    if state["recording"] or not state["ready"]:
        return
    state["recording"] = True
    state["frames"] = []

    def cb(indata, f, t, s):
        state["frames"].append(indata[:, 0].copy())

    state["stream"] = sd.InputStream(samplerate=SR, channels=CH, blocksize=BLOCK, callback=cb)
    state["stream"].start()


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
        broadcast({"type": "utterance", "time": now, "speaker": speaker, "lang": lang,
                   "text": text, "translation": trans, "danger": hits,
                   "proc_sec": round(proc, 2)})
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
    except WebSocketDisconnect:
        state["clients"].discard(ws)
    except Exception:
        state["clients"].discard(ws)


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
    args = ap.parse_args()
    state["args"] = args

    logpath = HERE / f"log_{state['stamp']}.csv"
    state["logf"] = open(logpath, "w", newline="", encoding="utf-8-sig")
    state["logw"] = csv.writer(state["logf"])
    state["logw"].writerow(["time", "speaker", "lang", "text", "translation", "danger", "wav"])
    print(f"[교신 로그] {logpath}")

    threading.Thread(target=load_backend_thread, args=(args,), daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    print(f"[SeaTalk AI] http://localhost:{args.port} 에서 UI가 열립니다.")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
