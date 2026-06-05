"""
해상 무전 음성 인식 시스템 (Maritime Radio STT System)
온디바이스 AI 기반 | 1024x800 해상도 | 오프라인 동작
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, font
import threading
import queue
import time
import os
import csv
import json
import sys
import numpy as np
from datetime import datetime
import sounddevice as sd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_EMERGENCY = set(["mayday","sos","emergency","fire","collision","sinking",
    "man overboard","distress","rescue","abandon","danger","engine failure",
    "help","urgent","메이데이","조난","긴급","구조","화재","침수","충돌","위험",
    "사고","응급","구급","탈출","비상"])

def check_emergency(text):
    t = text.lower()
    return any(kw in t for kw in ALL_EMERGENCY)

class STTEngine:
    def __init__(self):
        self.model = None
        self.ready = False

    def load(self, progress_cb=None):
        try:
            from faster_whisper import WhisperModel
            if progress_cb: progress_cb("Whisper 모델 로딩 중...")
            self.model = WhisperModel("small", device="cpu", compute_type="int8",
                                      download_root=MODEL_DIR)
            self.ready = True
            if progress_cb: progress_cb("모델 로딩 완료 ✓")
        except Exception as e:
            if progress_cb: progress_cb(f"모델 로딩 실패: {e}")

    def transcribe(self, audio_np, language=None):
        if not self.ready: return "", "unknown", 0.0
        try:
            segments, info = self.model.transcribe(
                audio_np.astype("float32"), language=language,
                beam_size=3, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="해상 무전 교신. VHF radio maritime communication.")
            text = " ".join(s.text.strip() for s in segments)
            return text.strip(), info.language, info.language_probability
        except Exception as e:
            return f"[오류: {e}]", "unknown", 0.0

class AudioRecorder:
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 3.0
    SILENCE_THRESH = 0.01

    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.running = False
        self._device_id = None

    def set_device(self, device_id):
        self._device_id = device_id

    def start(self):
        self.running = True
        threading.Thread(target=self._record_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _record_loop(self):
        chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=chunk_samples,
                                device=self._device_id) as stream:
                while self.running:
                    audio_chunk, _ = stream.read(chunk_samples)
                    audio_np = audio_chunk.flatten()
                    if float(np.sqrt(np.mean(audio_np**2))) > self.SILENCE_THRESH:
                        self.audio_queue.put(audio_np.copy())
        except Exception as e:
            self.audio_queue.put(("ERROR", str(e)))

class LogManager:
    def __init__(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_path = os.path.join(LOG_DIR, f"log_{now}.txt")
        self.csv_path = os.path.join(LOG_DIR, f"log_{now}.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["timestamp","language","confidence","emergency","text"])

    def write(self, ts, lang, conf, is_emerg, text):
        tag = "🚨[긴급]" if is_emerg else "  [교신]"
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} {tag} [{lang.upper()}] {text}\n")
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([ts, lang, f"{conf:.2f}", "Y" if is_emerg else "N", text])

class MaritimeSTTApp:
    C = {"bg":"#0a1628","panel":"#0d2040","panel2":"#112244","accent":"#00aaff",
         "accent2":"#0066cc","emergency":"#ff3333","emerg_bg":"#3a0000",
         "text":"#e8f4ff","dim":"#6688aa","green":"#00cc66","yellow":"#ffcc00",
         "border":"#1a3a5c","log_bg":"#050f1e","log_em":"#ff6666",
         "log_n":"#88ccff","hdr":"#061020"}

    def __init__(self, root):
        self.root = root
        self.root.title("해상 무전 음성 인식 시스템 v1.0")
        self.root.geometry("1024x800+100+50")
        self.root.resizable(False, False)
        self.root.configure(bg=self.C["bg"])

        self.fT = font.Font(family="Malgun Gothic", size=14, weight="bold")
        self.fH = font.Font(family="Malgun Gothic", size=10, weight="bold")
        self.fB = font.Font(family="Malgun Gothic", size=11)
        self.fL = font.Font(family="Courier New",   size=10)
        self.fS = font.Font(family="Malgun Gothic", size=9)
        self.fG = font.Font(family="Malgun Gothic", size=13, weight="bold")

        self.audio_queue = queue.Queue()
        self.stt = STTEngine()
        self.recorder = AudioRecorder(self.audio_queue)
        self.log_manager = LogManager()
        self._is_recording = False
        self._total = 0
        self._emerg = 0
        self._lang_override = None

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()
        self._poll()

    def _build_ui(self):
        C = self.C
        hdr = tk.Frame(self.root, bg=C["hdr"], height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⚓  해상 무전 음성 인식 시스템",
                 bg=C["hdr"], fg=C["accent"], font=self.fT).pack(side="left", padx=18)
        self._clk = tk.StringVar()
        tk.Label(hdr, textvariable=self._clk, bg=C["hdr"],
                 fg=C["dim"], font=self.fS).pack(side="right", padx=18)
        self._tick()

        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=8, pady=4)

        # 왼쪽 패널
        left = tk.Frame(main, bg=C["bg"], width=310)
        left.pack(side="left", fill="y", padx=(0,6))
        left.pack_propagate(False)

        # 상태
        b = self._card(left, "◉  시스템 상태")
        self._dot = tk.Label(b, text="●", fg=C["dim"], bg=C["panel"], font=self.fG)
        self._dot.pack(side="left")
        self._sv = tk.StringVar(value="모델 로딩 중...")
        tk.Label(b, textvariable=self._sv, bg=C["panel"],
                 fg=C["text"], font=self.fB).pack(side="left", padx=8)

        # 녹음
        b = self._card(left, "🎙  녹음 제어")
        self._btn = tk.Button(b, text="▶  녹음 시작", font=self.fG,
                              bg=C["accent2"], fg="white", relief="flat",
                              padx=12, pady=8, cursor="hand2",
                              command=self._toggle, state="disabled")
        self._btn.pack(fill="x", pady=(2,6))
        tk.Label(b, text="음성 레벨", bg=C["panel"], fg=C["dim"], font=self.fS).pack(anchor="w")
        vu = tk.Frame(b, bg=C["log_bg"], height=14); vu.pack(fill="x", pady=2)
        vu.pack_propagate(False)
        self._vu = tk.Canvas(vu, bg=C["log_bg"], height=14, highlightthickness=0)
        self._vu.pack(fill="both", expand=True)
        self._level = 0.0

        tk.Label(b, text="인식 언어", bg=C["panel"], fg=C["dim"],
                 font=self.fS).pack(anchor="w", pady=(4,0))
        self._lv = tk.StringVar(value="자동")
        lf = tk.Frame(b, bg=C["panel"]); lf.pack(fill="x")
        for lbl, val in [("자동",None),("한국어","ko"),("English","en")]:
            tk.Radiobutton(lf, text=lbl, variable=self._lv, value=lbl,
                           bg=C["panel"], fg=C["text"], selectcolor=C["panel2"],
                           activebackground=C["panel"], font=self.fS,
                           command=lambda v=val: setattr(self,"_lang_override",v)
                           ).pack(side="left", padx=4)

        # 통계
        b = self._card(left, "📊  실시간 통계")
        self._st = tk.StringVar(value="0")
        self._se = tk.StringVar(value="0")
        self._sl = tk.StringVar(value="-")
        for lbl, var, col in [("총 교신 수",self._st,C["accent"]),
                               ("긴급 교신",self._se,C["emergency"]),
                               ("감지 언어",self._sl,C["yellow"])]:
            tk.Label(b, text=lbl, bg=C["panel"], fg=C["dim"],
                     font=self.fS, anchor="w").pack(fill="x")
            tk.Label(b, textvariable=var, bg=C["panel"], fg=col,
                     font=self.fG, anchor="w").pack(fill="x", pady=(0,5))

        # 장치
        b = self._card(left, "🔊  입력 장치")
        devs = []
        try:
            for i,d in enumerate(sd.query_devices()):
                if d["max_input_channels"]>0: devs.append(f"{i}: {d['name']}")
        except: pass
        self._dv = tk.StringVar()
        cb = ttk.Combobox(b, textvariable=self._dv, values=devs or ["기본"],
                          state="readonly", font=self.fS)
        cb.pack(fill="x")
        if devs: cb.set(devs[0])
        cb.bind("<<ComboboxSelected>>", lambda e: self.recorder.set_device(
            int(self._dv.get().split(":")[0])))

        # 오른쪽
        right = tk.Frame(main, bg=C["bg"]); right.pack(fill="both", expand=True)

        # 긴급 배너
        self._ef = tk.Frame(right, bg=C["emerg_bg"], pady=6, padx=12)
        self._el = tk.Label(self._ef, text="", bg=C["emerg_bg"], fg=C["emergency"],
                            font=self.fG, wraplength=660, justify="left")
        self._el.pack(anchor="w")

        # 자막
        so = tk.Frame(right, bg=C["border"], padx=1, pady=1)
        so.pack(fill="x", pady=(0,6))
        si = tk.Frame(so, bg=C["panel"]); si.pack(fill="both")
        tk.Label(si, text="📺  실시간 자막", bg=C["panel2"], fg=C["accent"],
                 font=self.fH, anchor="w", padx=10, pady=5).pack(fill="x")
        self._sub = tk.StringVar(value="— 대기 중 —")
        tk.Label(si, textvariable=self._sub, bg=C["panel"], fg=C["text"],
                 font=font.Font(family="Malgun Gothic",size=16,weight="bold"),
                 wraplength=660, justify="left", pady=14, padx=12).pack(fill="x")
        mf = tk.Frame(si, bg=C["panel"]); mf.pack(fill="x", padx=12, pady=(0,8))
        self._lb = tk.StringVar(); self._cb = tk.StringVar()
        tk.Label(mf, textvariable=self._lb, bg=C["accent2"], fg="white",
                 font=self.fS, padx=6, pady=2).pack(side="left")
        tk.Label(mf, textvariable=self._cb, bg=C["panel"], fg=C["dim"],
                 font=self.fS, padx=8).pack(side="left")

        # 로그
        lo = tk.Frame(right, bg=C["border"], padx=1, pady=1)
        lo.pack(fill="both", expand=True)
        li = tk.Frame(lo, bg=C["panel"]); li.pack(fill="both", expand=True)
        lh = tk.Frame(li, bg=C["panel2"]); lh.pack(fill="x")
        tk.Label(lh, text="📋  교신 로그", bg=C["panel2"], fg=C["accent"],
                 font=self.fH, anchor="w", padx=10, pady=5).pack(side="left")
        tk.Button(lh, text="지우기", font=self.fS, bg=C["accent2"], fg="white",
                  relief="flat", padx=8, pady=3, cursor="hand2",
                  command=self._clear).pack(side="right", padx=8, pady=4)
        self._log = scrolledtext.ScrolledText(li, bg=C["log_bg"], fg=C["log_n"],
                                              font=self.fL, wrap="word",
                                              relief="flat", bd=0, padx=8, pady=6,
                                              state="disabled")
        self._log.pack(fill="both", expand=True)
        self._log.tag_config("e", foreground=C["log_em"],
                             font=font.Font(family="Courier New",size=10,weight="bold"))
        self._log.tag_config("n", foreground=C["log_n"])

        # 상태바
        bar = tk.Frame(self.root, bg=C["hdr"], height=26)
        bar.pack(fill="x", side="bottom"); bar.pack_propagate(False)
        self._bv = tk.StringVar(value=f"  로그 저장: {os.path.abspath(LOG_DIR)}")
        tk.Label(bar, textvariable=self._bv, bg=C["hdr"], fg=C["dim"],
                 font=self.fS, anchor="w").pack(side="left", padx=10)

    def _card(self, parent, title):
        C = self.C
        o = tk.Frame(parent, bg=C["border"], pady=1, padx=1)
        o.pack(fill="x", pady=4)
        i = tk.Frame(o, bg=C["panel"]); i.pack(fill="both")
        tk.Label(i, text=title, bg=C["panel2"], fg=C["accent"],
                 font=self.fH, anchor="w", padx=10, pady=5).pack(fill="x")
        b = tk.Frame(i, bg=C["panel"], pady=6, padx=10)
        b.pack(fill="both", expand=True)
        return b

    def _load_model(self):
        self.stt.load(lambda m: self.root.after(0, lambda: self._sv.set(m)))
        if self.stt.ready:
            self.root.after(0, lambda: (self._dot.config(fg=self.C["green"]),
                                        self._btn.config(state="normal")))
        else:
            self.root.after(0, lambda: self._dot.config(fg=self.C["emergency"]))

    def _toggle(self):
        if not self._is_recording:
            self._is_recording = True
            self._btn.config(text="⏹  녹음 중지", bg=self.C["emergency"])
            self._dot.config(fg=self.C["emergency"])
            self._sv.set("녹음 중...")
            self.recorder.start()
            self._vu_anim()
        else:
            self._is_recording = False
            self.recorder.stop()
            self._btn.config(text="▶  녹음 시작", bg=self.C["accent2"])
            self._dot.config(fg=self.C["green"])
            self._sv.set("대기 중")
            self._level = 0.0

    def _vu_anim(self):
        if not self._is_recording: return
        c = self._vu; c.delete("all")
        w = c.winfo_width() or 280
        bw = int(w * min(self._level * 8, 1.0))
        if bw > 0:
            col = self.C["emergency"] if self._level > 0.1 else self.C["green"]
            c.create_rectangle(0, 0, bw, 14, fill=col, outline="")
        self.root.after(80, self._vu_anim)

    def _poll(self):
        try:
            while True:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple) and item[0]=="ERROR":
                    self._add_log(datetime.now().strftime("%H:%M:%S"),
                                  "sys",0.0,False,f"[오디오 오류] {item[1]}")
                else:
                    self._level = float(np.sqrt(np.mean(item**2)))
                    threading.Thread(target=self._process,
                                     args=(item,), daemon=True).start()
        except queue.Empty: pass
        self.root.after(100, self._poll)

    def _process(self, audio):
        text, lang, conf = self.stt.transcribe(audio, self._lang_override)
        if not text or text.startswith("[오류"): return
        ts = datetime.now().strftime("%H:%M:%S")
        is_e = check_emergency(text)
        self.log_manager.write(ts, lang, conf, is_e, text)
        self.root.after(0, lambda: self._update(ts, lang, conf, is_e, text))

    def _update(self, ts, lang, conf, is_e, text):
        self._total += 1
        if is_e: self._emerg += 1
        self._sub.set(text)
        names = {"ko":"한국어","en":"English","ja":"日本語","zh":"中文"}
        self._lb.set(f" {names.get(lang,lang.upper())} ")
        self._cb.set(f"신뢰도 {conf*100:.0f}%")
        self._st.set(str(self._total))
        self._se.set(str(self._emerg))
        self._sl.set(names.get(lang, lang.upper()))
        if is_e:
            self._el.config(text=f"🚨  긴급 상황 감지!  {text}")
            self._ef.pack(fill="x", pady=(0,4))
            self.root.after(8000, self._ef.pack_forget)
        self._add_log(ts, lang, conf, is_e, text)

    def _add_log(self, ts, lang, conf, is_e, text):
        self._log.config(state="normal")
        tag = "e" if is_e else "n"
        self._log.insert("end",
            f"{'🚨' if is_e else '◈ '} {ts}  [{lang.upper()}]  {text}\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0","end")
        self._log.config(state="disabled")

    def _tick(self):
        self._clk.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick)

def main():
    root = tk.Tk()
    app = MaritimeSTTApp(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app.recorder.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
