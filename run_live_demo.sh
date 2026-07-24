#!/bin/bash
# SeaTalk AI 실엔진 연동 데모 실행 (엔진 + PySide6 UI 동시 시작)
# 사용법: ./run_live_demo.sh [--model small] [--translate]
set -e
cd "$(dirname "$0")"

MODEL_ARGS="${@:---model small --translate}"

echo "[1/2] STT 엔진 시작 중... (로그: engine.log)"
python3 engine/realtime_stt_web.py $MODEL_ARGS --no-browser > engine.log 2>&1 &
ENGINE_PID=$!
trap "kill $ENGINE_PID 2>/dev/null" EXIT

echo "[2/2] SeaTalk UI 시작 (엔진 준비되면 PTT 버튼이 활성화됩니다)"
python3 seatalk_ai_vhf_connected.py

# UI를 닫으면 엔진도 함께 종료됩니다 (trap)
