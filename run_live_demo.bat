@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/2] STT 엔진을 별도 창으로 시작합니다... (로그가 그 창에 표시됨)
start "SeaTalk Engine" cmd /k python engine\realtime_stt_web.py --model small --translate --no-browser

echo [2/2] SeaTalk UI를 시작합니다. (엔진 준비되면 PTT 버튼이 활성화됩니다)
python seatalk_ai_vhf_connected.py

echo UI가 종료되었습니다. 엔진 창(SeaTalk Engine)도 닫아주세요.
pause
