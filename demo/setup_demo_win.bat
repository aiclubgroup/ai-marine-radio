@echo off
chcp 65001 >nul
REM =============================================================================
REM setup_demo_win.bat — 실시간 STT 데모 설치 스크립트 (Windows)
REM
REM 사용법: 이 파일과 realtime_stt_gui.py 를 같은 폴더에 두고 더블클릭
REM         (또는 명령 프롬프트에서 setup_demo_win.bat 실행)
REM
REM 사전 준비: 파이썬 3.9+ 가 설치되어 있어야 함 (python.org 설치판 권장 —
REM            설치 시 "Add Python to PATH" 체크!)
REM =============================================================================

echo === [1/4] 파이썬 확인 ===
python --version >nul 2>&1
if errorlevel 1 (
  echo 파이썬이 없습니다. https://www.python.org/downloads/ 에서 설치하세요.
  echo 설치할 때 "Add Python to PATH" 반드시 체크!
  pause
  exit /b 1
)
python --version

echo === [2/4] 가상환경 생성 ===
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q

echo === [3/4] 패키지 설치 (몇 분 걸릴 수 있음) ===
pip install -q faster-whisper sounddevice soundfile numpy
REM 윈도우는 sounddevice에 PortAudio가 내장되어 별도 설치 불필요

echo === [4/4] 설치 검증 ===
python -c "import faster_whisper, sounddevice, soundfile, numpy, tkinter; print('  모든 모듈 OK'); devs=[d for d in sounddevice.query_devices() if d['max_input_channels']>0]; print('  마이크 장치:', devs[0]['name'] if devs else '없음!')"

echo.
echo =============================================
echo  설치 완료! 실행 방법:
echo    venv\Scripts\activate
echo    python realtime_stt_gui.py --model small
echo.
echo  * 첫 실행 시 모델(~500MB) 자동 다운로드 — 인터넷 필요
echo  * 마이크 권한 팝업이 뜨면 허용
echo  * 느리면: --model base 로 낮춰 실행
echo =============================================
pause
