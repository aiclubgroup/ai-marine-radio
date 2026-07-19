#!/bin/bash
# =============================================================================
# setup_demo_mac.sh — 실시간 STT 데모 설치 스크립트 (macOS)
#
# 사용법: 터미널에서
#   cd ~/Documents/Claude/Projects/ICT-해상무전기/demo
#   bash setup_demo_mac.sh
#
# 하는 일: portaudio 설치 → 파이썬 가상환경 생성 → 필요 패키지 설치 → 검증
# =============================================================================
set -e
echo "=== [1/5] Homebrew 확인 ==="
if ! command -v brew &> /dev/null; then
  echo "Homebrew가 없습니다. 먼저 설치하세요: https://brew.sh"
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi
echo "OK"

echo "=== [2/5] portaudio 설치 (마이크 입력용) ==="
brew list portaudio &>/dev/null || brew install portaudio
echo "OK"

echo "=== [3/5] 파이썬 가상환경 생성 (venv) ==="
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip -q
echo "OK ($(python --version))"

echo "=== [4/5] 패키지 설치 (몇 분 걸릴 수 있음) ==="
pip install -q faster-whisper sounddevice soundfile numpy
echo "OK"

echo "=== [5/5] 설치 검증 ==="
python - << 'EOF'
import faster_whisper, sounddevice, soundfile, numpy
print("  faster-whisper OK")
try:
    import tkinter
    if tkinter.TkVersion < 8.6:
        print(f"  [경고] 구형 Tk {tkinter.TkVersion} 감지 — GUI가 하얗게(스타일 없이) 나옵니다!")
        print("         해결: python.org에서 Python 3.12 설치 후, rm -rf venv 하고")
        print("         python3.12 -m venv venv 로 이 스크립트를 다시 실행하세요.")
    else:
        print(f"  tkinter OK (Tk {tkinter.TkVersion}, GUI 정상)")
except ImportError:
    print("  [주의] tkinter 없음 → brew install python-tk 후 다시 실행")
devs = [d for d in sounddevice.query_devices() if d['max_input_channels'] > 0]
print(f"  마이크 입력 장치 {len(devs)}개 감지:", devs[0]['name'] if devs else "없음!")
EOF

echo ""
echo "============================================="
echo " 설치 완료! 실행 방법:"
echo "   source venv/bin/activate"
echo "   python realtime_stt_gui.py --model small"
echo ""
echo " * 첫 실행 시 모델(~500MB) 자동 다운로드 — 인터넷 필요"
echo " * 마이크 권한 팝업이 뜨면 '허용' 클릭"
echo " * 느리면: --model base 로 낮춰 실행"
echo "============================================="
