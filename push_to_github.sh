#!/bin/bash
# 이 폴더의 내용을 aiclubgroup/ai-marine-radio 의 main 브랜치에 push
# 사용법: ./push_to_github.sh   (GitHub 로그인 되어 있어야 함)
set -e
cd "$(dirname "$0")"

if [ ! -d .git ]; then
  git init
  git remote add origin https://github.com/aiclubgroup/ai-marine-radio.git
fi

git fetch origin
# 기존 히스토리 위에 현재 폴더 상태를 얹는다 (파일은 그대로 유지)
git reset --mixed origin/main

git add -A
git commit -m "feat: 서울팀 STT 엔진 통합 + PySide6 UI 실엔진 연동판 (v2.0)

- engine/: STT 엔진 (도메인 프롬프트, user_dict, 후처리, 번역, WS 서버)
- seatalk_ai_vhf_connected.py: PySide6 UI를 WS 규격으로 실엔진에 연동
- 비상 오버레이에 실제 감지 문장·키워드 표시
- run_live_demo.sh: 엔진+UI 통합 실행
- requirements/README 정리 (Jetson Orin Nano 메모 포함)"

git push origin HEAD:main
echo ""
echo "✅ push 완료! 팀원들은 git pull 하면 됩니다."
