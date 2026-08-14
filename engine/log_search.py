#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_search.py — 교신 로그 검색·조회.
데모가 남긴 log_*.csv(교신 기록)에서 조건으로 검색한다.

컬럼: time, speaker, lang, text, translation, danger, wav, analysis

사용 예:
  python log_search.py --q 침수                 # 본문/번역/분석에 '침수' 포함
  python log_search.py --danger                 # 위험(조난어 탐지)만
  python log_search.py --speaker 화자2          # 특정 화자
  python log_search.py --lang en                # 특정 언어
  python log_search.py --q 메이데이 --danger    # 조건 결합
"""
import argparse, csv, glob, os


def rows(pattern="log_*.csv"):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_file"] = os.path.basename(path)
                yield r


def match(r, q, danger, speaker, lang):
    if q:
        hay = " ".join([r.get("text", ""), r.get("translation", ""), r.get("analysis", "")])
        if q not in hay:
            return False
    if danger and not (r.get("danger") or "").strip():
        return False
    if speaker and speaker not in (r.get("speaker") or ""):
        return False
    if lang and lang != (r.get("lang") or ""):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=None, help="본문/번역/분석 포함 검색어")
    ap.add_argument("--danger", action="store_true", help="위험(조난어 탐지)만")
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--files", default="log_*.csv", help="로그 파일 glob")
    a = ap.parse_args()

    hits = [r for r in rows(a.files) if match(r, a.q, a.danger, a.speaker, a.lang)]
    if not hits:
        print("일치하는 교신 없음."); return
    print(f"검색 결과 {len(hits)}건\n" + "=" * 70)
    for r in hits:
        flag = "🚨" if (r.get("danger") or "").strip() else "  "
        print(f"{flag} [{r.get('time','')}] {r.get('speaker','')} ({r.get('lang','')})")
        print(f"     {r.get('text','')}")
        if r.get("translation"):
            print(f"     ↳ 번역: {r['translation']}")
        if (r.get("analysis") or "").strip():
            print(f"     ↳ 분석: {r['analysis']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
