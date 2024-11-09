# AI - GPT-soVITS-smallest-Inference

## 개요

- git pu

## 환경 기록

py -3.9 -m venv venv

pip install einops
pip install LangSegment>=0.2.0
pip install pyinstaller

eunjeon 이 프로그램 호출로 install 되어버림 (Using cached eunjeon-0.4.0-cp310-cp310-win_amd64.whl)

## 빌드

pyinstaller --onedir inference_webui.py -n main --noconsole --contents-directory=files --noconfirm # 메인 프로그램
