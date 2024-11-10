# AI - GPT-soVITS-smallest-Inference

## 개요

- GPT-soVITS v2를 실행하기 위한 최소 라이브러리 및 함수 세팅
- 백엔드 통신을 통한 음성합성 기능

## 환경 세팅

- venv, library 세팅

    ``` bash
        py -3.9 -m venv venv

        pip install einops
        pip install LangSegment==0.3.5
        pip install pyinstaller
        pip install pytorch-lightning
        pip install soundfile
        pip install eunjeon
        pip install transformers
        pip install numpy==1.23.4
        pip install scipy
        pip install librosa==0.9.2
        pip install matplotlib
        pip install pyopenjtalk==0.3.4
        pip install jamo
        pip install ko_pron
        pip install g2p_en
        pip install g2pk2
        pip install wordsegment
        pip install ffmpeg-python
        pip install numba==0.56.4
        pip install pandas
    ```
