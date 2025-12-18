# AI - GPT-soVITS-smallest-Inference

## 개요

- GPT-soVITS v2를 실행하기 위한 최소 라이브러리 및 함수 세팅
- 백엔드 통신을 통한 음성합성 기능

## 환경 세팅

- venv, library 세팅

    ``` bash
        py -3.10 -m venv venv

        pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        pip install einops
        pip install LangSegment==0.3.5
        pip install pyinstaller
        pip install pytorch-lightning
        pip install soundfile
        pip install eunjeon
        pip install transformers==4.46.2
        pip install numpy==1.23.4
        pip install scipy
        pip install librosa==0.9.2
        pip install matplotlib
        pip install pyopenjtalk==0.3.4  # cmake needed
        pip install jamo
        pip install ko_pron
        pip install g2p_en
        pip install g2pk2
        pip install wordsegment
        pip install ffmpeg-python
        pip install numba==0.56.4
        pip install pandas
        pip install pyngrok
        pip install supabase 

        # Backserver 추가(여기서부터 torch 붙음)
        pip install silero-vad
        pip install faster-whisper
        pip install pyannote-audio

        pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        pip install torchaudio==2.5.1
        pip install torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

        pip install flask
        pip install waitress  # WSGI for production
    ```

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)의 해당항목 이동
  - 최상단에 ffmpeg.exe, ffprobe.exe 세팅
  - voices 이동
    - GPT_weights_v2/로 ckpt 이동
    - SoVITS_weights_v2/로 pth 이동
  - venv의 LangSegment, pyopenjtalk 이동
  - pretrained_models 이동

## 빌드

- pyinstaller --onedir main.py -n main --noconsole --contents-directory=files --noconfirm # 메인 프로그램
- pyinstaller --onedir tts_backend.py -n server --contents-directory=files_server --noconfirm # 서버 인터페이스
- --icon=./icon_plana.ico
- 몇몇 라이브러리 이동 필요

## 트러블 슈팅

- torch jit (torch\jit\_script.py) 이슈
  - AR 폴더 변경사항
  - @torch.jit.script > @torch.jit._script_if_tracing
- text 의 cleaner에 import 추가하여 강제 로딩
  - import text.japanese  
    import text.korean  
    import text.english 추가  
