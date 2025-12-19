import voice_inference
import util_pyngrok
import util_silerovad
import util_speech_diarization

import os
import shutil
import uuid

# Server-Flask
from flask import Flask, Response, request, jsonify, send_file, abort
from waitress import serve
app = Flask(__name__)

# Health check endpoint - 접근성을 낮춘 경로
# 차후 공격시 Rate Limiting + 큐 시스템 + 캐싱
@app.route('/alive', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'tts-voice-service',
        'timestamp': int(__import__('time').time())
    }), 200

# 한국어 텍스트를 입력받아 변환
@app.route('/getSound/jp', methods=['POST'])  # legacy
@app.route('/getSound/ko', methods=['POST'])  # legacy
@app.route('/getSound', methods=['POST'])
def synthesize_sound():
    def get_sound_text_ja(text):
        # text = text.lower()
        text = text.replace('RABBIT', 'ラビット')
        text = text.replace('SCHALE', 'シャーレ')
        return text   
    
    if True:  # state.get_DEV_MODE():
        print('###getSound request', request.json)
    text = request.json.get('text', '안녕하십니까.')
    char = request.json.get('char', 'arona')
    lang = request.json.get('lang', 'ko')
    speed = request.json.get('speed', 100)  # % 50~100
    speed = float(speed)/100 
    chat_idx = request.json.get('chatIdx', '-1')  
    
    if lang == 'ja' or lang =='jp':
        lang = 'ja'  # 단어보정
        text = get_sound_text_ja(text) 
    
    
    result = voice_inference.synthesize_char(char, text, audio_language=lang, speed=speed)  # 'output*.wav'
    if result == 'early stop':
        abort(500, description="Synthesis process stopped early.")
    response = send_file(result, mimetype="audio/wav")
    response.headers['Chat-Idx'] = chat_idx
    return response
    
# wav에 번역포함 답변 답변
@app.route('/stt', methods=['POST'])
def main_stream_stt():  # main logic
    def transcribe_audio_to_text(audio_path, expected_stt_lang='en', model_name= "small") -> str:
        from faster_whisper import WhisperModel
        try:
            # Load the Whisper model
            print(f"Loading Whisper model: {model_name}...")
            model = WhisperModel(model_name, device="cpu", download_root='./model')
            
            # Transcribe the audio file
            print(f"Transcribing {audio_path}...")
            segments, info = model.transcribe(audio_path)
            text =""
            for segment in segments:
                text = text + segment.text 
            # if state.get_DEV_MODE():
            #     print('stt response :', text.lower(), '-', info.language)    
                
            return text.lower(), info.language
        except Exception as e:
            print(f"Error occurred during transcription: {e}")
            return ""
    
    # STT 변수
    try:
        # Get 'lang' and 'level' from the form data
        stt_lang = request.form.get('lang', 'ko') 
        stt_level = request.form.get('level', 'small')  
        stt_chat_idx = request.form.get('chatIdx', '-1')  
        
        # Handle the uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        
        # 파일 저장
        audio_path = os.path.join('./files', f"{uuid.uuid4()}.wav")  # 충돌방지용
        os.makedirs('./files', exist_ok=True)
        file.save(audio_path)
        
        # 최소 0.3초 이상일 경우에만 판단
        trim_silence_len = util_silerovad.get_trim_silence_len(audio_path)
        if  trim_silence_len < 0.3:  # '안녕' 정도의 길이
            print(f"too short wav : {trim_silence_len}s")
            # # Test용 파일저장
            # if state.get_DEV_MODE():
            #     stt_file_name = "stt_" + str(datetime.now().strftime("%y%m%d_%H%M%S")) + "_short.wav"
            #     stt_audio_path = os.path.join('./test/stt', stt_file_name)  # 충돌방지용
            #     os.makedirs('./test/stt', exist_ok=True) 
            #     # 기존 저장된 파일을 복사
            #     shutil.copy(audio_path, stt_audio_path)
                
            # Clean up the temporary file
            os.remove(audio_path)  # 충돌주의
                
            return jsonify({"error": f"too short wav : {trim_silence_len}s"}), 500 

        # Transcribe the audio file
        trans_text, trans_lang = transcribe_audio_to_text(audio_path, stt_lang, stt_level)
        
        # # Test용 파일저장
        # if state.get_DEV_MODE():
        #     try:
        #         stt_file_name = "stt_" + str(datetime.now().strftime("%y%m%d_%H%M%S")) + "_" + trans_text + ".wav"
        #         stt_audio_path = os.path.join('./test/stt', stt_file_name)  # 충돌방지용
        #         os.makedirs('./test/stt', exist_ok=True)
        #         # 기존 저장된 파일을 복사
        #         shutil.copy(audio_path, stt_audio_path)
        #     except:
        #         # trans_text가 파일명으로 쓰기 힘든 경우 우려
        #         print('fail saving stt wav')

        # Clean up the temporary file
        os.remove(audio_path)  # 충돌남

        # Build the response
        response = {"text": trans_text, "lang": trans_lang, "chatIdx":stt_chat_idx}
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /stt endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500
    # finally:
    #     # 파일 삭제
    #     if os.path.exists(audio_path):
    #         os.remove(audio_path)

# 음성 화자 분석 및 필터링
@app.route('/speech_diarization', methods=['POST'])
def speech_diarization_filter():
    """
    음성 화자 분석을 통한 음성 필터링
    
    Parameters:
    - file: wav 파일
    - player: 플레이어 이름 (기본값: sensei)
    - char: 캐릭터 이름 (기본값: arona)
    - ai_voice_filter_idx: 필터 모드
      - 0: 무조건 False 반환 (무시)
      - 1: 캐릭터와 음성 일치 여부에 따라 True/False
      - 2: 무조건 False 반환 (무시)
    
    Returns:
    - should_ignore: bool (True면 무시해야 함)
    - similarity: float (유사도 점수, mode=1일 때만 유효)
    - character: str (비교 대상 캐릭터)
    """
    try:
        # 파라미터 가져오기
        player_name = request.form.get('player', 'sensei')
        char_name = request.form.get('char', 'arona')
        ai_voice_filter_idx = request.form.get('ai_voice_filter_idx', '0')
        
        # 파일 체크
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "should_ignore": True,
                "similarity": 0.0,
                "character": char_name
            }), 400
            
        file = request.files['file']
        
        # 파일 저장
        audio_path = os.path.join('./files', f"speech_check_{uuid.uuid4()}.wav")
        os.makedirs('./files', exist_ok=True)
        file.save(audio_path)
        
        # 필터 모드에 따른 처리
        if ai_voice_filter_idx == '0':
            # 모드 0: 무조건 False (무시하지 않음)
            result = {
                "should_ignore": False,
                "similarity": 0.0,
                "character": char_name,
                "mode": "disabled"
            }
            
        elif ai_voice_filter_idx == '1':
            # 모드 1: 캐릭터 음성 일치 여부 확인
            try:
                # 음성 화자 분석 수행
                speaker_result = util_speech_diarization.identify_speaker(
                    input_audio_path=audio_path,
                    character_name=char_name,
                    threshold=0.6,  # 기본 임계값
                    model_type='ecapa',  # 기본 모델 (빠른 처리용)
                    use_gpu=False  # CPU 사용 (안정성 우선)
                )
                
                similarity = float(speaker_result['similarity'])  # numpy float을 Python float으로 변환
                is_match = bool(speaker_result['is_match'])  # numpy bool_을 Python bool로 변환
                
                result = {
                    "should_ignore": is_match,  # 일치하면 무시
                    "similarity": similarity,
                    "character": char_name,
                    "threshold": float(speaker_result['threshold']),  # 이것도 변환
                    "mode": "character_match",
                    "match_status": "match" if is_match else "no_match"
                }
                
            except Exception as e:
                
                # 에러 발생시 안전하게 무시하지 않음
                result = {
                    "should_ignore": False,
                    "similarity": 0.0,
                    "character": char_name,
                    "mode": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
        elif ai_voice_filter_idx == '2':
            # 모드 2: 무조건 False (무시하지 않음)
            result = {
                "should_ignore": False,
                "similarity": 0.0,
                "character": char_name,
                "mode": "disabled"
            }
            
        else:
            # 잘못된 모드: 안전하게 무시하지 않음
            result = {
                "should_ignore": False,
                "similarity": 0.0,
                "character": char_name,
                "mode": "invalid",
                "error": f"Invalid ai_voice_filter_idx: {ai_voice_filter_idx}"
            }
        
        # 임시 파일 삭제
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print('### /speech_diarization\n', result)
        print('###')
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Main exception in /speech_diarization endpoint: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Main traceback: {traceback.format_exc()}")
        
        # 임시 파일 삭제 (에러 시에도)
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                print(f"[DEBUG] Cleaning up temp file: {audio_path}")
                os.remove(audio_path)
        except Exception as cleanup_e:
            print(f"[ERROR] Error during cleanup: {cleanup_e}")
            
        return jsonify({
            "error": "Internal server error",
            "should_ignore": False,  # 에러 시 안전하게 무시하지 않음
            "similarity": 0.0,
            "character": char_name if 'char_name' in locals() else 'unknown',
            "error_details": str(e),
            "error_type": type(e).__name__
        }), 500

# ===== 캐시 관리 API =====

@app.route('/cache/clear_all', methods=['POST'])
def cache_clear_all():
    """모든 캐시된 actor 모델 제거"""
    try:
        voice_inference.clear_all_cache()
        return jsonify({
            "status": "success",
            "message": "All cached models cleared"
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/clear_all: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/cache/remain', methods=['POST'])
def cache_remain():
    """최근 사용된 N개 actor만 남기고 나머지 제거"""
    try:
        count = request.json.get('count', 2)
        count = int(count)
        if count < 0:
            return jsonify({
                "status": "error",
                "message": "count must be non-negative"
            }), 400
        
        voice_inference.remain_n_actors(count)
        return jsonify({
            "status": "success",
            "message": f"Kept only {count} most recent actors",
            "count": count
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/remain: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/cache/remain/<int:n>', methods=['GET'])
def cache_remain_get(n):
    """최근 사용된 N개 actor만 남기고 나머지 제거 (테스트용)"""
    try:
        if n < 0:
            return jsonify({
                "status": "error",
                "message": "n must be non-negative"
            }), 400
        
        voice_inference.remain_n_actors(n)
        return jsonify({
            "status": "success",
            "message": f"Kept only {n} most recent actors",
            "count": n
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/remain/{n}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/cache/load', methods=['POST'])
def cache_load():
    """특정 actor 모델을 미리 로딩"""
    try:
        actor = request.json.get('actor', None)
        if not actor:
            return jsonify({
                "status": "error",
                "message": "actor parameter is required"
            }), 400
        
        voice_inference.preload_actor(actor)
        return jsonify({
            "status": "success",
            "message": f"Actor {actor} loaded",
            "actor": actor
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/load: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/cache/load/<actor>', methods=['GET'])
def cache_load_get(actor):
    """특정 actor 모델을 미리 로딩 (테스트용)"""
    try:
        voice_inference.preload_actor(actor)
        return jsonify({
            "status": "success",
            "message": f"Actor {actor} loaded",
            "actor": actor
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/load/{actor}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/cache/status', methods=['GET'])
def cache_status():
    """현재 캐시 상태 조회"""
    try:
        cached_vq = voice_inference.vq_models.keys()
        cached_t2s = voice_inference.t2s_models.keys()
        vram_info = voice_inference.get_vram_info()
        
        return jsonify({
            "status": "success",
            "cached_actors": cached_vq,
            "count": len(cached_vq),
            "max_cache_size": voice_inference.MAX_CACHED_ACTORS,
            "vram": vram_info
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/status: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # TODO : 선행세팅, main에서 local화 진행시 같이 진행
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    # 로그 시스템 초기화
    voice_inference.init_logging()
    
    # preloading
    voice_inference.synthesize_char('arona', '안녕하세요!', audio_language='ja')
    util_pyngrok.start_ngrok(id='dev_voice')
    
    # Server run
    tts_port = 5010
    # app.run( host='0.0.0.0', port=tts_port)
    serve(app, host="0.0.0.0", port=tts_port)
