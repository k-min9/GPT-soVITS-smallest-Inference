import voice_inference
import util_pyngrok

# Server-Flask
from flask import Flask, Response, request, jsonify, send_file, abort
from waitress import serve
app = Flask(__name__)

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
    

if __name__ == '__main__':
    # TODO : 선행세팅, main에서 local화 진행시 같이 진행
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    # preloading
    voice_inference.synthesize_char('noa', '안녕하세요!', audio_language='ja')
    util_pyngrok.start_ngrok(id='dev_voice')
    
    # Server run
    tts_port = 5000
    app.run( host='0.0.0.0', port=tts_port)
    # serve(app, host="0.0.0.0", port=5000)
