import main

# Server-Flask
from flask import Flask, Response, request, jsonify, send_file
app = Flask(__name__)

# 일본어 텍스트를 입력받아 변환
@app.route('/getSound/jp', methods=['GET', 'POST'])
def synthesize_jp():
    text = '오늘은 맑은 날씨 때때로 흐림입니다.'
    text_language = 'ko'
    char = 'arona'
    try:
        # POST test
        text = request.json.get('text', 'API 테스트중! 테스트중.')
        text_language = request.json.get('text_language', 'ko')
        char = request.json.get('char', 'arona')
    except:
        try:
            # GET test
            text = request.args.get('text', 'GET 테스트중? 테스트중!')
            text_language = request.args.get('text_language', 'ko')
            char = request.args.get('char', 'arona')      
            print('###Get Test :', text, text_language, char)
        except:
            pass
    
    # 기본값 = arona
    ref_wav_path = './voices/arona.wav'
    prompt_text = 'メリークリスマス。プレゼントもちゃんと用意しましたよ'
    prompt_language = 'ja'
    gpt_path = 'voices/arona-e15.ckpt'
    sovits_path = 'voices/arona_e8_s296.pth'
    if char == 'prana':
        ref_wav_path = './voices/prana.wav'
        prompt_text = '混乱。理解できない行動です。つつかないで下さい。故障します。'
        prompt_language = 'ja'
        gpt_path = 'voices/prana-e15.ckpt'
        sovits_path = 'voices/prana_e8_s72.pth'
    elif char == 'mika':
        ref_wav_path = './voices/mika.wav'
        prompt_text = 'おかえり、先生！ちゃーんといい子でお留守番してたよ。'
        prompt_language = 'ja'
        gpt_path = 'voices/mika-e15.ckpt'
        sovits_path = 'voices/mika_e8_s160.pth'
    elif char == 'noa':
        ref_wav_path = './voices/noa.wav'
        prompt_text = 'お疲れ様でした、先生。勉強になりました。'
        prompt_language = 'ja'
        gpt_path = 'voices/noa-e15.ckpt'
        sovits_path = 'voices/noa_e8_s192.pth'

    main.change_sovits_weights(sovits_path)
    main.change_gpt_weights(gpt_path)

    main.get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language)
    return send_file('output.wav', mimetype="audio/wav")
    

if __name__ == '__main__':
    # TODO : 선행세팅, main에서 local화 진행시 같이 진행
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    gpt_path = 'voices/arona-e15.ckpt'
    sovits_path = 'voices/arona_e8_s296.pth'
    main.change_sovits_weights(sovits_path)
    main.change_gpt_weights(gpt_path)
    
    # Server run
    tts_port = 5000
    app.run( host='0.0.0.0', port=tts_port)
