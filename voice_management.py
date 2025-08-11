'''
voices 이하 파일 관리용 CRUD
'''
import json
import os

FILE_PATH = './voices/info.json'

# sample
info_sample = {
    "voices": [],
    "prompts": []
}

voice_sample = {
    "name": "arona",  # 보여줄 이름 # 실제 다운로드 시 생성
    "gpt_path" : "voices/arona-e15.ckpt",  # url을 알기 쉽게 표현한 이름
    "sovits_path": "voices/arona_e8_s296.pth",  # 파일 위치  # 실제 다운로드 시 생성
    "url": "https://huggingface.co/spaces/zomehwh/vits-models/resolve/main/pretrained_models/mika/mika.pth",
}

prompt_sample = {
    "name" : "arona",
    "prompts" : {
        "normal" : {
            'language': 'ja',
            'wav_path': './voices/arona/1.wav',
            'text': 'メリークリスマス。プレゼントもちゃんと用意しましたよ'
        },
        "angry" : {
            'language': 'ja',
            'wav_path': './voices/arona/56.wav',
            'text': 'メリークリスマス。プレゼントもちゃんと用意しましたよ'
        }
    }
}

def get_voice_name():
    # JSON 파일에서 데이터 불러오기
    with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    name_catalogs = list()
    for voice in json_data["voices"]:
        name_catalogs.append(voice["name"])

    # 생성된 딕셔너리 출력
    return name_catalogs


def get_voice_info_from_name(name):
    # JSON 파일에서 데이터 불러오기
    with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        
    for voice in json_data["voices"]:
        if voice["name"] == name:
            return voice
    return None

def add_voice_info(name='', gpt_path='', sovits_path='', url='', update_flag=False):
    # voices 폴더 없으면 생성
    if not os.path.exists('./voices'):
        os.makedirs('./voices')    
        
    try:
        # 기존 JSON 파일이 있는 경우 데이터 불러오기
        with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            
    except FileNotFoundError:
        # 기존 JSON 파일이 없는 경우 빈 데이터 생성
        json_data = {
            "voices": [],
            "prompts": []
        }

    # 기존 데이터에 이미 존재하는 name인지 확인
    for voice in json_data["voices"]:
        if voice["name"] == name:
            if update_flag:
                # 덮어쓰기 허용 시 업데이트
                voice['gpt_path'] = gpt_path
                voice['sovits_path'] = sovits_path
                voice['url'] = url
                break
            else:
                # 덮어쓰기 비허용 시 False 반환
                return False
    else:
        # 기존 데이터 없음-새로운 voice 추가
        new_voice = dict()
        new_voice['name'] = name
        new_voice['gpt_path'] = gpt_path
        new_voice['sovits_path'] = sovits_path
        new_voice['url'] = url
        json_data["voices"].append(new_voice)
    
    # 새로운 데이터를 JSON 파일에 덤프
    with open(FILE_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    
    # 무사 종료시 True
    return True

# voices catalog에서 해당 이름 제거 
def remove_voice_by_name(name):
    try:
        # 기존 JSON 파일이 있는 경우 데이터 불러오기
        with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        for voice in json_data["voices"]:
            if voice["name"] == name:
                json_data["voices"].remove(voice)
                
        # 새로운 데이터를 JSON 파일에 덤프
        with open(FILE_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    except:
        pass   

# Prompt 정보 추가
def add_prompt_info(name='', emotion='normal', language='ja', wav_path='', text='', update_flag=False):
    # JSON 데이터 불러오기
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        # 기존 JSON 파일이 없는 경우 초기화
        json_data = {
            "voices": [],
            "prompts": []
        }
    
    # 해당 name이 존재하는지 확인
    for prompt in json_data["prompts"]:
        if prompt["name"] == name:
            if emotion in prompt["prompts"]: # 해당 emotion이 기존재
                if update_flag:
                    # 덮어쓰기 허용 시 업데이트
                    prompt["prompts"][emotion] = {
                        'language': language,
                        'wav_path': wav_path,
                        'text': text
                    }
                    break
                else:
                    # 덮어쓰기 비허용 시 False 반환
                    return False
            else:
                # 새로운 emotion 추가
                prompt["prompts"][emotion] = {
                    'language': language,
                    'wav_path': wav_path,
                    'text': text
                }
                break
    else:
        # 새 name과 prompt 추가
        new_prompt = {
            "name": name,
            "prompts": {
                emotion: {
                    'language': language,
                    'wav_path': wav_path,
                    'text': text
                }
            }
        }
        json_data["prompts"].append(new_prompt)

    # JSON 데이터 업데이트
    with open(FILE_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    
    return True

# 특정 캐릭터의 변수가 없을
def get_prompt_info_from_name(name, emotion='normal'):
    # JSON 파일에서 데이터 불러오기
    with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        
    for prompt in json_data["prompts"]:
        if prompt["name"] == name:
            prompt_info = prompt["prompts"].get('normal')  # TODO : 반드시 있어야 함. 없을때의 대비 필요
            try:
                 prompt_info = prompt["prompts"].get(emotion, prompt_info)
            except:
                pass
            return prompt_info
    return None

# emotion이 없으면 통째로 삭제 있을 경우, 그 emotion 정보만 삭제
def remove_prompt_by_name(name, emotion=''):
    try:
        # JSON 데이터 불러오기
        with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        
        if emotion:  #  emotion 입력있음
            for prompt in json_data["prompts"]:
                if prompt["name"] == name:
                    if emotion in prompt["prompts"]:
                        del prompt["prompts"][emotion]
                        # 해당 name의 모든 emotion 삭제 시 prompt 자체 제거
                        if not prompt["prompts"]:
                            json_data["prompts"].remove(prompt)
                        break
        else:  # 이름 관련 삭제
            for prompt in json_data["prompts"]:
                if prompt["name"] == name:
                    json_data["prompts"].remove(prompt)
                    break


        # JSON 데이터 업데이트
        with open(FILE_PATH, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
    except:
        pass   
    
if __name__ == "__main__":
    # 최초 카탈로그 생성
    def init_voices_info():
        add_voice_info(name='arona', gpt_path='voices/arona-e15.ckpt', sovits_path='voices/arona_e8_s248.pth', url='', update_flag=False)
        add_voice_info(name='prana', gpt_path='voices/prana-e15.ckpt', sovits_path='voices/prana_e8_s72.pth', url='', update_flag=False)
        add_voice_info(name='mika', gpt_path='voices/mika-e15.ckpt', sovits_path='voices/mika_e8_s160.pth', url='', update_flag=False)
        add_voice_info(name='yuuka', gpt_path='voices/yuuka-e15.ckpt', sovits_path='voices/yuuka_e8_s112.pth', url='', update_flag=False)
        add_voice_info(name='noa', gpt_path='voices/noa-e15.ckpt', sovits_path='voices/noa_e8_s192.pth', url='', update_flag=False)
        add_voice_info(name='koyuki', gpt_path='voices/koyuki-e15.ckpt', sovits_path='voices/koyuki_e8_s128.pth', url='', update_flag=False)
        add_voice_info(name='nagisa', gpt_path='voices/nagisa-e15.ckpt', sovits_path='voices/nagisa_e8_s136.pth', url='', update_flag=False)
        add_voice_info(name='mari', gpt_path='voices/mari-e15.ckpt', sovits_path='voices/mari_e8_s128.pth', url='', update_flag=False)
        add_voice_info(name='kisaki', gpt_path='voices/kisaki-e15.ckpt', sovits_path='voices/kisaki_e8_s128.pth', url='', update_flag=False)
        add_voice_info(name='miyako', gpt_path='voices/miyako-e15.ckpt', sovits_path='voices/miyako_e8_s192.pth', url='', update_flag=False)
        add_voice_info(name='ui', gpt_path='voices/ui-e15.ckpt', sovits_path='voices/ui_e8_s136.pth', url='', update_flag=False)
        add_voice_info(name='seia', gpt_path='voices/seia-e15.ckpt', sovits_path='voices/seia_e8_s216.pth', url='', update_flag=False)
        

    def init_prompts_info():
        add_prompt_info(name='arona', emotion='normal', language='ja', wav_path='./voices/arona/193.wav', text='それでは、先生。良い結果になるよう、応援していますね。頑張ってください', update_flag=False)
        add_prompt_info(name='arona', emotion='surprise', language='ja', wav_path='./voices/arona/056.wav', text='先生！このシーケンスを使って早く脱出してください！', update_flag=False)
        add_prompt_info(name='prana', emotion='normal', language='ja', wav_path='./voices/prana/prana.wav', text='混乱。理解できない行動です。つつかないで下さい。故障します。', update_flag=False)
        add_prompt_info(name='mika', emotion='normal', language='ja', wav_path='./voices/mika/CH0069_Lobby_4.wav', text='何か手伝おっか？任せてもらえたら、頑張るね！', update_flag=False)
        add_prompt_info(name='yuuka', emotion='normal', language='ja', wav_path='./voices/yuuka/CH0184_Formation_Select.wav', text='どうですか?活動的かつ合理的でしょ?', update_flag=False)
        # add_prompt_info(name='yuuka', emotion='normal', language='ja', wav_path='./voices/yuuka/CH0184_Relationship_Up_4.wav', text='先生と一緒にいると私の計算能力が鈍ってしまう時があります', update_flag=True)
        add_prompt_info(name='noa', emotion='normal', language='ja', wav_path='./voices/noa/noa.wav', text='さすがです、先生。勉強になりました。', update_flag=False)
        add_prompt_info(name='koyuki', emotion='normal', language='ja', wav_path='./voices/koyuki/CH0198_LogIn_1.wav', text='お帰りなさい、先生！ 待ちくたびれましたよ～！', update_flag=False)
        add_prompt_info(name='koyuki', emotion='question', language='ja', wav_path='./voices/koyuki/CH0198_Lobby_2.wav', text='計算ですか? 楽勝ですよ！好きじゃないだけで。', update_flag=False)
        add_prompt_info(name='nagisa', emotion='normal', language='ja', wav_path='./voices/nagisa/Nagisa_Lobby_5_2.wav', text='よろしければ先生も、シャーレに新しい椅子を買ってみては？', update_flag=False)
        add_prompt_info(name='nagisa', emotion='calm', language='ja', wav_path='./voices/nagisa/Nagisa_Relationship_Up_4.wav', text='先生のお心遣いに感謝を。トリニティの生徒一同を代表し、お礼申し上げます。', update_flag=False)  # 점잖고 숨쉬는 내용이 많음.
        add_prompt_info(name='mari', emotion='normal', language='ja', wav_path='./voices/mari/176.wav', text='お誕生日おめでとうございます、先生。何か欲しいものはありますか?', update_flag=False)  
        add_prompt_info(name='mari', emotion='calm', language='ja', wav_path='./voices/mari/24.wav', text='おかえりなさい、先生。今日もお会いできて嬉しいです。', update_flag=False)  
        add_prompt_info(name='kisaki', emotion='normal', language='ja', wav_path='./voices/kisaki/Kisaki_EventMission_Get_1.wav', text='ああ、そなたがやり遂げたのだから、遠慮せず持っていくがよい。', update_flag=False)  
        add_prompt_info(name='miyako', emotion='normal', language='ja', wav_path='./voices/miyako/59.wav', text='どんな命令でも完璧に遂行してみせます。', update_flag=False)  
        add_prompt_info(name='ui', emotion='normal', language='ja', wav_path='./voices/ui/034.wav', text='せ、先生…これは…とういう…', update_flag=True)  
        add_prompt_info(name='ui', emotion='normal2', language='ja', wav_path='./voices/ui/029.wav', text='いらっしゃいましたか、先生。', update_flag=True)  
        add_prompt_info(name='seia', emotion='normal', language='ja', wav_path='./voices/seia/seia_cardgame_act_3.wav', text='先生が熱を上げているおもちゃだったか。', update_flag=True)  


    init_voices_info()
    init_prompts_info()
    
    # print(get_prompt_info_from_name(name='arona'))
    # print(get_prompt_info_from_name(name='arona', emotion='angry'))
    # print(get_prompt_info_from_name(name='arona', emotion='error'))  # normal을 잘 가져옴
