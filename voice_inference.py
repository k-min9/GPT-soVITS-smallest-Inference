# 변수 정리
version = 'v2'

import voice_management
# import state

import logging
import traceback
from datetime import datetime

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
import LangSegment
import os, re, sys, json
import torch

import soundfile as sf

gpt_path = 'GPT_weights_v2/arona-e5.ckpt'
sovits_path = 'SoVITS_weights_v2/arona_e4_s148.pth'
cnhubert_base_path = './pretrained_models/chinese-hubert-base'
bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
# is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()  # dtype=np.float16 if is_half == True else np.float32,
is_half = True
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio

# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
device = 'cuda'

# LRU Cache Configuration
MAX_CACHED_ACTORS = 10

# Logging Configuration
LOG_FILE_PATH = None
VRAM_AVAILABLE = False

def init_logging():
    """로그 시스템 초기화"""
    global LOG_FILE_PATH, VRAM_AVAILABLE
    
    # VRAM 측정 기능 확인
    try:
        import pynvml
        pynvml.nvmlInit()
        VRAM_AVAILABLE = True
        print("[LOG] VRAM monitoring enabled")
    except:
        VRAM_AVAILABLE = False
        print("[LOG] VRAM monitoring not available")
    
    # 로그 파일 생성
    os.makedirs('./log', exist_ok=True)
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    LOG_FILE_PATH = os.path.join('./log', log_filename)
    print(f"[LOG] Log file created: {LOG_FILE_PATH}")

def get_vram_info():
    """VRAM 사용량 조회"""
    if not VRAM_AVAILABLE:
        return "N/A"
    
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / (1024 ** 3)
        total_gb = info.total / (1024 ** 3)
        return f"{used_gb:.1f}GB/{total_gb:.1f}GB"
    except:
        return "N/A"

def log_cache_access(actor, action=None):
    """캐시 접근 로깅"""
    if LOG_FILE_PATH is None:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 현재 캐시된 actor 목록 (최근 사용 순서)
    cached_actors = vq_models.keys()  # t2s_models와 동일하게 관리됨
    actors_str = ",".join(cached_actors)
    
    # VRAM 정보
    vram_info = get_vram_info()
    
    # ACTION 결정
    if action is None:
        action = f"{actor.upper()} CALLED"
    
    # 로그 기록
    log_line = f"[{timestamp}] {action} | actors=[{actors_str}] | VRAM={vram_info}\n"
    
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(log_line)
    except Exception as e:
        print(f"[LOG ERROR] Failed to write log: {e}")

def clear_all_cache():
    """모든 캐시 제거"""
    vq_models.clear()
    t2s_models.clear()
    log_cache_access("", "CLEAR ALL")
    print("[CACHE] All models cleared")

def remain_n_actors(n):
    """최근 사용된 N개 actor만 남기고 나머지 제거"""
    vq_models.remove_oldest_n(n)
    t2s_models.remove_oldest_n(n)
    log_cache_access("", f"REMAIN {n}")
    print(f"[CACHE] Kept only {n} most recent actors")

def preload_actor(actor):
    """특정 actor 모델을 미리 로딩"""
    if actor not in vq_models:
        voice_info = voice_management.get_voice_info_from_name(actor)
        sovits_path = voice_info['sovits_path']
        change_sovits_weights(actor, sovits_path)
    
    if actor not in t2s_models:
        voice_info = voice_management.get_voice_info_from_name(actor)
        gpt_path = voice_info['gpt_path']
        change_gpt_weights(actor, gpt_path)
    
    # 캐시 hit - 최근 사용으로 업데이트
    vq_models.get(actor)
    t2s_models.get(actor)
    
    log_cache_access(actor, f"{actor.upper()} LOADED")
    print(f"[CACHE] Actor {actor} preloaded")

class LRUModelCache:
    """LRU(Least Recently Used) 기반 모델 캐시 관리 클래스"""
    def __init__(self, max_size=10, cache_type="unknown"):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_type = cache_type  # 로깅용
    
    def get(self, key):
        """캐시에서 모델 조회. 조회 시 최근 사용으로 이동"""
        if key not in self.cache:
            return None
        # 최근 사용 순서로 이동
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, model):
        """캐시에 모델 추가. max_size 초과 시 가장 오래된 모델 제거"""
        # 이미 존재하면 업데이트하고 최신으로 이동
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = model
            return
        
        # max_size 초과 시 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
            oldest_key, oldest_model = self.cache.popitem(last=False)
            print(f"[LRU] Evicting {self.cache_type} model: {oldest_key}")
            del oldest_model
            torch.cuda.empty_cache()
        
        # 새 모델 추가
        self.cache[key] = model
    
    def keys(self):
        """현재 캐시된 키 목록 반환 (최근 사용 순서)"""
        return list(self.cache.keys())
    
    def clear(self):
        """캐시의 모든 모델 제거"""
        for key, model in self.cache.items():
            del model
        self.cache.clear()
        torch.cuda.empty_cache()
    
    def remove_oldest_n(self, keep_count):
        """최근 사용된 keep_count개만 남기고 나머지 제거"""
        current_count = len(self.cache)
        if current_count <= keep_count:
            return
        
        remove_count = current_count - keep_count
        for _ in range(remove_count):
            oldest_key, oldest_model = self.cache.popitem(last=False)
            print(f"[LRU] Removing {self.cache_type} model: {oldest_key}")
            del oldest_model
        
        torch.cuda.empty_cache()
    
    def __contains__(self, key):
        """'in' 연산자 지원"""
        return key in self.cache

vq_models = LRUModelCache(max_size=MAX_CACHED_ACTORS, cache_type="SoVITS")
t2s_models = LRUModelCache(max_size=MAX_CACHED_ACTORS, cache_type="GPT")

# Lazy Loading: BERT 모델은 중국어 사용 시에만 로드
tokenizer = None
bert_model = None

def load_bert_model():
    """중국어 처리를 위한 BERT 모델을 Lazy Loading"""
    global tokenizer, bert_model
    
    if bert_model is not None:
        return  # 이미 로드됨
    
    print("🔄 Loading BERT model for Chinese language support...")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    
    print("✅ BERT model loaded successfully!")


def get_bert_feature(text, word2ph):
    """중국어 BERT feature 추출 - Lazy Loading 적용"""
    # 중국어 처리를 위해 BERT 모델 로드 (최초 1회만)
    load_bert_model()
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(actor, sovits_path,prompt_language=None,text_language=None):
    global hps, version
    
    # 캐시에서 모델 조회
    cached_model = vq_models.get(actor)
    if cached_model is not None:
        return cached_model
    
    device = 'cuda'
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del model.enc_q
    if is_half == True:
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    print(model.load_state_dict(dict_s2["weight"], strict=False))
    
    # 캐시에 저장
    vq_models.put(actor, model)
    return model

def change_gpt_weights(actor, gpt_path):
    global hz, max_sec, config
    
    # 캐시에서 모델 조회
    cached_model = t2s_models.get(actor)
    if cached_model is not None:
        return cached_model
    
    device = 'cuda'
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        model = model.half()
    model = model.to(device)
    model.eval()
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    
    # 캐시에 저장
    t2s_models.put(actor, model)
    return model

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    """언어별 BERT feature 생성 - 중국어만 실제 BERT 사용, 나머지는 zero tensor"""
    language=language.replace("all_","")
    if language == "zh":
        # 중국어: BERT 모델 사용 (Lazy Loading)
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        # 일본어/한국어/영어: zero tensor 사용 (VRAM 절약)
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

# from text import chinese
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                # formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                # formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        # print('get_phones_and_bert textlist', textlist)
        # print('get_phones_and_bert langlist', langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


tts_idx = 0  # 전송중 파일 생성으로 인한 충돌 방지용
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k=15, top_p=1, temperature=1, ref_free =False, speed=1, if_freeze=False,inp_refs=None, actor='arona'):
    global tts_idx
    
    # 캐릭터가 없는 경우 모델 로딩
    if actor not in vq_models:
        voice_info = voice_management.get_voice_info_from_name(actor)     
        sovits_path = voice_info['sovits_path']
        change_sovits_weights(actor, sovits_path)
    if actor not in t2s_models:
        voice_info = voice_management.get_voice_info_from_name(actor)
        gpt_path = voice_info['gpt_path']
        change_gpt_weights(actor, gpt_path)
    
    # 캐시 hit - 최근 사용으로 업데이트
    vq_model = vq_models.get(actor)
    t2s_model = t2s_models.get(actor)
    
    # 로그 기록
    log_cache_access(actor)

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    # print('get_tts_wav is_half', is_half)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                print(wav16k.shape[0], "### Cutline : 48000~160000")
                raise OSError('### prompt 오디오를 3~10초 범위로 세팅')
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)
        
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

    audio_opt = []
    # print('get_tts_wav ref_free', ref_free)
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    # print('get_tts_wav texts', texts)
    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): 
            if text_language != "en":
                text += "。"  
            else:
                text += "."

        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        
        if not ref_free:  # 기본 false
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
 
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(  # GPT weights
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,  # 1500
                # early_stop_num=600,
            )
            # print('#####', pred_semantic)
            if idx==0:
                return 'early stop'
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        refers=[]
        if(inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path.name).to(dtype).to(device)
                    refers.append(refer)
                except:
                    traceback.print_exc()
        if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]

        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
        max_audio=np.abs(audio).max()  # 16비트 폭음 간단 방지
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

    tts_idx = (tts_idx+1)%10
    result = 'output'+str(tts_idx)+'.wav'
    

    if os.path.exists('./files_server/'):  # files에도 필요한 경우가 있음(API 전송용)
        sf.write("./files_server/"+result,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16), 32000)
    else:
        sf.write("./"+result,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16), 32000)
    
    # Test용 파일저장
    # if state.get_DEV_MODE():
    if True:
        try:
            voice_file_name = "voice_" + str(datetime.now().strftime("%y%m%d_%H%M%S")) + "_" + text + ".wav"
            voice_audio_path = os.path.join('./test/voice', voice_file_name)  # 충돌방지용
            os.makedirs('./test/voice', exist_ok=True) 
            sf.write(voice_audio_path,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16), 32000)
        except:
            print('fail saving get_tts_wav')
    
    # print('get_tts_wav end')
    return result

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError('###process_text valueerror')
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def synthesize_char(char_name, audio_text, audio_language='ja', speed=1):
    global gpt_path, sovits_path
    
    prompt_info = voice_management.get_prompt_info_from_name(char_name)  # Todo : 없을때의 Try Catch
    print(prompt_info)
    
    prompt_language = prompt_info['language'] # 'ja'
    ref_wav_path = prompt_info['wav_path'] #'./voices/noa.wav'
    prompt_text = prompt_info['text'] # 'さすがです、先生。勉強になりました。'

    result = get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor=char_name, speed=speed)
    return result


# Longtensor를 json으로 저장
def save_longtensor_to_json(tensor: torch.LongTensor, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tensor.tolist(), f, ensure_ascii=False)

# JSON 파일에서 LongTensor를 불러옴
def load_longtensor_from_json(path: str) -> torch.LongTensor:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return torch.LongTensor(data)

def save_phones_and_bert(ref_wav_path, prompt_text, prompt_language, actor='arona'):
    is_half = False
    
    # 캐시에서 모델 조회
    vq_model = vq_models.get(actor)
    if vq_model is None:
        raise ValueError(f"Model for actor {actor} not loaded")
    
    prompt_text = prompt_text.strip("\n")
    if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
    
    zero_wav = np.zeros(
        int(32000 * 0.3), # int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError('###chk421')
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)
        print('prompt', prompt)

    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
    print('get_tts_wav phones1', phones1)
    print('get_tts_wav bert1', bert1)
    print('get_tts_wav norm_text1', norm_text1)

if __name__ == '__main__':    
    # TODO : 로컬화할 경우, 영향도 파악 (현재 패키징은 가능)
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    cnhubert_base_path = './pretrained_models/chinese-hubert-base'
    bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'
    _CUDA_VISIBLE_DEVICES = 0
    # is_half = False  # Float32
    is_half = True  # Float32
    
    # audio_text = '안녕? 난 민구라고 해'
    # audio_text = '테스트중! 메리크리스마스!'
    # audio_text = 'API 사용 가능한가요?'
    # audio_text = 'Only English Spokened'
    # audio_text = '오케이!'
    # audio_text = 'python can be spoken'
    # audio_text = 'get some rest sensei! 안녕하세요?'
    # audio_language = 'ko'
    audio_text = '待っておったぞ、先生。'
    audio_text = 'そなたはイタズラが好きなのじゃな。'
    # audio_text = '新しきを知るのは良いことじゃ。そうじゃろ?'
    # audio_text = 'じゃが、ゆるそう。'
    # audio_text = 'ほれ、カボチャとナツメの料理じゃ。そなたと一緒に食べたいと思ってな。'
    audio_text = '右クリックでメニューを開き、設定を変更することができます。'
    # audio_text = 'ふぅえ…'
    # audio_text = 'ひえええっ！'
    # audio_text = '先生を信用しているつもりです。'  # miyako idx = 0 오류
    audio_language = 'ja'
    # print('error?')
    
    audio_text = audio_text.replace('AI', 'えいあい')
    audio_text = audio_text.replace('MY-Little-JARVIS-3D', 'マイリトル・ジャービス スリーでぃ')
    audio_text = audio_text.replace('MY-Little-JARVIS', 'マイリトル・ジャービス')
    audio_text = audio_text.replace('Android', 'アンドロイド')
    audio_text = audio_text.replace('Windows', 'ウィンドウズ')
    audio_text = audio_text.replace('方', 'かた')
    audio_text = audio_text.replace('.exe', 'ドット exe')
    
    print(audio_text)
    
    actor = 'mari'

    prompt_info = voice_management.get_prompt_info_from_name(actor)  # Todo : 없을때의 Try Catch
    prompt_language = prompt_info['language'] # 'ja'
    ref_wav_path = prompt_info['wav_path'] #'./voices/noa.wav'
    prompt_text = prompt_info['text'] # 'さすがです、先生。勉強になりました。'
       
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor=actor)
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')

    result = synthesize_char(actor, audio_text, audio_language='ja', speed=1)
    print('save at ' + result)
    
    # save_phones_and_bert(ref_wav_path, prompt_text, prompt_language)
    
    
    # print('end!!')