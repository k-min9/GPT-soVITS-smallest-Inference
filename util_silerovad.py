'''
pip install silero-vad
'''
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio, collect_chunks

SAMPLING_RATE = 16000
model = None

# audio_path에 vad로 인정된 음성길이(초)
def get_trim_silence_len(audio_path):
    global model
    if not model:
        model = load_silero_vad()
    
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    # print(speech_timestamps)
    
    time = 0
    for timestamp in speech_timestamps:
        time += (timestamp['end'] - timestamp['start'])
    return time/SAMPLING_RATE
    
# 예시용 실제로는 안쓰임
def get_trim_silence_wav(audio_path):
    global model
    if not model:
        model = load_silero_vad()
    
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    
    output_path = './output_vad.wav'
    save_audio(output_path, collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
    
    return output_path

if __name__ == "__main__":
    # model = load_silero_vad(onnx=False)
    
    time = get_trim_silence_len('./output.wav')
    print(time)