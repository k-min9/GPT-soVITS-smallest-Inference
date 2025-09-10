'''
음성 화자 분석 및 비교 유틸리티 / pyannote-audio를 사용한 speaker verification
pip install pyannote-audio librosa soundfile pandas scipy
'''
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import torch
import librosa
import soundfile as sf
import os
import voice_management

from kei import HF_TOKEN

HF_TOKEN = HF_TOKEN
embedding_model = None
audio = None
current_model_type = None
current_device = None

# 사용 가능한 모델들
AVAILABLE_MODELS = {
    'ecapa': "speechbrain/spkrec-ecapa-voxceleb",  # 기존 모델
    'xvector': "speechbrain/spkrec-xvect-voxceleb",  # X-Vector (더 나은 성능)
    'resnet': "speechbrain/spkrec-resnet34-voxceleb",  # ResNet 기반 (더 나은 성능)
    'wavlm': "microsoft/wavlm-base-plus-sv"  # WavLM 기반 (최신, 가장 좋은 성능)
}

# Speaker embedding 모델 초기화
def init_speaker_model(model_type='ecapa', use_gpu=False):
    global embedding_model, audio, current_model_type, current_device
    
    # 동일한 설정이면 재초기화하지 않음
    if (embedding_model is not None and 
        current_model_type == model_type and 
        current_device == ('cuda' if use_gpu else 'cpu')):
        return
    
    # 디바이스 설정
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    device_name = "GPU" if device.type == 'cuda' else "CPU"
    
    # 모델 선택
    if model_type not in AVAILABLE_MODELS:
        print(f"Warning: Unknown model type '{model_type}', using 'ecapa' as fallback")
        model_type = 'ecapa'
    
    model_path = AVAILABLE_MODELS[model_type]
    model_name = model_type.upper()
    
    print(f"Loading {model_name} model on {device_name}...")
    
    try:
        if model_type == 'wavlm':
            # WavLM 모델은 다른 방식으로 로드할 수 있음
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            try:
                embedding_model = PretrainedSpeakerEmbedding(
                    model_path,
                    use_auth_token=HF_TOKEN
                )
            except:
                print(f"Failed to load WavLM model, falling back to X-Vector...")
                model_path = AVAILABLE_MODELS['xvector']
                model_name = "X-Vector"
                embedding_model = PretrainedSpeakerEmbedding(
                    model_path,
                    use_auth_token=HF_TOKEN
                )
        else:
            embedding_model = PretrainedSpeakerEmbedding(
                model_path,
                use_auth_token=HF_TOKEN
            )
        
        # GPU로 이동
        if use_gpu and torch.cuda.is_available():
            embedding_model = embedding_model.to(device)
            print(f"Model moved to GPU (CUDA)")
        
        audio = Audio(sample_rate=16000)
        
        current_model_type = model_type
        current_device = device.type
        
        print(f"{model_name} model loaded successfully on {device_name}")
        
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        if model_type != 'ecapa':
            print("Falling back to ECAPA-TDNN model...")
            return init_speaker_model('ecapa', use_gpu)
        else:
            raise e

# 음성 파일로부터 임베딩 추출
def extract_embedding(file_path, model_type='ecapa', use_gpu=False):
    init_speaker_model(model_type, use_gpu)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # 1차 시도: librosa 사용
        duration = librosa.get_duration(filename=file_path)
        segment = Segment(0, duration)
        waveform, sample_rate = audio.crop(file_path, segment)
        
        # mono 채널로 강제 변환 (pyannote는 1채널만 지원)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # GPU 사용시 텐서를 GPU로 이동
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        waveform = waveform.to(device)
        
        with torch.no_grad():
            return embedding_model(waveform[None])
            
    except Exception as e:
        print(f"Error with librosa for {file_path}: {type(e).__name__} - {e}")
        print("Trying with soundfile as fallback...")
        
        try:
            # 2차 시도: soundfile 사용
            data, sample_rate = sf.read(file_path)
            
            # 스테레오를 모노로 변환 (필요시)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # 샘플레이트 변환 (16kHz로 맞추기)
            if sample_rate != 16000:
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # torch tensor로 변환
            waveform = torch.tensor(data).unsqueeze(0).float()
            
            # mono 채널로 강제 변환 (pyannote는 1채널만 지원)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # GPU 사용시 텐서를 GPU로 이동
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            waveform = waveform.to(device)
            
            with torch.no_grad():
                return embedding_model(waveform)
                
        except Exception as e2:
            print(f"Error with soundfile fallback for {file_path}:")
            print(f"  Error type: {type(e2).__name__}")
            print(f"  Error message: {e2}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return None

# 두 임베딩 간의 cosine similarity 계산
def calculate_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    try:
        emb1 = embedding1.squeeze()
        emb2 = embedding2.squeeze()
        similarity = 1 - cosine(emb1, emb2)
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

# 두 음성 파일의 유사도 비교
def compare_audio_files(audio_path1, audio_path2, model_type='ecapa', use_gpu=False):
    embedding1 = extract_embedding(audio_path1, model_type, use_gpu)
    embedding2 = extract_embedding(audio_path2, model_type, use_gpu)
    
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    return calculate_similarity(embedding1, embedding2)

# 입력 음성이 특정 캐릭터의 음성과 일치하는지 확인
def identify_speaker(input_audio_path, character_name, emotion='normal', threshold=0.6, model_type='ecapa', use_gpu=False):
    # 캐릭터의 기준 음성 정보 가져오기
    prompt_info = voice_management.get_prompt_info_from_name(character_name, emotion)
    
    if prompt_info is None:
        return {
            'is_match': False,
            'similarity': 0.0,
            'character': character_name,
            'reference_path': None,
            'error': f'No reference audio found for character: {character_name}'
        }
    
    reference_path = prompt_info.get('wav_path')
    if not reference_path or not os.path.exists(reference_path):
        return {
            'is_match': False,
            'similarity': 0.0,
            'character': character_name,
            'reference_path': reference_path,
            'error': f'Reference audio file not found: {reference_path}'
        }
    
    # 음성 비교
    similarity = compare_audio_files(input_audio_path, reference_path, model_type, use_gpu)
    is_match = (similarity >= threshold)
    
    return {
        'is_match': is_match,
        'similarity': similarity,
        'character': character_name,
        'reference_path': reference_path,
        'threshold': threshold,
        'model_type': model_type,
        'device': 'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'
    }

# 입력 음성을 여러 캐릭터와 비교하여 가장 유사한 캐릭터 찾기
def identify_speaker_from_multiple_characters(input_audio_path, character_list=None, threshold=0.6, model_type='ecapa', use_gpu=False):
    if character_list is None:
        # voice_management에서 모든 캐릭터 이름 가져오기
        try:
            character_list = voice_management.get_voice_name()
        except Exception as e:
            return {
                'best_match': None,
                'best_similarity': 0.0,
                'all_results': [],
                'is_match': False,
                'error': f'Failed to get character list: {e}'
            }
    
    all_results = []
    best_match = None
    best_similarity = 0.0
    
    for character in character_list:
        result = identify_speaker(input_audio_path, character, threshold=threshold, model_type=model_type, use_gpu=use_gpu)
        all_results.append({
            'character': character,
            'similarity': result['similarity'],
            'is_match': result['is_match']
        })
        
        if result['similarity'] > best_similarity:
            best_similarity = result['similarity']
            best_match = character
    
    return {
        'best_match': best_match,
        'best_similarity': best_similarity,
        'all_results': sorted(all_results, key=lambda x: x['similarity'], reverse=True),
        'is_match': best_similarity >= threshold,
        'threshold': threshold,
        'model_type': model_type,
        'device': 'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'
    }

# 여러 음성 파일들 간의 유사도 행렬 계산
def get_speaker_similarity_matrix(audio_paths, model_type='ecapa', use_gpu=False):
    print("🔄 Extracting embeddings...")
    embeddings = {}
    
    for path in audio_paths:
        if os.path.exists(path):
            embeddings[path] = extract_embedding(path, model_type, use_gpu)
        else:
            print(f"Warning: File not found - {path}")
            embeddings[path] = None
    
    print("📊 Calculating similarities...")
    similarity_matrix = []
    
    for path1 in audio_paths:
        row = []
        for path2 in audio_paths:
            if embeddings[path1] is None or embeddings[path2] is None:
                similarity = 0.0
            else:
                similarity = calculate_similarity(embeddings[path1], embeddings[path2])
            row.append(similarity)
        similarity_matrix.append(row)
    
    return similarity_matrix

# 모델 전역 변수 초기화
def reset_model_globals():
    global embedding_model, current_model_type, current_device
    embedding_model = None
    current_model_type = None
    current_device = None

# 사용 가능한 모델 정보 출력
def print_available_models():
    print("🎯 Available Speaker Verification Models:")
    print("   ecapa  : ECAPA-TDNN (기본 모델, 안정적)")
    print("   xvector: X-Vector (더 나은 성능)")  
    print("   resnet : ResNet-34 (더 나은 성능)")
    print("   wavlm  : WavLM (최신, 가장 좋은 성능)")
    print()
    print("📊 Performance comparison (일반적):")
    print("   ECAPA-TDNN < X-Vector ≈ ResNet < WavLM")
    print("   CPU 속도  : ResNet > ECAPA > X-Vector > WavLM")
    print("   GPU 속도  : 모든 모델 유사")
    print()

if __name__ == "__main__":
    import glob
    
    # ========== 모델 및 GPU 설정 ==========
    # 사용할 모델 선택: 'ecapa', 'xvector', 'resnet', 'wavlm'
    MODEL_TYPE = 'ecapa'  # 기본값: ecapa (기존 모델)
    # MODEL_TYPE = 'xvector'  # X-Vector (더 나은 성능)
    # MODEL_TYPE = 'resnet'   # ResNet 기반 (더 나은 성능)
    MODEL_TYPE = 'wavlm'    # WavLM 기반 (최신, 가장 좋은 성능)
    
    # GPU 사용 여부
    USE_GPU = True  # True: GPU 사용, False: CPU 사용
    
    # 사용할 설정 출력
    device_name = "GPU" if USE_GPU and torch.cuda.is_available() else "CPU"
    print(f"🚀 Configuration:")
    print(f"   Model: {MODEL_TYPE.upper()}")
    print(f"   Device: {device_name}")
    if USE_GPU and not torch.cuda.is_available():
        print("   Warning: GPU requested but not available, using CPU")
    print(f"   Available models: {list(AVAILABLE_MODELS.keys())}")
    print()

    # 테스트 폴더의 wav 파일들 가져오기
    test_folder = "./test/voice_speech_diarization"
    wav_files = glob.glob(os.path.join(test_folder, "*.wav"))
    
    if not wav_files:
        print("No test files available.")
        exit()
    
    print(f"Found {len(wav_files)} test files:")
    for i, file_path in enumerate(wav_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    print()
    
    # 각 wav 파일에 대해 테스트 수행
    for test_file in wav_files:
        file_name = os.path.basename(test_file)
        print(f"{'='*50}")
        print(f"Testing: {file_name}")
        print(f"{'='*50}")
        
        # 1. 특정 캐릭터들과 개별 비교 테스트
        # FILE_PATH = './voices/info.json'
        test_characters = []  # 테스트할 캐릭터 목록
        test_characters.append('arona')
        test_characters.append('plana')
        test_characters.append('mika')
        test_characters.append('yuuka')
        test_characters.append('noa')
        test_characters.append('koyuki')
        test_characters.append('nagisa')
        test_characters.append('mari')
        test_characters.append('kisaki')
        test_characters.append('miyako')
        test_characters.append('ui')
        test_characters.append('seia')
        # test_characters.append('prana')

        print(f"[{file_name}] === 특정 캐릭터들과 개별 비교 테스트 ===")
        
        for char in test_characters:
            try:
                result = identify_speaker(test_file, char, model_type=MODEL_TYPE, use_gpu=USE_GPU)
                reference_file = os.path.basename(result.get('reference_path', '')) if result.get('reference_path') else 'N/A'
                match_indicator = "✓" if result['is_match'] else "✗"
                device_used = result.get('device', 'N/A')
                
                print(f"  {char}: {result['similarity']:.4f} {match_indicator} (ref: {reference_file}) [{device_used}]")
                
                if 'error' in result:
                    print(f"    Error: {result['error']}")
            except Exception as e:
                print(f"  {char}: Error - {e}")
        print()
        
    # ========== 모델 성능 비교 테스트 ==========
    if False:
        print("=" * 60)
        print("🚀 MODEL PERFORMANCE COMPARISON TEST")
        print("=" * 60)
        
        import time
        
        # 테스트할 모든 조합
        test_configs = [
            ('ecapa', False),   # ECAPA + CPU
            ('ecapa', True),    # ECAPA + GPU
            ('xvector', False), # X-Vector + CPU
            ('xvector', True),  # X-Vector + GPU
            ('resnet', False),  # ResNet + CPU
            ('resnet', True),   # ResNet + GPU
            ('wavlm', False),   # WavLM + CPU
            ('wavlm', True),    # WavLM + GPU
        ]
        
        # 테스트용 파일 선택 (첫 번째 파일 사용)
        if wav_files:
            test_file = wav_files[0]
            test_char = 'arona'  # 테스트 캐릭터
            
            results = []
            
            print(f"Testing with file: {os.path.basename(test_file)}")
            print(f"Testing character: {test_char}")
            print()
            
            for model_type, use_gpu in test_configs:
                device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
                print(f"Testing {model_type.upper()} on {device_name}...")
                try:
                    # 기존 모델 제거를 위한 함수 호출
                    reset_model_globals()
                    
                    # 모델 로딩 시간 측정
                    start_time = time.time()
                    
                    # 모델 로딩
                    init_speaker_model(model_type, use_gpu)
                    model_load_time = time.time() - start_time
                    
                    # 단일 추론 시간 측정 (Warm-up)
                    start_time = time.time()
                    result = identify_speaker(test_file, test_char, model_type=model_type, use_gpu=use_gpu)
                    warmup_time = time.time() - start_time
                    
                    # 실제 추론 시간 측정 (5회 평균)
                    inference_times = []
                    for _ in range(5):
                        start_time = time.time()
                        result = identify_speaker(test_file, test_char, model_type=model_type, use_gpu=use_gpu)
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                    
                    avg_inference_time = sum(inference_times) / len(inference_times)
                    similarity = result['similarity']
                    
                    results.append({
                        'model': model_type.upper(),
                        'device': device_name,
                        'model_load_time': model_load_time,
                        'warmup_time': warmup_time,
                        'avg_inference_time': avg_inference_time,
                        'similarity': similarity,
                        'success': True
                    })
                    
                    print(f"  ✓ Model Load: {model_load_time:.3f}s")
                    print(f"  ✓ Warm-up: {warmup_time:.3f}s") 
                    print(f"  ✓ Avg Inference: {avg_inference_time:.3f}s")
                    print(f"  ✓ Similarity: {similarity:.4f}")
                    print()
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    results.append({
                        'model': model_type.upper(),
                        'device': device_name,
                        'model_load_time': 0,
                        'warmup_time': 0,
                        'avg_inference_time': 0,
                        'similarity': 0,
                        'success': False,
                        'error': str(e)
                    })
                    print()
            
            # 결과 요약 테이블
            print("=" * 80)
            print("📊 PERFORMANCE SUMMARY")
            print("=" * 80)
            print(f"{'Model':<10} {'Device':<6} {'Load(s)':<8} {'Warmup(s)':<10} {'Inference(s)':<12} {'Similarity':<10} {'Status':<10}")
            print("-" * 80)
            
            for result in results:
                if result['success']:
                    status = "✓ OK"
                    print(f"{result['model']:<10} {result['device']:<6} {result['model_load_time']:<8.3f} "
                        f"{result['warmup_time']:<10.3f} {result['avg_inference_time']:<12.3f} "
                        f"{result['similarity']:<10.4f} {status:<10}")
                else:
                    status = "❌ FAIL"
                    print(f"{result['model']:<10} {result['device']:<6} {'N/A':<8} {'N/A':<10} {'N/A':<12} {'N/A':<10} {status:<10}")
            
            print("-" * 80)
            
            # 성공한 결과만으로 순위 매기기
            successful_results = [r for r in results if r['success']]
            
            if successful_results:
                print("\n🏆 RANKINGS:")
                
                # 로딩 시간 순위
                print("\n⚡ Fastest Model Loading:")
                sorted_by_load = sorted(successful_results, key=lambda x: x['model_load_time'])
                for i, result in enumerate(sorted_by_load[:3], 1):
                    print(f"  {i}. {result['model']} ({result['device']}): {result['model_load_time']:.3f}s")
                
                # 추론 시간 순위
                print("\n🚀 Fastest Inference:")
                sorted_by_inference = sorted(successful_results, key=lambda x: x['avg_inference_time'])
                for i, result in enumerate(sorted_by_inference[:3], 1):
                    print(f"  {i}. {result['model']} ({result['device']}): {result['avg_inference_time']:.3f}s")
                
                # 유사도 순위 (높은 순)
                print("\n🎯 Highest Similarity:")
                sorted_by_similarity = sorted(successful_results, key=lambda x: x['similarity'], reverse=True)
                for i, result in enumerate(sorted_by_similarity[:3], 1):
                    print(f"  {i}. {result['model']} ({result['device']}): {result['similarity']:.4f}")
                
                # GPU vs CPU 비교
                print("\n💡 GPU vs CPU Analysis:")
                gpu_results = [r for r in successful_results if r['device'] == 'GPU']
                cpu_results = [r for r in successful_results if r['device'] == 'CPU']
                
                if gpu_results and cpu_results:
                    avg_gpu_inference = sum(r['avg_inference_time'] for r in gpu_results) / len(gpu_results)
                    avg_cpu_inference = sum(r['avg_inference_time'] for r in cpu_results) / len(cpu_results)
                    speedup = avg_cpu_inference / avg_gpu_inference if avg_gpu_inference > 0 else 0
                    
                    print(f"  Average GPU inference: {avg_gpu_inference:.3f}s")
                    print(f"  Average CPU inference: {avg_cpu_inference:.3f}s")
                    print(f"  GPU Speedup: {speedup:.2f}x")
                
            print("\n" + "=" * 60)
        
    print("\nTest completed.")
