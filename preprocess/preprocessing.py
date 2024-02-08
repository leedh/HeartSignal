import os
import librosa
import torch
import torchaudio

# load custom modules
from preprocess.denoising import *
from preprocess.labels import *
from utils import *

def preprocess_audio_and_label(wav_name, data_dir, target_sr, lowpass_cutoff_freq, hop_length, win_length, n_fft, n_mels):
    # 작업할 wav 파일 데이터 로드
    wav_path = os.path.join(data_dir, wav_name)
    waveform, orig_sr = librosa.load(wav_path, sr=None) 
    
    # 1. denosing
    #waveform = denoise_audio(waveform, wavelet='db1', level=1)
    # 신호처리 이론에 따라 denoising을 위한 여러가지 방법을 적용할 수 있다.
    # TODO: denosing (1): wavelet transform
    # TODO: denosing (2): thresholding
    # TODO: denosing (3): wavelet inverse transform
    
    # 2. resampling
    waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    # interpolation에 대한 이론을 익혀서 resampling을 적용할 수 있다.
    #waveform = librosa.resample(waveform, orig_sr, target_sr, res_type='kaiser_fast')
    
    # 3. filtering
    waveform = torch.from_numpy(waveform)
    for _ in range(5):
        waveform = torchaudio.functional.lowpass_biquad(waveform, target_sr, cutoff_freq=lowpass_cutoff_freq, Q=0.707)
    waveform = np.array(waveform.numpy())
    # TODO: 몇번 필터링을 적용할지 결정
    # TODO: 필터링 (1): low pass filter
    # TODO: 필터링 (2): high pass filter
    # TODO: 필터링 (3): bandpass filter
    # TODO: 필터링 (4): notch filter
    # TODO: 필터링 (5): bandstop filter
    
    # 4. Z-score normalization
    waveform = librosa.util.normalize(waveform)
    
    # mel spectrogram 생성
    mel = create_mel_spectrogram_raw(waveform, target_sr, hop_length, win_length, n_fft, n_mels)
    
    # tsv 레이블 데이터에서 시간 정보를 가져오기.
    boundaries_time = calculate_boundaries_time(wav_path[:-3] + 'tsv', target_sr, hop_length)
    
    # for "50782_MV_1.tsv"
    # if len(boundaries_time) == 0:
    #     S_dB_list.append(0)
    #     continue

    # 레이블 있는 부분만 크롭
    cropped_start, cropped_end = calculate_cropped_indices(boundaries_time)
    
    # 결과값
    c_mel = get_cropped_mel(mel, cropped_start, cropped_end)
    adjusted_boundaries = adjust_boundaries(boundaries_time, cropped_start)

    return c_mel, adjusted_boundaries