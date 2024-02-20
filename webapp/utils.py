import os
import numpy as np
import librosa
import torch
import torchaudio
import cv2
import cmapy

def load_files(dirPath):
    files = list()
    file_list = os.listdir(dirPath)
    for f in file_list:
        if f.endswith('.wav'):
            files.append(f)
    return files

def create_mel_spectrogram_raw(waveform, target_sr, hop_length, win_length, n_fft, n_mels, resz=0):
    S = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
    # Convert Mel spectrogram to Decibel scale
    S = librosa.power_to_db(S, ref=np.max) # why np.max?: https://dsp.stackexchange.com/questions/64509/why-should-one-choose-the-maximum-as-reference-for-power-to-decibel-conversion
    # normalization
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width*resz, height*resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img


# 멜 스펙트로그램의 시작 및 종료 시점 기준 크롭
def get_cropped_mel(mel, n_frames=250):
    # Ensure the cropping indices are within the bounds of the Mel spectrogram
    median = mel.shape[1] // 2
    
    cropped_start = median - n_frames // 2
    cropped_end = median + n_frames // 2

    # Crop and return the Mel spectrogram
    return mel[:, cropped_start:cropped_end]

def preprocess_audio_and_label(wav_name, data_dir, target_sr, lowpass_cutoff_freq, hop_length, win_length, n_fft, n_mels):
    # 작업할 wav 파일 데이터 로드
    wav_path = os.path.join(data_dir, wav_name)
    waveform, orig_sr = librosa.load(wav_path, sr=None) 
    
    # 1. denosing
    #waveform = denoise_audio(waveform, wavelet='db1', level=1)    
    # 2. resampling
    waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    
    # 3. filtering
    waveform = torch.from_numpy(waveform)
    for _ in range(5):
        waveform = torchaudio.functional.lowpass_biquad(waveform, target_sr, cutoff_freq=lowpass_cutoff_freq, Q=0.707)
    waveform = np.array(waveform.numpy())
    
    # 4. Z-score normalization
    waveform = librosa.util.normalize(waveform)
    
    # mel spectrogram 생성
    mel = create_mel_spectrogram_raw(waveform, target_sr, hop_length, win_length, n_fft, n_mels)
    
    # 결과값
    c_mel = get_cropped_mel(mel, n_frames=250)
    

    return c_mel


# Blank Region Clipping (BRC) from RespireNet
def blank_region_clipping(mel, n_mels):
    
    mel_color = cv2.cvtColor(mel, cv2.COLOR_BGR2RGB) # BRG to RGB
    mel_gray = cv2.cvtColor(mel, cv2.COLOR_BGR2GRAY) # BRG to Gray

    mel_gray[mel_gray < 10] = 0
    for row in range(mel_gray.shape[0]):
        # 위쪽 row부터 블랙 픽셀(0) 비율 계산. 80% 이상이면 break
        black_percent = len(np.where(mel_gray[row,:]==0)[0])/len(mel_gray[row,:])
        if black_percent < 0.80:
            break

    if (row+1) < mel_color.shape[0]:
        mel_color = mel_color[(row+1):, :, :]
    mel_color = cv2.resize(mel_color, (mel_color.shape[1], n_mels), interpolation=cv2.INTER_LINEAR)

    return mel_color