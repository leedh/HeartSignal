#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import numpy as np
import librosa

# load custom modules
from webapp import utils

class DataLoader:
    def __init__(self, data_dir, resample=8000, lowpass_cutoff_freq=500, n_mels=96):

        # 파라미터 설정
        self.data_dir = data_dir
        self.sample_rate = resample
        self.n_mels = n_mels
        self.frame_stride = 0.01
        self.frame_length = 0.025
        self.win_length = int(round(self.sample_rate * self.frame_length))
        self.hop_length = int(round(self.sample_rate * self.frame_stride))
        self.lowpass_cutoff_freq = lowpass_cutoff_freq
        self.n_fft = 2 ** int(np.ceil(np.log2(self.win_length))) # Calculate the number of FFT components (n_fft) as the next power of two from win_length
        
        # .wav 파일
        print("Loading wav files ...")
        self.wav_files = utils.load_files(self.data_dir)

        # 인스턴스 정보 출력
        print("Total number of audio files:", len(self.wav_files))
        
    def __getitem__(self, index): # index값에 따라 데이터를 반환하는 함수
        return self.wav_files[index]

    def __len__(self): # 데이터 로더의 전체 길이를 반환하는 함수
        return len(self.wav_files)
    
    def audio_file_loader(self, file_name): # .wav 파일을 로드하는 함수
        print("Loading wav files ...")
        audio_path = os.path.join(self.data_dir, file_name)
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def mel_spectrogram_loader(self, file_name): # .wav 파일을 mel spectrogram으로 변환하는 함수
        # get mel spectrogram and adjusted boundaries
        c_mel = utils.preprocess_audio_and_label(file_name, self.data_dir, self.sample_rate, self.lowpass_cutoff_freq, self.hop_length, self.win_length, self.n_fft, self.n_mels)
        
        # blank region clipping     
        mel_prep =utils.blank_region_clipping(c_mel, self.n_mels)
        
        return mel_prep