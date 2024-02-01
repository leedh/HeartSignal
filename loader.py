#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import numpy as np
import glob

import tensorflow as tf 
# from tensorflow.keras.utils import Sequence
import librosa

from tqdm import tqdm

# load custom modules
from utils import *

class DataLoader:
    def __init__(self, data_dir,sample_rate=4000,n_mels=96, n_fft=1024,hop_length=512,frame_stride=0.01,frame_length=0.025):

        # 파라미터 설정
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_stride = frame_stride
        self.frame_length = frame_length
        
        # .wav 파일 로드
        print("Loading wav files ...")
        self.wav_files = load_files(self.data_dir)

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
        mel_spectrogram = get_preprocessed_mel_spectrogram(file_name, self.data_dir, self.sample_rate, self.n_mels, self.frame_length, self.frame_stride)
        return mel_spectrogram
    
    def label_loader(self, file_name): # .wav 파일에 대한 세그멘테이션 라벨을 반환하는 함수
        label = get_label(file_name, self.data_dir, self.sample_rate, self.n_mels, self.frame_length, self.frame_stride)
        return label

class ImageLoader:
    def __init__(self, data_dir, folder, img_height, img_width, batch_size):
        self.data_dir = data_dir
        self.folder = folder
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def parse_image(self, img_path: tf.Tensor, normalize=True):
        # 이미지 파일을 불러와서 사이즈 조절 및 정규화
        img = tf.io.read_file(os.path.join(img_path, self.folder)) # 이미지 파일 로드
        img = tf.image.decode_png(img, channels=3) # 이미지 디코드
        img = tf.image.resize(img, [self.img_height, self.img_width]) # 이미지 크기 조정
        img = tf.image.convert_image_dtype(img, "float32")
        if normalize: # 이미지 정규화 (선택적)
            img = img / 255.0
        return img

    def parse_label(self, label_path: tf.Tensor):
        # 레이블 이미지 파일을 불러와서 사이즈 조절
        label = tf.io.read_file(os.path.join(label_path, self.folder)) # 이미지 파일 로드
        label = tf.image.decode_png(label, channels=1) # 이미지 디코드
        label = tf.image.resize(label, [self.img_height, self.img_width]) # 이미지 크기 조정
        label = tf.image.convert_image_dtype(label, "uint8") # 이미지 데이터 타입 변환
        return label

    def load_dataset(self):
        # 이미지와 레이블 파일 경로를 가져옵니다.
        img_paths = glob.glob(os.path.join(self.data_dir, '*_img.png'))
        label_paths = [path.replace('_img.png', '_label.png') for path in img_paths]

        # TensorFlow Dataset 객체를 생성합니다.
        img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        img_dataset = img_dataset.map(self.parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        label_dataset = tf.data.Dataset.from_tensor_slices(label_paths)
        label_dataset = label_dataset.map(self.parse_label, num_parallel_calls=tf.data.AUTOTUNE)

        # 이미지와 레이블 데이터셋을 결합합니다.
        dataset = tf.data.Dataset.zip((img_dataset, label_dataset))

        # 배치 처리, 셔플, 프리패치를 설정합니다.
        dataset = dataset.batch(self.batch_size).shuffle(buffer_size=100).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset