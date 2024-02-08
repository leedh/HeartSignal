#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import numpy as np

import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
import cv2
import cmapy

from tqdm import tqdm
import gc

# load custom modules
from preprocess.labels import *
from preprocess.biquad import *
from preprocess.denoising import *
from preprocess.blank_region_clipping import *

# time count decorator
def time_calculator(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: <{func.__name__}> execution took {end_time - start_time:.2f} sec")
        return result
    return wrapper

# load the files from dirPath
def load_files(dirPath):
    files = list()
    file_list = os.listdir(dirPath)
    for f in tqdm(file_list, desc='Loading files'):
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

#def figure_mel_spec(mel, type, save_path, sr, hop_length):
def figure_mel_spec(mel, type, save_path):
    colormap = 'magma' if type == 'img' else 'gray'
    # Figure와 Axes 객체를 명시적으로 생성
    fig, ax = plt.subplots(dpi=95)  # DPI 설정을 여기에 포함
    ax.imshow(mel, cmap=colormap)
    ax.axis('off')  # 축 끄기

    # tight_layout 호출로 레이아웃 최적화
    fig.tight_layout(pad=0)  # pad를 0으로 설정하여 여백 최소화

    # 이미지 파일로 저장
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # Figure 닫기 (메모리 해제를 위해)
    plt.close(fig)  # 명시적으로 Figure 객체를 닫음

    # 가비지 컬렉션 실행
    gc.collect()
    
    # plt.figure(dpi=95) # set the size of the figure
    # plt.imshow(mel, cmap=colormap)
    # plt.axis('off') # turn off axis
    # #librosa.display.specshow(mel, sr=sr, hop_length=hop_length, vmin=-80, vmax=0, cmap='inferno')
    # #ax.set_aspect('auto') # set aspect ratio to be equal
    # plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.close()  # close the figure
    
# 저장된 이미지들의 컬러값 분석
def analyze_image_colors(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        pixels = list(img.getdata())

        # caculate the ratio of used colors in the image
        color_count = Counter(pixels)
        total_pixels = len(pixels)
        color_ratios = {color: count / total_pixels for color, count in color_count.items()}

        return color_ratios
        
# resize images
def resize(input_image, input_mask, size=(128, 128)):
    input_image = tf.image.resize(input_image, size, method="nearest")
    input_mask = tf.image.resize(input_mask, size, method="nearest")

    return input_image, input_mask

# data augmentation
def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask

# normalization
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask