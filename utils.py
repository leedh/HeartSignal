#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import io
import random
import numpy as np

import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, lfilter, iirnotch
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, save_img

from tqdm import tqdm

# load custom modules
from preprocess.labels import *

# load the files from dirPath
def load_files(dirPath):
    files = list()
    file_list = os.listdir(dirPath)
    for f in tqdm(file_list, desc='Loading files'):
        if f.endswith('.wav'):
            files.append(f)
    return files

# apply bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = lfilter(b,a, data)
    return filtered_data

# apply notch filter (a.k.a. bandstop filter)
def notch_pass_filter(data, center, interval=20, sr=4000, normalized=False):
    center = center / (sr/2) if normalized else center
    b, a = iirnotch(center, interval/interval, sr)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def get_preprocessed_mel_spectrogram(wav_name, data_dir,sr=4000,n_mels=96,frame_length=0.025,frame_stride=0.01):
    # 변수 정의
    file_name = wav_name[:-4]
    wav_path = os.path.join(data_dir, wav_name)

    # 작업할 wav 파일 데이터 로드
    y, sr = librosa.load(wav_path, sr=None)
    mel = get_mel_spectrogram(y, sr, n_mels, frame_length, frame_stride)

    # 원본 멜 스펙트로그램 위에 tsv 레이블 데이터 표시
    boundaries_time = calculate_boundaries_time(wav_path[:-3] + 'tsv', sr, frame_stride)
    
    # for "50782_MV_1.tsv"
    # if len(boundaries_time) == 0:
    #     S_dB_list.append(0)
    #     continue

    # 레이블 있는 부분만 크롭
    cropped_start, cropped_end = calculate_cropped_indices(boundaries_time)
    c_mel = get_cropped_mel(mel, cropped_start, cropped_end)
    adjusted_boundaries = adjust_boundaries(boundaries_time, cropped_start)

    # 전체 길이 조정 (2.5초)
    new_mel, new_b = process_mel_and_boundaries(c_mel, adjusted_boundaries, sr, frame_stride, 2.5)
    seconds_cropped_mel = crop_mel_by_sec(c_mel, sr, frame_stride, 2.5)

    # squared mel spectrogram
    S_dB, _ = draw_mel_square_spec(seconds_cropped_mel, sr, frame_stride)
    
    return S_dB

# get segmentation labels
def get_label(wav_name, data_dir,sr=4000,n_mels=96,frame_length=0.025,frame_stride=0.01):

    # 변수 정의
    file_name = wav_name[:-4]
    wav_path = os.path.join(data_dir, wav_name)

    # 작업할 wav 파일 데이터 로드
    y, sr = librosa.load(wav_path, sr=None)
    mel = get_mel_spectrogram(y, sr, n_mels, frame_length, frame_stride)

    # 원본 멜 스펙트로그램 위에 tsv 레이블 데이터 표시
    boundaries_time = calculate_boundaries_time(wav_path[:-3] + 'tsv', sr, frame_stride)
    
    # for "50782_MV_1.tsv"
    # if len(boundaries_time) == 0:
    #     S_dB_list.append(0)
    #     continue

    # 레이블 있는 부분만 크롭
    cropped_start, cropped_end = calculate_cropped_indices(boundaries_time)
    c_mel = get_cropped_mel(mel, cropped_start, cropped_end)
    adjusted_boundaries = adjust_boundaries(boundaries_time, cropped_start)

    # 전체 길이 조정 (2.5초)
    new_mel, new_b = process_mel_and_boundaries(c_mel, adjusted_boundaries, sr, frame_stride, 2.5)
    seconds_cropped_mel = crop_mel_by_sec(c_mel, sr, frame_stride, 2.5)

    # squared mel spectrogram
    S_dB = draw_mel_square_spec_with_boundaries_to_label_and_filter(seconds_cropped_mel, new_b, -33, sr, frame_stride)
    
    return S_dB

def figure_mel_spec(S_dB, save_path, sr, hop_length):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=95) # set the size of the figure
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, vmin=-80, vmax=0, cmap='inferno')
    ax.set_aspect('auto') # set aspect ratio to be equal
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # close the figure
    
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
