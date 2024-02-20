#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import wget
import zipfile
import fnmatch

import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

# custom modules
from utils import figure_mel_spec

def download_dataset(basedir):
    # Download PhysioNet dataset
    url = "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
    wget.download(url, out=basedir)

    # Unzip the downloaded file
    with zipfile.ZipFile(basedir + "/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip", 'r') as zip_ref:
        zip_ref.extractall(path=basedir)

    # Delete the zip file
    os.remove(basedir + "/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
    
    # Detete a missing file
    # "50782_MV_1.tsv" has no information about time boundaries. So, we will remove it from the dataset.
    datdir = basedir + "/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/"
    for filename in os.listdir(datdir):
        if fnmatch.fnmatch(filename, '50782_MV_1.*'):
            os.remove(os.path.join(datdir, filename))

def dataset_split(basedir, preprocess_name, data):
    filename = os.path.join(basedir, 'the-circor-digiscope-phonocardiogram-dataset-1.0.3', 'training_data.csv')
    description = pd.read_csv(filename, usecols=['Patient ID', 'Sex'])  # 필요한 열만 로드

    # 남성과 여성 데이터를 분리 및 분할
    train, temp = train_test_split(description, test_size=0.2, stratify=description['Sex'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['Sex'], random_state=42)

    # 훈련/검증/테스트 데이터셋별 인덱스 저장
    filename_list = [data.wav_files[i][:-4] for i in range(data.__len__())]  # 확장자 제거
    pattern = re.compile(r'(\d+)')

    train_ids = set(train['Patient ID'])
    val_ids = set(val['Patient ID'])
    test_ids = set(test['Patient ID'])

    train_group_idx, val_group_idx, test_group_idx = [], [], []

    for idx, num in tqdm(enumerate(filename_list), total=len(filename_list)):
        key = int(pattern.search(num).group(1))
        if key in train_ids:
            train_group_idx.append(idx)
        elif key in val_ids:
            val_group_idx.append(idx)
        elif key in test_ids:
            test_group_idx.append(idx)

    # 환자 기준으로 데이터셋 분리 저장
    def save_data(group_idx, folder):
        splitdir = os.path.join(basedir, preprocess_name, folder)
        hop_length = int(round(data.sample_rate * data.frame_stride))
        
        print(f"Saving {folder} data in {splitdir}...")
        for idx in tqdm(group_idx):
            audio_path = os.path.join(data.data_dir,data.wav_files[idx])
            mel_spectrogram, n = data.mel_spectrogram_loader(audio_path)
            label = data.label_loader(audio_path)
            # split한 데이터를 list에서 추출해 따로 저장
            for i in range(n):
                filename_n = f"{filename_list[idx]}_{i+1}"
                mel_spectrogram_i = mel_spectrogram[i]
                label_i = label[i]
                
                # Save img and label figures
                for suffix, content in zip(['img', 'label'], [mel_spectrogram_i, label_i]):
                    save_path = f"{splitdir}/{filename_n}_{suffix}.png"
                    if not os.path.exists(save_path):
                        #figure_mel_spec(content, suffix, save_path, data.sample_rate, hop_length)
                        figure_mel_spec(content, suffix, save_path)
            del audio_path, mel_spectrogram, label
            gc.collect() # garbege collection
    # Save data
    #save_data(train_group_idx, 'train')
    #save_data(val_group_idx, 'val')
    save_data(test_group_idx, 'test')

    print("Data successfully saved.")