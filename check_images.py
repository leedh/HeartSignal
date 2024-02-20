

#TODO: 레이블 이미지가 손상되었는지 안되었는지, 입력 이미지가 손상되었는지 확인하는 코드 넣기

import os
import tensorflow as tf
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import unpackage_inputs, image_overlay
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def check_images_and_labels(directory):
    input_img_files = sorted([f for f in os.listdir(directory) if f.endswith('_img.png')])
    target_img_files = sorted([f for f in os.listdir(directory) if f.endswith('_label.png')])

    print(f"Directory: {directory}")
    print(f"입력 이미지 파일 개수: {len(input_img_files)}")
    print(f"타겟 이미지 파일 개수: {len(target_img_files)}")

    all_files = input_img_files + target_img_files
    if not all_files:
        print("이미지 파일이 없습니다.")
        return

    size_dict = defaultdict(int)

    # 모든 이미지들의 크기 확인 및 기록
    for file in all_files:
        img_path = os.path.join(directory, file)
        img = Image.open(img_path)
        size_dict[img.size] += 1

    # 이미지 크기의 종류가 1개보다 많으면, 다양한 크기 정보 출력
    if len(size_dict) > 1:
        print("다양한 크기의 이미지가 존재합니다:")
        for size, count in size_dict.items():
            print(f"크기: {size}, 개수: {count}")
    else:
        print("모든 이미지가 동일한 크기입니다:", next(iter(size_dict)))

    print("")

def analyze_label_colors(target_dir, num_samples):
    # _label.png로 끝나는 파일만 선택
    target_img_files = sorted([f for f in os.listdir(target_dir) if f.endswith('_label.png')])
    
    # 파일이 요청한 샘플 수보다 적은 경우, 모든 파일 사용
    if len(target_img_files) < num_samples:
        selected_files = target_img_files
    else:
        # 랜덤하게 num_samples 개수만큼 파일 선택
        selected_files = random.sample(target_img_files, num_samples)
    
    overall_color_counts = {}

    for img_file in selected_files:
        target_path = os.path.join(target_dir, img_file)
        target = load_img(target_path, color_mode='grayscale')
        target = img_to_array(target)
        target = target.astype("uint8")
        
        unique_colors, counts = np.unique(target, return_counts=True)
        
        for color, count in zip(unique_colors, counts):
            if color in overall_color_counts:
                overall_color_counts[color] += count
            else:
                overall_color_counts[color] = count
    
    # 전체 색상맵 값 중에서 가장 많이 나타난 색상맵 값을 정렬하여 추출
    sorted_colors = sorted(overall_color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 결과 출력
    print(f"Top 5 most common colors in {target_dir}:")
    for color, count in sorted_colors:
        print(f"Color: {color}, Count: {count}")

def analyze_colors_in_dataset(data, n_images):
    """
    데이터셋에서 랜덤하게 n_images만큼의 이미지와 마스크를 분석하여 색상 분포를 출력합니다.
    
    Parameters:
    - data: TensorFlow 데이터셋 객체. 이미지와 마스크를 포함해야 합니다.
    - n_images: 분석할 이미지의 수.
    """

    estimated_size = data.cardinality().numpy()
    if estimated_size == tf.data.experimental.AUTOTUNE:
        raise ValueError("Data size is unknown. Please ensure the dataset size is calculable.")

    overall_image_color_counts = {}
    overall_mask_color_counts = {}

    selected_indices = random.sample(range(estimated_size), n_images)

    for index in selected_indices:
        image, mask = next(iter(data.skip(index).take(1)))

        # 이미지 색상 분포 분석
        image = image.numpy()
        image_colors = image.reshape(-1, image.shape[-1])
        unique_image_colors, counts_image = np.unique(image_colors, axis=0, return_counts=True)
        for color, count in zip(map(tuple, unique_image_colors), counts_image):
            overall_image_color_counts[color] = overall_image_color_counts.get(color, 0) + count

        # 마스크 색상 분포 분석
        mask = mask.numpy().squeeze()
        unique_mask_colors, counts_mask = np.unique(mask, return_counts=True)
        for color, count in zip(unique_mask_colors, counts_mask):
            overall_mask_color_counts[color] = overall_mask_color_counts.get(color, 0) + count

    # 가장 많이 나타난 색상값을 정렬하여 추출
    sorted_image_colors = sorted(overall_image_color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_mask_colors = sorted(overall_mask_color_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # 결과 출력
    print("Image color distribution in the dataset:")
    for color, count in sorted_image_colors:
        print(f"Color: {color}, Count: {count}")

    print("\nMask color distribution in the dataset:")
    for color, count in sorted_mask_colors:
        print(f"Color: {color}, Count: {count}")

if __name__ == "__main__":
    # 상황에 따라 로컬 디렉토리 경로 설정
    base_path = "/hs/HeartSignal/data/filtered"
    # base_path = "/hs/HeartSignal/data/v1_img"
    directories = ["train", "val", "test"]

    for dir in directories:
        full_path = os.path.join(base_path, dir)
        print(f"Checking images and labels in {dir}...")
        check_images_and_labels(full_path)
        print(f"Analyzing label colors in {dir}...")
        analyze_label_colors(full_path, num_samples=10)
        print('-----------------------------------------')