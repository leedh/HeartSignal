

#TODO: 레이블 이미지가 손상되었는지 안되었는지, 입력 이미지가 손상되었는지 확인하는 코드 넣기

import os
import tensorflow as tf
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from utils import unpackage_inputs
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

def save_batch_images_with_masks(dataset, save_dir="/hs/HeartSignal/eda/check_train_ds", batch_size=3):
    """
    TensorFlow 데이터셋에서 배치를 추출하여 이미지와 마스크를 오버레이한 뒤 파일로 저장합니다.

    Parameters:
    - dataset: TensorFlow 데이터셋 객체. 이미지와 마스크를 포함해야 합니다.
    - save_dir: 저장할 디렉토리 경로.
    - batch_size: 저장할 배치 크기.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 데이터셋에서 배치를 추출
    plot_dataset = dataset.map(unpackage_inputs).batch(batch_size)
    image_batch, mask_batch = next(iter(plot_dataset.take(1)))

    for i, (image, mask) in enumerate(zip(image_batch, mask_batch)):
        # 마스크의 채널 차원 제거 및 오버레이 준비
        mask = tf.squeeze(mask, axis=-1).numpy()

        # 오버레이된 이미지 준비
        overlayed_image = np.copy(image.numpy().astype(np.uint8))
        overlayed_image[mask == 1] = [255, 0, 0]  # 예시: 레이블 1에 대해 빨간색 오버레이 적용

        # 오버레이된 이미지 저장
        plt.imsave(os.path.join(save_dir, f"overlayed_image_{i}.png"), overlayed_image)   

if __name__ == "__main__":
    # 상황에 따라 로컬 디렉토리 경로 설정
    base_path = "/hs/HeartSignal/data/filtered"
    # base_path = "/hs/HeartSignal/data/v1"
    directories = ["train", "val", "test"]

    for dir in directories:
        full_path = os.path.join(base_path, dir)
        print(f"Checking images and labels in {dir}...")
        check_images_and_labels(full_path)
        print(f"Analyzing label colors in {dir}...")
        analyze_label_colors(full_path, num_samples=10)
        print('-----------------------------------------')