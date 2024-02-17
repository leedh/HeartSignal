import os
from PIL import Image
from collections import defaultdict
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

def filter_and_save_images(input_dir, output_dir, keep_colors):
    """
    특정 색상을 유지하며 레이블 이미지를 필터링하고 지정된 다른 폴더에 저장하는 함수.

    Parameters:
    - input_dir: 입력 이미지 파일이 있는 디렉토리 경로.
    - output_dir: 출력 이미지 파일을 저장할 디렉토리 경로.
    - keep_colors: 유지할 색상 값의 리스트.
    """
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리에서 _label.png로 끝나는 이미지 파일 목록을 가져옴
    img_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('_label.png')]

    # 이미지 파일들을 순회하며 처리
    for file in img_files:
        # 이미지 불러오기
        img_path = os.path.join(input_dir, file)
        img = load_img(img_path, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = img_array.astype("uint8")

        # 유지할 색상이 아닌 값들을 0으로 설정
        mask = np.isin(img_array, keep_colors)
        filtered_img = np.where(mask, img_array, 0)

        # 채널 차원을 유지하도록 조정
        if filtered_img.ndim == 2:
            filtered_img = np.expand_dims(filtered_img, axis=-1)

        # 처리된 이미지를 다른 이름으로 저장
        filtered_img_file = 'filtered_' + os.path.splitext(file)[0] + '.png'
        filtered_img_path = os.path.join(output_dir, filtered_img_file)
        save_img(filtered_img_path, filtered_img)

if __name__ == "__main__":
    # 로컬 디렉토리 경로 설정
    base_path = "/hs/HeartSignal/data/v1"
    output_base_path = "/hs/HeartSignal/data/filtered"  # 출력 디렉토리 기본 경로 설정
    directories = ["train", "val", "test"]
    keep_colors = [0, 127, 255]  # 유지할 색상 값

    for dir in directories:
        input_dir = os.path.join(base_path, dir)
        output_dir = os.path.join(output_base_path, dir)
        print(f"Processing {dir} directory...")
        filter_and_save_images(input_dir, output_dir, keep_colors)