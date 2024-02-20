import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import os
import random

# 이미지를 시각화하고 파일로 저장하는 함수
def save_visualizations(input_dir, target_dir, input_files, target_files, save_dir, num_images):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 랜덤으로 이미지 선택하기 위해 인덱스 생성 후 섞기
    indices = list(range(min(len(input_files), len(target_files))))
    random.shuffle(indices)
    selected_indices = indices[:num_images]

    for j, idx in enumerate(selected_indices):
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))  # figsize를 조정하여 이미지 크기 증가
        
        input_path = os.path.join(input_dir, input_files[idx])
        target_path = os.path.join(target_dir, target_files[idx])
        
        axs[0].axis("off")
        axs[0].imshow(load_img(input_path))
        
        axs[1].axis("off")
        axs[1].imshow(load_img(target_path))
        
        # 시각화된 이미지를 파일로 저장
        save_path = os.path.join(save_dir, f"visualization_{j+1}.png")
        plt.savefig(save_path)
        plt.close()  # 현재 플롯 닫기

if __name__ == "__main__":
    # 로컬 디렉토리 경로 설정
    input_dir = "/hs/HeartSignal/data/v1/train"
    target_dir = "/hs/HeartSignal/data/v1/train"
    save_dir = "/hs/HeartSignal/eda/check_img_label"
    
    # 이미지 파일 목록 생성
    input_img_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_img.png')])
    target_img_files = sorted([f for f in os.listdir(target_dir) if f.endswith('_label.png')])
    
    # 시각화 함수 호출
    save_visualizations(input_dir, target_dir, input_img_files, target_img_files, save_dir, 10)
