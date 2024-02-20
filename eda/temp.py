# import numpy as np
# from keras.preprocessing.image import load_img, img_to_array

# # 이미지를 불러옵니다
# target = load_img('/hs/HeartSignal/data/v1/val/40840_AV_1_label_filtered.png', color_mode='grayscale')
# target = img_to_array(target)
# target = target.astype("uint8")

# # 고유한 색상 값 출력
# unique_colors = np.unique(target)
# print("Unique colors:", unique_colors)

# # 이미지 사이즈 확인
# height, width, _ = target.shape
# print("Image size:", height, "x", width)

# # 이미지의 높이와 너비만 출력하려면
# print("Height:", height)
# print("Width:", width)
##################################################################################
from PIL import Image
import os

def check_image_sizes(folder_path):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(('.png'))]
    sizes_dict = {}

    for file in image_files:
        try:
        # with 문을 사용하여 이미지 파일 열기
            with Image.open(os.path.join(folder_path, file)) as img:
                if img.size in sizes_dict:
                    sizes_dict[img.size] += 1
                else:
                    sizes_dict[img.size] = 1
        except Exception as e:
            print(f"Error opening file {file}: {e}")            

    # 크기별 이미지 수 출력
    for size, count in sizes_dict.items():
        print(f"크기 {size}의 이미지 수: {count}")

    # 모든 이미지가 동일한 크기인지 확인
    if len(sizes_dict) == 1:
        print("모든 이미지가 동일한 크기입니다.")
    else:
        print(f"다른 크기의 이미지가 {len(sizes_dict)}개 있습니다.")

folder_path_test = '/hs/HeartSignal/data/v1/test'
folder_path_val = '/hs/HeartSignal/data/v1/val'
folder_path_train = '/hs/HeartSignal/data/v1/train'
check_image_sizes(folder_path_test)
check_image_sizes(folder_path_val)
check_image_sizes(folder_path_train)
##################################################################################
# from PIL import Image
# import os

# def resize_images_in_folder(folder_path, target_size=(256, 256)):
#     # 폴더 내의 모든 파일 가져오기
#     files = os.listdir(folder_path)

#     # 이미지 파일 목록을 저장할 리스트
#     image_files = [file for file in files if file.lower().endswith(('.png'))]

#     # 이미지 크기 조정 및 저장
#     for file in image_files:
#         file_path = os.path.join(folder_path, file)
#         img = Image.open(file_path)
#         # 최신 Pillow 버전에서는 Image.ANTIALIAS 대신 Image.Resampling.LANCZOS 사용
#         resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # 고화질로 리사이즈
#         resized_img.save(file_path)  # 원본 파일 덮어쓰기

#     print(f"모든 이미지가 {target_size} 크기로 변경되었습니다.")

# # 함수 사용 예시
# folder_path = '/hs/HeartSignal/data/v1/train'  # 변경할 폴더 경로
# resize_images_in_folder(folder_path)
##################################################################################