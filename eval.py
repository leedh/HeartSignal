

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import keras_cv
from config import data_config, train_config, id2color
from tensorflow.keras.applications import ResNet50V2
import cv2
print ("import has done successfully")


############## HeartSoundAnalyzer 클래스 정의 ############################################


class HeartSoundAnalyzer:
    def __init__(self, model_path):
        # ResNet50V2 백본 모델을 초기화합니다.
        self.backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
            preset=train_config.MODEL,
            input_shape=data_config.IMAGE_SIZE + (3,),
            load_weights=True
        )
        # DeepLabV3Plus 모델을 초기화합니다.
        self.model = keras_cv.models.segmentation.DeepLabV3Plus(
            num_classes=data_config.NUM_CLASSES,
            backbone=self.backbone  # Corrected to use self.backbone
        )
        # 사전 훈련된 가중치를 로드합니다.
        self.model.load_weights(model_path)
        self.input_size = data_config.IMAGE_SIZE  # 모델이 기대하는 입력 크기

    def overlay_mask_on_image(self, image, mask):
        # 원본 이미지의 크기를 얻습니다.
        height, width, _ = image.shape
        # 마스크의 크기를 원본 이미지와 일치하도록 조정합니다.
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # 컬러맵을 적용하여 색상 마스크를 생성합니다.
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])  # 클래스 수에 맞게 조정
        mask_colored = colormap[resized_mask].astype(np.uint8)
        # 오버레이를 적용합니다.
        overlayed_image = ((1 - 0.5) * image + 0.5 * mask_colored).astype(np.uint8)
        return overlayed_image

    def inference_and_save_results(self, image_path, save_dir):
        # 이미지 로딩 및 전처리
        image_data = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        image_resized = tf.image.resize(image, self.input_size, method="nearest")
        image_resized = tf.cast(image_resized, tf.float32)
        image_batch = tf.expand_dims(image_resized, axis=0)

        # 모델을 사용한 추론
        pred_mask = self.model.predict(image_batch)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[0, ...].numpy()  # 배치 차원 제거

        # 결과 저장
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 원본 이미지 저장
        original_image_path = os.path.join(save_dir, "original_image.png")
        plt.imsave(original_image_path, image.numpy())

        # 예측 마스크 저장
        predicted_mask_path = os.path.join(save_dir, "predicted_mask.png")
        plt.imsave(predicted_mask_path, pred_mask, cmap='gray')

        # 오버레이된 이미지 저장
        overlayed_image = self.overlay_mask_on_image(image.numpy(), pred_mask)  # 이미지 정규화 생략
        overlayed_image_path = os.path.join(save_dir, "overlayed_prediction.png")
        plt.imsave(overlayed_image_path, overlayed_image, vmin=0, vmax=255)  # [0, 255] 범위 지정

        return original_image_path, predicted_mask_path, overlayed_image_path

    def find_and_count_columns(self, image_path):
        # 이미지 로드 및 배열로 변환
        image = load_img(image_path, color_mode='grayscale')
        image = img_to_array(image)
        image = image.astype("uint8")
        print(np.unique(image))

        # 이미지 크기
        height, width = image.shape[:2]

        # 이미지의 가장 아래 행 인덱스
        y = height - 1

        # 각 기둥의 시작과 끝 위치를 저장할 리스트 및 카운트
        white_columns = []
        gray_columns = []
        white_columns_count = 0
        gray_columns_count = 0

        in_white_column = False
        in_gray_column = False
        white_start = 0
        gray_start = 0

        for x in range(width):
            column_color = image[y, x, 0]

            # 흰색 기둥 처리
            if column_color == 255:
                if not in_white_column:
                    in_white_column = True
                    white_start = x
                elif x == width - 1 and white_start != x:
                    # 픽셀 길이가 3 초과인 경우만 처리
                    if x - white_start > 3:
                        white_columns.append((white_start, x))
                        white_columns_count += 1
            elif in_white_column:
                in_white_column = False
                # 픽셀 길이가 3 초과인 경우만 처리
                if white_start != x - 1 and x - 1 - white_start > 3:
                    white_columns.append((white_start, x - 1))
                    white_columns_count += 1

            # 회색 기둥 처리
            if column_color == 128:
                if not in_gray_column:
                    in_gray_column = True
                    gray_start = x
                elif x == width - 1 and gray_start != x:
                    # 픽셀 길이가 3 초과인 경우만 처리
                    if x - gray_start > 3:
                        gray_columns.append((gray_start, x))
                        gray_columns_count += 1
            elif in_gray_column:
                in_gray_column = False
                # 픽셀 길이가 3 초과인 경우만 처리
                if gray_start != x - 1 and x - 1 - gray_start > 3:
                    gray_columns.append((gray_start, x - 1))
                    gray_columns_count += 1

        file_name = os.path.basename(image_path)

        return white_columns, gray_columns, white_columns_count, gray_columns_count, file_name

    def evaluate_single_image(self, image_path):
        original_image_path, predicted_mask_path, overlayed_image_path = self.inference_and_save_results(image_path, save_dir)
        # find_and_count_columns 함수에 predicted_mask_path를 전달합니다.
        white_columns, gray_columns, white_columns_count, gray_columns_count, file_name = self.find_and_count_columns(predicted_mask_path)

        # 예측 결과를 처리하고 출력
        print(f"S1 positions: {white_columns}")
        print(f"S2 positions: {gray_columns}")
        print(f"Total S1 sounds: {white_columns_count}, Total S2 sounds: {gray_columns_count}")
        print(f'File name: {file_name}')

if __name__ == "__main__":
    model_path = "/hs/HeartSignal/models/deeplabv3_plus_resnet50_v2.h5"
    image_path = "/hs/HeartSignal/models/input_dir/O2.png"
    save_dir = "/hs/HeartSignal/models/save_dir"
    analyzer = HeartSoundAnalyzer(model_path)
    analyzer.evaluate_single_image(image_path)