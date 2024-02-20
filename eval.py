
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import keras_cv
from nets.segnet import segnet
from nets.unet_pp import unet_pp_model
from config import data_config, train_config, id2color
from tensorflow.keras.applications import ResNet50V2
import cv2
import matplotlib.image as mpimg

print ("import has done successfully")

############## HeartSoundAnalyzer 클래스 정의 ############################################

class HeartSoundAnalyzer:
    def __init__(self, model_path):
        self.model = unet_pp_model(img_size=(256, 256), num_classes=data_config.NUM_CLASSES)
        # self.model = segnet(img_size=(256, 256), num_classes=data_config.NUM_CLASSES)
        self.model.load_weights(model_path)
        self.input_size = data_config.IMAGE_SIZE

    def overlay_mask_on_image(self, image, mask):
        height, width, _ = image.shape
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])
        mask_colored = colormap[resized_mask].astype(np.uint8)
        overlayed_image = ((1 - 0.5) * image + 0.5 * mask_colored).astype(np.uint8)
        return overlayed_image

    def fill_below_found_color(self, image_cv):
        height, width = image_cv.shape[:2]

        for x in range(width):
            for y in range(height):
                pixel_color = image_cv[y, x]
                if pixel_color == 1 or pixel_color == 2:  # 수정된 조건
                    image_cv[y:, x] = pixel_color  # 해당 위치부터 아래까지 같은 색으로 채움
                    break  # 해당 열에서 특정 클래스의 픽셀을 찾으면 나머지 행은 무시

        return image_cv  # 수정된 이미지 반환


    def inference_and_save_results(self, image_path, save_dir):
        image_data = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        image_resized = tf.image.resize(image, self.input_size, method="nearest")
        image_resized = tf.cast(image_resized, tf.float32) / 255
        image_batch = tf.expand_dims(image_resized, axis=0)

        pred_mask = self.model.predict(image_batch)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[0, ...].numpy()

        filled_mask = self.fill_below_found_color(pred_mask)  # 이미 pred_mask가 클래스 인덱스로 구성되어 있음

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        original_image_path = os.path.join(save_dir, "original_image.png")
        plt.imsave(original_image_path, image.numpy())

        predicted_mask_path = os.path.join(save_dir, "predicted_mask.png")
        plt.imsave(predicted_mask_path, filled_mask, cmap='gray')

        overlayed_image = self.overlay_mask_on_image(image.numpy(), filled_mask)
        overlayed_image_path = os.path.join(save_dir, "overlayed_prediction.png")
        plt.imsave(overlayed_image_path, overlayed_image)

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
                    # 픽셀 길이가 5 초과인 경우만 처리
                    if x - white_start > 5:
                        white_columns.append((white_start, x))
                        white_columns_count += 1
            elif in_white_column:
                in_white_column = False
                # 픽셀 길이가 5 초과인 경우만 처리
                if white_start != x - 1 and x - 1 - white_start > 5:
                    white_columns.append((white_start, x - 1))
                    white_columns_count += 1

            # 회색 기둥 처리
            if column_color == 128:
                if not in_gray_column:
                    in_gray_column = True
                    gray_start = x
                elif x == width - 1 and gray_start != x:
                    # 픽셀 길이가 5 초과인 경우만 처리
                    if x - gray_start > 5:
                        gray_columns.append((gray_start, x))
                        gray_columns_count += 1
            elif in_gray_column:
                in_gray_column = False
                # 픽셀 길이가 15 초과인 경우만 처리
                if gray_start != x - 1 and x - 1 - gray_start > 5:
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
    model_path = "/hs/HeartSignal/models/checkpoints/unet_pp.h5"
    image_path = "/hs/HeartSignal/models/input_dir/J1.png"
    save_dir = "/hs/HeartSignal/models/save_dir"
    analyzer = HeartSoundAnalyzer(model_path)
    analyzer.evaluate_single_image(image_path)

#################### 훈련 그래프 ####################

def plot_training_history(history, epochs_range, save_dir):
    """
    훈련 및 검증 손실과 정확도 그래프를 그리고 저장합니다.

    Parameters:
    - history: 훈련 과정의 기록을 담은 history 객체.
    - epochs_range: 에포크 범위.
    - save_dir: 그래프 이미지를 저장할 디렉토리 경로.
    """
    # 손실 그래프
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history.history["loss"], "b", label="Training loss")
    plt.plot(epochs_range, history.history["val_loss"], "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(save_dir, "training_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.show()

    # 정확도 그래프
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history.history["accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], "r", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    accuracy_plot_path = os.path.join(save_dir, "training_validation_accuracy.png")
    plt.savefig(accuracy_plot_path)
    plt.show()

# 사용 예
# plot_training_history(history, range(1, train_config.EPOCHS + 1), "/path/to/save/directory")