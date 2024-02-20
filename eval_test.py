from config import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from nets.unet import unet_model
from nets.unet_pp import unet_pp_model
from nets.segnet import segnet
from utils import *
import random
import numpy as np
import cv2
import keras_cv

def fill_below_found_color(image_path, output_dir):
    # 이미지 로드 및 배열로 변환
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    image = image.astype("uint8")

    # OpenCV로 이미지 변환 (넘파이 배열을 OpenCV 이미지로 변환)
    image_cv = np.uint8(image)

    # 이미지 크기
    height, width = image.shape[:2]

    # 각 열을 순회하며 특정 색상의 픽셀 아래 부분을 같은 색으로 채우기
    for x in range(width):
        for y in range(height):
            pixel_color = image_cv[y, x, 0]
            if pixel_color == 255 or pixel_color == 128:  # 흰색 또는 회색 픽셀 발견
                image_cv[y:, x, 0] = pixel_color  # 해당 위치부터 아래까지 같은 색으로 채움
                break  # 해당 열에서 색상을 찾으면 나머지 행은 무시

    # 원본 파일명을 기반으로 고유한 출력 파일 경로 생성
    base_filename = os.path.basename(image_path)
    output_filename = f"preprocessed_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)

    # 수정된 이미지 저장
    cv2.imwrite(output_path, image_cv)

    return output_path

def process_images(directory, output_directory):
    # 지정된 디렉토리의 .png 파일 목록을 가져오고 정렬
    file_list = sorted([file for file in os.listdir(directory) if file.lower().endswith('.png')])

    processed_files = []
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        processed_image_path = fill_below_found_color(file_path, output_directory)
        processed_files.append(processed_image_path)

    return processed_files

def find_and_count_columns(image_path):
    # 이미지 로드 및 배열로 변환
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    image = image.astype("uint8")

    # OpenCV로 이미지 변환
    image_cv = np.uint8(image)

    # 이미지 크기
    height, width = image.shape[:2]

    # 각 기둥의 시작과 끝 위치를 저장할 리스트 및 카운트
    white_columns = []
    gray_columns = []
    white_columns_count = 0
    gray_columns_count = 0

    # 255번째 행에서 기둥의 범위 찾기 및 카운트
    y = 255
    in_white_column = False
    in_gray_column = False
    white_start = 0
    gray_start = 0

    for x in range(width):
        column_color = image_cv[y, x, 0]

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

        # Extract filename from image_path
    file_name = os.path.basename(image_path)

    return white_columns, gray_columns, white_columns_count, gray_columns_count, file_name

def calculate_overlap_percentage(gt_columns, pred_columns):
    overlap_percentages = []

    for gt_start, gt_end in gt_columns:
        gt_length = gt_end - gt_start
        max_overlap = 0

        for pred_start, pred_end in pred_columns:
            # 겹치는 부분 찾기
            overlap_start = max(gt_start, pred_start)
            overlap_end = min(gt_end, pred_end)
            if overlap_start < overlap_end:  # 실제 겹침이 있는 경우
                overlap_length = overlap_end - overlap_start
                max_overlap = max(max_overlap, overlap_length)

        # 겹침 비율 계산
        if gt_length > 0:
            overlap_percentage = (max_overlap / gt_length) * 100
            overlap_percentages.append(overlap_percentage)
        else:
            overlap_percentages.append(0)

    return overlap_percentages

if __name__ == "__main__":

    base_path_gt = path_config.BASE_PATH_GT
    base_path_label = path_config.BASE_PATH_LABEL
    directories = path_config.DIRECTORIES

    input_img_files_test = sorted([os.path.join(base_path_gt, 'test', f) for f in os.listdir(os.path.join(base_path_gt, 'test')) if f.endswith('_img.png')])

    target_img_files_test = sorted([os.path.join(base_path_label, 'test', f) for f in os.listdir(os.path.join(base_path_label, 'test')) if f.endswith('_label.png')])

    test_data = tf.data.Dataset.from_tensor_slices((input_img_files_test, target_img_files_test))

    test_ds = test_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    # test_dataset_full = (
    #     test_ds
    #     .map(unpackage_inputs)
    #     .batch(1)  # 배치 크기를 1로 설정
    #     .prefetch(buffer_size=tf.data.AUTOTUNE)
    #     )

    test_dataset_random_samples = (
        test_ds.shuffle(buffer_size=len(test_ds))
        .take(10)  # 처음 10개의 샘플만 선택
        .map(unpackage_inputs)
        .batch(1)  # 배치 크기를 1로 설정
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    # model = segnet(img_size=(256,256), num_classes=data_config.NUM_CLASSES)
    model = unet_pp_model(img_size=(256,256), num_classes=data_config.NUM_CLASSES)
    model.load_weights("/hs/HeartSignal/models/checkpoints/unet_pp.h5")

    ious = []
    for batch_img, batch_mask in test_dataset_random_samples:
        batch_pred = model.predict(batch_img, verbose=0)
        batch_iou = mean_iou(batch_mask, batch_pred)
        ious.append(batch_iou.numpy())

    # 평균 IoU 계산
    mean_iou_value = np.mean(ious)
    print("Mean IoU all over test samples:", mean_iou_value)

    losses = []

    for batch_img, batch_label in test_dataset_random_samples:
        # 모델 예측
        batch_pred = model.predict(batch_img, verbose=0)

        # 배치별 손실 계산
        batch_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_label, batch_pred, from_logits=False)
        batch_loss_mean = tf.reduce_mean(batch_loss).numpy()
        losses.append(batch_loss_mean)

    mean_loss = np.mean(losses)

    print("Mean Loss over the test samples:", mean_loss)

    # 전체 데이터셋에 대해서 추론 진행
    saved_image = inference(model, test_dataset_random_samples, 10)

    gt_dir = '/hs/HeartSignal/models/save_dir/gt'
    output_gt_dir = '/hs/HeartSignal/models/save_dir/gt_pre'
    pred_masks = '/hs/HeartSignal/models/save_dir/pred'
    output_masks_dir = '/hs/HeartSignal/models/save_dir/pred_pre'

    # 출력 디렉토리 확인 및 생성
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)

    # 이미지 처리 및 파일 경로 저장
    processed_files_gt = process_images(gt_dir, output_gt_dir)
    processed_files_pred = process_images(pred_masks, output_masks_dir)

    # Define the directory path
    directory_path_gt = '/hs/HeartSignal/models/save_dir/gt_pre'
    # List all files in the directory
    files_gt = os.listdir(directory_path_gt)
    files_gt = sorted(files_gt)

    all_white_columns_gt = []
    all_gray_columns_gt = []
    all_white_counts_gt = []
    all_gray_counts_gt = []
    file_name_gt = []

    # Loop through each file
    for file_gt in files_gt:
        # Construct the full file path
        file_path_gt = os.path.join(directory_path_gt, file_gt)
        # Call your function
        gt_white_columns, gt_gray_columns, gt_white_count, gt_gray_count, gt_file_name = find_and_count_columns(file_path_gt)

        all_white_columns_gt.append(gt_white_columns)
        all_gray_columns_gt.append(gt_gray_columns)
        all_white_counts_gt.append(gt_white_count)
        all_gray_counts_gt.append(gt_gray_count)
        file_name_gt.append(gt_file_name)

    directory_path_pred = '/hs/HeartSignal/models/save_dir/pred_pre'
    files_pred = os.listdir(directory_path_pred)
    files_pred = sorted(files_pred)

    all_white_columns_pred = []
    all_gray_columns_pred = []
    all_white_counts_pred = []
    all_gray_counts_pred = []
    file_name_pred = []

    # Loop through each file
    for file_pred in files_pred:
        # Construct the full file path
        file_path_pred = os.path.join(directory_path_pred, file_pred)
        # Call your function
        pred_white_columns, pred_gray_columns, pred_white_count, pred_gray_count, pred_file_name = find_and_count_columns(file_path_pred)

        all_white_columns_pred.append(pred_white_columns)
        all_gray_columns_pred.append(pred_gray_columns)
        all_white_counts_pred.append(pred_white_count)
        all_gray_counts_pred.append(pred_gray_count)
        file_name_pred.append(pred_file_name)

    white_overlap_percentages = calculate_overlap_percentage(all_white_columns_gt[0], all_white_columns_pred[0])
    gray_overlap_percentages = calculate_overlap_percentage(all_gray_columns_gt[0], all_gray_columns_pred[0])

    # 결과 출력
    for i, percentage in enumerate(white_overlap_percentages):
        print(f'S1 심음 {i+1}: 겹침 비율 {percentage:.2f}%')
    print('--------------------------------')
    for i, percentage in enumerate(gray_overlap_percentages):
        print(f'S2 심음 {i+1}: 겹침 비율 {percentage:.2f}%')

    total_white_overlap = 0
    total_gray_overlap = 0
    num_files = len(all_white_columns_gt)

    for i in range(num_files):
        # Calculate overlap percentages for each file
        white_overlap = calculate_overlap_percentage(all_white_columns_gt[i], all_white_columns_pred[i])
        gray_overlap = calculate_overlap_percentage(all_gray_columns_gt[i], all_gray_columns_pred[i])

        # Compute file-wise average if there are columns
        avg_white_overlap_file = sum(white_overlap) / len(white_overlap) if white_overlap else 0
        avg_gray_overlap_file = sum(gray_overlap) / len(gray_overlap) if gray_overlap else 0

        # Accumulate the averages
        total_white_overlap += avg_white_overlap_file
        total_gray_overlap += avg_gray_overlap_file

    # Calculate overall average overlap percentages
    avg_white_overlap = total_white_overlap / num_files
    avg_gray_overlap = total_gray_overlap / num_files

    # Print the results
    print(f'Overlap Percentage for S1 (White): {avg_white_overlap:.2f}%')
    print(f'Overlap Percentage for S2 (Gray): {avg_gray_overlap:.2f}%')

    guess_ave = 0
    total_images = len(all_white_counts_gt)

    for i in range(total_images):
        gt_white_count = all_white_counts_gt[i]
        gt_gray_count = all_gray_counts_gt[i]
        heart_beat_gt = (gt_white_count + gt_gray_count) / 2

        pred_white_count = all_white_counts_pred[i]
        pred_gray_count = all_gray_counts_pred[i]
        heart_beat_pred = (pred_white_count + pred_gray_count) / 2

        # Avoid division by zero
        if heart_beat_gt != 0:
            guess_by_file = (heart_beat_pred / heart_beat_gt) * 100
        else:
            guess_by_file = 0

        if heart_beat_gt == heart_beat_pred:
            guess_ave += 1

        # Print results for each file
        print(f'File: {file_name_gt[i]}')
        print(f'GT 심박수:{heart_beat_gt}', f'Pred 심박수:{heart_beat_pred}')
        print('--------------------------------')

    # Calculate overall accuracy
    guess_ave = (guess_ave / total_images) * 100
    print(f'전체 심박수 추측 정확도:{guess_ave:.2f}%')
