#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import numpy as np
from config import data_config, train_config, id2color
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, save_img
import cv2
import cmapy
from tqdm import tqdm
import gc
# load custom modules
from preprocess.labels import *
from preprocess.biquad import *
from preprocess.denoising import *
from preprocess.blank_region_clipping import *

# time count decorator
def time_calculator(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: <{func.__name__}> execution took {end_time - start_time:.2f} sec")
        return result
    return wrapper

# load the files from dirPath
def load_files(dirPath):
    files = list()
    file_list = os.listdir(dirPath)
    for f in tqdm(file_list, desc='Loading files'):
        if f.endswith('.wav'):
            files.append(f)
    return files

def create_mel_spectrogram_raw(waveform, target_sr, hop_length, win_length, n_fft, n_mels, resz=0):
    S = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
    # Convert Mel spectrogram to Decibel scale
    S = librosa.power_to_db(S, ref=np.max) # why np.max?: https://dsp.stackexchange.com/questions/64509/why-should-one-choose-the-maximum-as-reference-for-power-to-decibel-conversion
    # normalization
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width*resz, height*resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img

#def figure_mel_spec(mel, type, save_path, sr, hop_length):
def figure_mel_spec(mel, type, save_path):
    colormap = 'magma' if type == 'img' else 'gray'
    # Figure와 Axes 객체를 명시적으로 생성
    fig, ax = plt.subplots(dpi=95)  # DPI 설정을 여기에 포함
    ax.imshow(mel, cmap=colormap)
    ax.axis('off')  # 축 끄기

    # tight_layout 호출로 레이아웃 최적화
    fig.tight_layout(pad=0)  # pad를 0으로 설정하여 여백 최소화

    # 이미지 파일로 저장
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # Figure 닫기 (메모리 해제를 위해)
    plt.close(fig)  # 명시적으로 Figure 객체를 닫음

    # 가비지 컬렉션 실행
    gc.collect()
    
    # plt.figure(dpi=95) # set the size of the figure
    # plt.imshow(mel, cmap=colormap)
    # plt.axis('off') # turn off axis
    # #librosa.display.specshow(mel, sr=sr, hop_length=hop_length, vmin=-80, vmax=0, cmap='inferno')
    # #ax.set_aspect('auto') # set aspect ratio to be equal
    # plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.close()  # close the figure
    
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

def check_dataset_exists(prepdir):
    """분할된 데이터셋의 폴더들이 비어있는지 확인합니다."""
    for folder in ['train', 'val', 'test']:
        folder_path = os.path.join(prepdir, folder)
        if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
            return True
    return False

def filter_and_save_images(dir_path, keep_colors):
    """
    특정 색상을 유지하며 레이블 이미지를 필터링하고 같은 폴더에 다른 이름으로 저장하는 함수.

    Parameters:
    - dir_path: 이미지 파일이 있는 디렉토리 경로. 이 경로는 입력과 출력 모두에 사용됩니다.
    - keep_colors: 유지할 색상 값의 리스트.
    """
    
    # 입력 디렉토리에서 _label.png로 끝나는 이미지 파일 목록을 가져옴
    img_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('_label.png')]
    print(img_files[0])
    # 이미지 파일들을 순회하며 처리
    for file in img_files:
        # 이미지 불러오기
        img_path = os.path.join(dir_path, file)
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
        filtered_img_path = os.path.splitext(img_path)[0] + '_filtered.png'
        save_img(filtered_img_path, filtered_img)

def read_image_mask(image_path, mask=False, size=data_config.IMAGE_SIZE):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=size, method="bicubic")

        # 원본 마스크 이미지에서 라벨 별로 픽셀을 분리
        label_0_mask = tf.cast(tf.equal(image, 0), tf.uint8)
        label_1_mask = tf.cast(tf.equal(image, 127), tf.uint8)
        label_2_mask = tf.cast(tf.equal(image, 255), tf.uint8)

        # 여러 라벨을 하나의 마스크로 결합
        combined_mask = label_0_mask * 0 + label_1_mask * 1 + label_2_mask * 2
        return combined_mask

    else:
        image = tf.io.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=size, method="bicubic")
        image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.float32)

    return image

def load_data(image_list, mask_list):
    image = read_image_mask(image_list)
    mask  = read_image_mask(mask_list, mask=True)
    return {"images":image, "segmentation_masks":mask}

def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks

def num_to_rgb(num_arr, color_map=id2color):
    # single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2]+(3,))
    for k in color_map.keys():
        output[num_arr==k] = color_map[k]
    return output.astype(np.uint8)

def image_overlay(image, segmented_image):
    alpha = 0.7 # Transparency for the original image.
    beta  = 0.9 # Transparency for the segmentation map.
    gamma = 0.0 # Scalar added to each sum.
    image = image.astype(np.uint8)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def display_image_and_mask(data_list, title_list, figsize, color_mask=False, color_map=id2color):
    # Create RGB segmentation map from grayscale segmentation map.
    rgb_gt_mask = num_to_rgb(data_list[1], color_map=color_map)
    mask_to_overlay = rgb_gt_mask
    if len(data_list)==3:
        rgb_pred_mask = num_to_rgb(data_list[-1], color_map=color_map)
        mask_to_overlay = rgb_pred_mask
    # Create the overlayed image.
    overlayed_image = image_overlay(data_list[0], mask_to_overlay)
    data_list.append(overlayed_image)
    fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)
    for idx, axis in enumerate(axes.flat):
        axis.set_title(title_list[idx])
        if title_list[idx] == "GT Mask":
            if color_mask:
                axis.imshow(rgb_gt_mask)
            else:
                axis.imshow(data_list[1], cmap="gray")
        elif title_list[idx] == "Pred Mask":
            if color_mask:
                axis.imshow(rgb_pred_mask)
            else:
                axis.imshow(data_list[-1], cmap="gray")
        else:
            axis.imshow(data_list[idx])
        axis.axis('off')
    plt.show()

def get_callbacks(
    train_config,
    monitor="val_mean_iou",
    mode="max",
    save_weights_only=True,
    save_best_only=True,
):

    # Initialize tensorboard callback for logging.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=train_config.LOGS_DIR,
        histogram_freq=20,
        write_graph=False,
        update_freq="epoch",
    )


    # Update file path if saving best model weights.
    if save_weights_only:
        checkpoint_filepath = train_config.CKPT_DIR

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=save_weights_only,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1,
    )

    return [tensorboard_callback, model_checkpoint_callback]

def mean_iou(y_true, y_pred):
    # Get total number of classes from model output.
    num_classes = y_pred.shape[-1]
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, axis=-1)
    y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes, axis=-1)
    # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))
    # Total Sum: |G| + |P|. Shape: (batch_size, num_classes)
    total = tf.math.reduce_sum(y_true, axis=(1, 2)) + tf.math.reduce_sum(y_pred, axis=(1, 2))
    union = total - intersection
    is_class_present =  tf.cast(tf.math.not_equal(total, 0), dtype=tf.float32)
    num_classes_present = tf.math.reduce_sum(is_class_present, axis=1)
    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.math.reduce_sum(iou, axis=1) / num_classes_present
    # Compute the mean across the batch axis. Shape: Scalar
    mean_iou = tf.math.reduce_mean(iou)
    return mean_iou

def save_mask_images(mask, pred, count):
    # GT와 Pred를 위한 디렉토리 생성
    gt_dir = '/hs/HeartSignal/data/256px_2500ms/gt'
    pred_dir = '/hs/HeartSignal/data/256px_2500ms/pred'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    # GT Mask 저장
    gt_mask_path = os.path.join(gt_dir, f'gt_{count}.png')
    plt.imsave(gt_mask_path, mask, cmap='gray')
    # Pred Mask 저장
    pred_mask_path = os.path.join(pred_dir, f'pred_{count}.png')
    plt.imsave(pred_mask_path, pred, cmap='gray')
    return gt_mask_path, pred_mask_path

def inference(model, dataset, samples_to_plot):
    saved_paths = []
    count = 0
    stop_plot = False
    titles = ["Image", "GT Mask", "Pred Mask", "Overlayed Prediction"]
    for idx, data in enumerate(dataset):
        if stop_plot:
            break
        batch_img, batch_mask = data[0], data[1]
        batch_pred = (model.predict(batch_img)).astype('float32')
        batch_pred = batch_pred.argmax(axis=-1)
        batch_img = batch_img.numpy().astype('uint8')
        batch_mask = batch_mask.numpy().squeeze(axis=-1)
        for image, mask, pred in zip(batch_img, batch_mask, batch_pred):
            display_image_and_mask([image, mask, pred], title_list=titles, figsize=(20,8), color_mask=True)
            # 이미지 저장
            gt_mask_path, pred_mask_path = save_mask_images(mask, pred, count + 1)
            saved_paths.append((gt_mask_path, pred_mask_path))
            count += 1
            if count >= samples_to_plot:
                stop_plot = True
                break
    return saved_paths