from config import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from nets.unet import unet_model
from nets.unet_pp import unet_pp_model
from nets.segnet import segnet
from train import Trainer
from utils import *
from check_images import analyze_colors_in_dataset
import random
import numpy as np
import cv2
import keras_cv
from eval import plot_training_history

if __name__ == "__main__":

    base_path_gt = path_config.BASE_PATH_GT
    base_path_label = path_config.BASE_PATH_LABEL
    directories = path_config.DIRECTORIES

    input_img_files_train = sorted([os.path.join(base_path_gt, 'train', f) for f in os.listdir(os.path.join(base_path_gt, 'train')) if f.endswith('_img.png')])
    input_img_files_val = sorted([os.path.join(base_path_gt, 'val', f) for f in os.listdir(os.path.join(base_path_gt, 'val')) if f.endswith('_img.png')])
    input_img_files_test = sorted([os.path.join(base_path_gt, 'test', f) for f in os.listdir(os.path.join(base_path_gt, 'test')) if f.endswith('_img.png')])
    target_img_files_train = sorted([os.path.join(base_path_label, 'train', f) for f in os.listdir(os.path.join(base_path_label, 'train')) if f.endswith('_label.png')])
    target_img_files_val = sorted([os.path.join(base_path_label, 'val', f) for f in os.listdir(os.path.join(base_path_label, 'val')) if f.endswith('_label.png')])
    target_img_files_test = sorted([os.path.join(base_path_label, 'test', f) for f in os.listdir(os.path.join(base_path_label, 'test')) if f.endswith('_label.png')])

    print("Input Directory_train의 이미지 파일:", input_img_files_train[0])
    print("Input Directory_val의 이미지 파일:", input_img_files_val[0])
    print("Input Directory_test의 이미지 파일:", input_img_files_test[0])
    print("Target Directory_train의 이미지 파일:", target_img_files_train[0])
    print("Target Directory_val의 이미지 파일:", target_img_files_val[0])
    print("Target Directory_test의 이미지 파일:", target_img_files_test[0])

    train_data = tf.data.Dataset.from_tensor_slices((input_img_files_train, target_img_files_train))
    val_data = tf.data.Dataset.from_tensor_slices((input_img_files_val, target_img_files_val))
    test_data = tf.data.Dataset.from_tensor_slices((input_img_files_test, target_img_files_test))

    train_ds = train_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = val_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    print(train_ds)
    print(valid_ds)
    print(test_ds)

    train_dataset = (
        train_ds.shuffle(data_config.BATCH_SIZE)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(data_config.BATCH_SIZE)
        .map(unpackage_inputs)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    valid_dataset = (
        valid_ds.batch(data_config.BATCH_SIZE)
        .map(unpackage_inputs)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    test_dataset_random_samples = (
        test_ds.shuffle(buffer_size=len(test_ds))
        .take(10)  # 처음 10개의 샘플만 선택
        .map(unpackage_inputs)
        .batch(1)  # 배치 크기를 1로 설정
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    # backbone = keras_cv.models.ResNet50V2Backbone.from_preset(preset=train_config.MODEL,input_shape=data_config.IMAGE_SIZE+(3,),load_weights=True)
    # backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(preset = efficient_train_config.MODEL,input_shape=data_config.IMAGE_SIZE+(3,),load_weights=True) # EfficientNet Backbone
    # model = keras_cv.models.segmentation.DeepLabV3Plus(num_classes=data_config.NUM_CLASSES, backbone=backbone)
    # model_efficient = keras_cv.models.segmentation.DeepLabV3Plus(num_classes=data_config.NUM_CLASSES, backbone=backbone)
    # model_unet = unet_model(img_size=(256,256), num_classes=data_config.NUM_CLASSES)
    # model_unet_pp = unet_pp_model(img_size=(256,256), num_classes=data_config.NUM_CLASSES)
    model_segnet = segnet(img_size=(256,256), num_classes=data_config.NUM_CLASSES)

    # Get callbacks
    callbacks = get_callbacks(train_config)
    # Define Loss.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Compile model.
    model_segnet.compile(
        optimizer=tf.keras.optimizers.Adam(train_config.LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy", mean_iou],
    )

    # model_segnet.load_weights('')

    analyze_colors_in_dataset(train_dataset, n_images=5)

    history = model_segnet.fit(
    train_dataset,
    epochs=train_config.EPOCHS,
    validation_data=valid_dataset,
    callbacks=callbacks
    )

    plot_training_history(history, range(1, train_config.EPOCHS + 1), "/hs/HeartSignal/models/history")

    saved_paths = inference(model_segnet, test_dataset_random_samples, 10)