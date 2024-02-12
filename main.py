#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import argparse
#import training
from loader import DataLoader, ImageLoader
from config import data_config, train_config, id2color
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, save_img
from data.dataset import *
from nets.unet import unet_model
from train import Trainer
from utils import *
import tensorflow as tf
import requests
from zipfile import ZipFile
import glob
from dataclasses import dataclass
import random
import numpy as np
import cv2
import keras_cv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Heart Sound Segmentation')
    # preprocessing hyperparameter
    parser.add_argument('--image_size', default=256, type=float, help='image size')
    parser.add_argument('--crop_time', default=2500, type=float, help='image crop duration')
    parser.add_argument('--prep_ver', default='v1', type=str, help='preprocessing version')
    # model hyperparameter
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--epoch', default=2, type=int, help='training epoch')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    parser.add_argument("--train_path", type=str, default='./data/v1/train')
    parser.add_argument("--val_path", type=str,	default='./data/v1/val')
    parser.add_argument("--test_path", type=str, default='./data/v1/test')
    parser.add_argument("--checkpoint_path",type=str, default='./checkpoints')
    args = parser.parse_args()
    print(args)
    
    # Directory setting
    basedir = os.path.join(os.getcwd(), 'data')
    datadir = os.path.join(basedir, 'the-circor-digiscope-phonocardiogram-dataset-1.0.3')

    # Download dataset
    if not os.path.exists(datadir):
        print("Dataset already exists. Start downloading the dataset...")
        download_dataset(basedir)
        print("Download completed.")

    # Preprocess dataset directory
    preprocess_name = f'v1'
    #preprocess_name = f'{args.image_size}px_{args.crop_time}ms_v{args.prep_ver}'
    prepdir = os.path.join(basedir, preprocess_name)
    
    # Make directories for datasets of train, validation and test
    if not os.path.exists(prepdir):
        os.makedirs(prepdir, exist_ok=True)
        
    for folder in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(prepdir, folder)):
            os.makedirs(os.path.join(prepdir, folder), exist_ok=True)
    
    if not check_dataset_exists(prepdir):
        # Load data into class
        data = DataLoader(datadir + '/training_data')
        # Split dataset
        dataset_split(basedir, preprocess_name, data)
    else:
        print("Dataset has already been split. Skipping dataset loading and splitting.")

    print("Data successfully prepared.")

    train_dir = args.train_path
    val_dir = args.val_path
    test_dir = args.test_path

    keep_colors = [0, 127, 255]
    filter_and_save_images(train_dir, keep_colors)
    filter_and_save_images(val_dir, keep_colors)
    filter_and_save_images(test_dir, keep_colors)

    train_dir_gt = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('_img.png')])
    val_dir_gt = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('_img.png')])
    test_dir_gt = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_img.png')])
    train_dir_label = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('_filtered.png')])
    val_dir_label = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('_filtered.png')])
    test_dir_label = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_filtered.png')])

    print("train_dir 원본 이미지 파일:", train_dir_gt[0])
    print("train_dir 레이블 이미지 파일:", train_dir_label[0])
    print("val_dir 원본 이미지 파일:", val_dir_gt[0])
    print("val_dir 레이블 이미지 파일:", val_dir_label[0])
    print("test_dir 원본 이미지 파일:", test_dir_gt[0])
    print("test_dir 레이블 이미지 파일:", test_dir_label[0])

    train_data = tf.data.Dataset.from_tensor_slices((train_dir_gt, train_dir_label))
    val_data = tf.data.Dataset.from_tensor_slices((val_dir_gt, val_dir_label))
    test_data = tf.data.Dataset.from_tensor_slices((test_dir_gt, test_dir_label))

    train_ds = train_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = val_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    plot_train_ds = train_ds.map(unpackage_inputs).batch(3)
    image_batch, mask_batch = next(iter(plot_train_ds.take(1)))

    titles = ["GT Image", "GT Mask", "Overlayed Mask"]

    for image, gt_mask in zip(image_batch, mask_batch):

        gt_mask = tf.squeeze(gt_mask, axis=-1).numpy()
        display_image_and_mask([image.numpy().astype(np.uint8), gt_mask],
                            title_list=titles,
                            figsize=(16,6),
                            color_mask=True)


    augment_fn = tf.keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandomBrightness(factor=data_config.BRIGHTNESS_FACTOR,
        value_range=(0, 255)),
        keras_cv.layers.RandomContrast(factor=data_config.CONTRAST_FACTOR,
        value_range=(0, 255)),
    ])

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

    # backbone = keras_cv.models.ResNet50V2Backbone.from_preset(preset = train_config.MODEL,
    #                                                       input_shape=data_config.IMAGE_SIZE+(3,),
    #                                                       load_weights = True)
    # model = keras_cv.models.segmentation.DeepLabV3Plus(num_classes=data_config.NUM_CLASSES, backbone=backbone)

    # # Build model.

    # # Get callbacks.
    # callbacks = get_callbacks(train_config)
    # # Define Loss.
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # # Compile model.
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(train_config.LEARNING_RATE),
    #     loss=loss_fn,
    #     metrics=["accuracy", mean_iou],
    # )

    # history = model.fit(
    # train_dataset,
    # epochs=train_config.EPOCHS,
    # validation_data=valid_dataset,
    # callbacks=callbacks
    # )

    img_size = (256, 256)
    num_classes = 3
    model = unet_model(img_size, num_classes)

    # 모델 컴파일
    model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"])
    
    # 훈련 및 검증 메트릭 준비
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Trainer 인스턴스 생성 및 학습 실행
    trainer = Trainer(model=model, epochs=args.epoch, batch=args.batch_size,
                    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer=tf.keras.optimizers.Adam(),
                    valid_dataset=valid_dataset)
    trainer.train(train_dataset=train_dataset,
                train_metric=train_metric,
                valid_metric=valid_metric)

    # 추론 함수 실행
    saved_image_paths = inference(model, test_dataset_random_samples, 5)
