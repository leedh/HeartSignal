from config import data_config, train_config, path_config, id2color, augment_fn, get_callbacks
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from nets.unet import unet_model
from train import Trainer
from utils import *
from check_images import save_batch_images_with_masks
import random
import numpy as np
import cv2
import keras_cv

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

    save_batch_images_with_masks(train_ds, "/hs/HeartSignal/eda/check_train_ds", batch_size=3)

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

    backbone = keras_cv.models.ResNet50V2Backbone.from_preset(preset=train_config.MODEL,input_shape=data_config.IMAGE_SIZE+(3,),load_weights = True)
    model = keras_cv.models.segmentation.DeepLabV3Plus(num_classes=data_config.NUM_CLASSES, backbone=backbone)

    # Get callbacks.
    callbacks = get_callbacks(train_config)
    # Define Loss.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_config.LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy", mean_iou],
    )

    # model.load_weights('/content/drive/MyDrive/aiffelthon_heartsignal/HeartSignal/models/deeplabv3_plus_resnet50_v2.h5')

    history = model.fit(
    train_dataset,
    epochs=train_config.EPOCHS,
    validation_data=valid_dataset,
    callbacks=callbacks
    )

    # img_size = (256, 256)
    # num_classes = 3
    # model = unet_model(img_size, num_classes)

    # # 모델 컴파일
    # model.compile(optimizer="adam",
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #             metrics=["accuracy"])
    
    # # 훈련 및 검증 메트릭 준비
    # train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    # valid_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # # Trainer 인스턴스 생성 및 학습 실행
    # trainer = Trainer(model=model, epochs=args.epoch, batch=args.batch_size,
    #                 loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #                 optimizer=tf.keras.optimizers.Adam(),
    #                 valid_dataset=valid_dataset)
    # trainer.train(train_dataset=train_dataset,
    #             train_metric=train_metric,
    #             valid_metric=valid_metric)