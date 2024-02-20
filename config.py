
from dataclasses import dataclass
import os
from dataclasses import dataclass, field
import tensorflow as tf
import keras_cv

@dataclass(frozen=True)
class PathConfig:
    BASE_PATH_GT: str = "/hs/HeartSignal/data/v1_img/"
    BASE_PATH_LABEL: str = "/hs/HeartSignal/data/filtered/"
    DIRECTORIES: list = field(default_factory=lambda: ["train", "val", "test"])

@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE:        tuple = (256, 256)
    BATCH_SIZE:          int = 32
    NUM_CLASSES:         int = 3
    BRIGHTNESS_FACTOR: float = 0.3
    CONTRAST_FACTOR:   float = 0.3

@dataclass(frozen=True)
class TrainingConfig:
    MODEL:           str = "Segnet_aug"
    EPOCHS:          int = 25
    LEARNING_RATE: float = 1e-3
    CKPT_DIR:        str = os.path.join("/hs/HeartSignal/models/checkpoints_"+"_".join(MODEL.split("_")[:2]),
                                        "model_"+"_".join(MODEL.split("_")[:2])+".h5")
    LOGS_DIR:        str = "logs_"+"_".join(MODEL.split("_")[:2])

data_config = DatasetConfig()
path_config = PathConfig()
train_config = TrainingConfig()

augment_fn = tf.keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandomBrightness(factor=data_config.BRIGHTNESS_FACTOR,value_range=(0, 1)),
        keras_cv.layers.RandomContrast(factor=data_config.CONTRAST_FACTOR,value_range=(0, 1))
    ])

id2color = {
    0: 0,    # 검정색
    1: 127,  # 회색
    2: 255   # 흰색
}

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