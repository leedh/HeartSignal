
import os
import argparse
#import training
from loader import DataLoader, ImageLoader
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from data.dataset import *
from nets.unet import unet_model
from train import Trainer
import tensorflow as tf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Heart Sound Segmentation')
    # preprocessing hyperparameter
    parser.add_argument('--image_size', default=256, type=float, help='image size')
    parser.add_argument('--crop_time', default=2500, type=float, help='image crop duration')
    
    # model hyperparameter
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='training epoch')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    parser.add_argument("--train_path", type=str, default='./data/train')
    parser.add_argument("--val_path", type=str,	default='./data/val')
    parser.add_argument("--test_path", type=str, default='./data/test')
    parser.add_argument("--checkpoint_path",type=str, default='./checkpoints')
    args = parser.parse_args()
    print(args)

    #%%
    basedir = os.path.join(os.getcwd(), 'data')
    datadir = os.path.join(basedir, 'the-circor-digiscope-phonocardiogram-dataset-1.0.3')

    # Download dataset
    if not os.path.exists(datadir):
        print("Dataset already exists. Start downloading the dataset...")
        download_dataset(basedir)
        print("Download completed.")

    # Preprocess dataset directory
    preprocess_name = f'{args.image_size}px_{args.crop_time}ms'
    prepdir = os.path.join(basedir, preprocess_name)
    
    # Make directories for datasets of train, validation and test
    if not os.path.exists(prepdir):
        os.makedirs(prepdir, exist_ok=True)
        
    for folder in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(prepdir, folder)):
            os.makedirs(os.path.join(basedir, folder), exist_ok=True)
    
    # Load data into class
    data = DataLoader(datadir + '/training_data')
    
    # Split dataset
    dataset_split(basedir, preprocess_name, data)

    print("Data successfully prepared.")
    
    # dataset
    # ImageLoader 인스턴스 생성
    # loader = ImageLoader(directory=data_dir, image_size=(256, 256), batch_size=32)
    # # 데이터셋 가져오기
    # dataset = loader.get_dataset()
    # # 데이터셋 사용 예시 (첫 배치의 이미지 출력)
    # import matplotlib.pyplot as plt

    # for images in dataset.take(1):
    #     plt.figure(figsize=(10, 10))
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy())
    #         plt.axis("off")
    #     plt.show()
    
    train_dir = args.train_path
    val_dir = args.val_path
    test_dir = args.test_path

    # 이미지 로더 인스턴스 생성 및 데이터셋 로드
    train_loader = ImageLoader(data_dir=train_dir, img_height=256, img_width=256, batch_size=args.batch_size)
    val_loader = ImageLoader(data_dir=val_dir, img_height=256, img_width=256, batch_size=args.batch_size)
    test_loader = ImageLoader(data_dir=test_dir, img_height=256, img_width=256, batch_size=args.batch_size)

    train_dataset = train_loader.load_dataset()
    val_dataset = val_loader.load_dataset()
    test_dataset = test_loader.load_dataset()

    # 모델 초기화 및 컴파일
    img_size = (256, 256)  # u_net 함수 호출 부분에 맞는 이미지 크기 설정
    num_classes = 3  # 분류할 클래스의 수 설정
    model = unet_model(img_size, num_classes)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # 모델 학습 및 평가
    if args.train == 'train':
        trainer = Trainer(model=model, epochs=args.epoch, batch=args.batch_size,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam())
        trainer.train(train_dataset=train_dataset, train_metric=tf.keras.metrics.CategoricalAccuracy())
    else:
        # 평가 로직은 구현되지 않았습니다. 필요하다면 여기에 구현하세요.
        print("Evaluation mode is not implemented.")

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
