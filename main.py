import os
import argparse
#import training
from loader import DataLoader
from data.dataset import *

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
    parser.add_argument("--checkpoint_path",type=str, default='./checkpoints')
    args = parser.parse_args()
    print(args)
    
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
    loader = ImageLoader(directory=preprocess_name, image_size=(256, 256), batch_size=32)
    # 데이터셋 가져오기
    dataset = loader.get_dataset()
    # 데이터셋 사용 예시 (첫 배치의 이미지 출력)
    import matplotlib.pyplot as plt

    for images in dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.axis("off")
        plt.show()
    
    # 사용 예시
    data_dir = 'path/to/your/dataset'
    img_height = 256
    img_width = 256
    batch_size = 32

    loader = ImageLoader(data_dir, img_height, img_width, batch_size)
    dataset = loader.load_dataset()

    # 이 데이터셋을 Keras 모델 훈련에 사용
    # model.fit(dataset, epochs=10)
    
    
    
    
    
    # # Model load & train
    # if args.train == 'train':
    #     learning = training.train_model(train_generator, test_generator, args.epoch)
        
    # else:
    #     train_acc = learning.eval_model(trainloader)
    #     test_acc = learning.eval_model(testloader)
    #     print(f' Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
        
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss="sparse_categorical_crossentropy",
    #               metrics="accuracy")