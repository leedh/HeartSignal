#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import argparse
from loader import DataLoader
from data.dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heart Sound Segmentation')
    # preprocessing hyperparameter
    parser.add_argument('--image_size', default=256, type=float, help='image size')
    parser.add_argument('--crop_time', default=2500, type=float, help='image crop duration')
    parser.prep_version('--prep_ver', default='v1', type=float, help='preprocessing version')
    
    # model hyperparameter
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='training epoch')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    parser.add_argument("--train_path", type=str, default='./data/train')
    parser.add_argument("--val_path", type=str,	default='./data/val')
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
    
    # Load data into class
    data = DataLoader(datadir + '/training_data')
    
    # Split dataset into train, val, test data and save them
    dataset_split(basedir, preprocess_name, data)
    
    print("Data successfully prepared.")