#!/usr/bin/python3

import keras
from keras.utils import to_categorical
from tensorflow.keras.utils import Progbar
# load custom modules
from dataloader import *

# from models import model

# def train_model(train_generator, valid_generator, epoch):
#     model.fit(train_generator,
#             validation_data=valid_generator,
#             epochs=epoch)

# def eval_model(test_generator):
#     _, acc = model.evaluate(test_generator)

#     return acc

class Trainer:
    def __init__(self, model, epochs, batch, loss_fn, optimizer):
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def train(self, train_dataset, train_metric):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset): # batch size만큼 데이터를 가져옴
                with tf.GradientTape() as tape: # 자동 미분을 위한 테이프
                    logits = self.model(x_batch_train, training=True) # 모델의 예측값
                    loss_value = self.loss_fn(y_batch_train, logits) # loss 계산
                grads = tape.gradient(loss_value, self.model.trainable_weights) # loss에 대한 모델의 trainable한 변수의 gradient 계산
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights)) # 모델의 변수 업데이트
                # Update training metric.
                train_metric.update_state(y_batch_train, logits) # y_batch_train과 logits으로 train_metric 업데이트
                # Log every 10 batches.
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * self.batch))
                    print(train_metric.result().numpy())
            # Display metrics at the end of each epoch.
            train_acc = train_metric.result()
            print("Training metric over epoch: %.4f" % (float(train_acc),))
        

        # # Train the model
        # model.fit(self.train_images, self.train_labels, epochs=5, batch_size=128)

        # # Evaluate the model
        # test_loss, test_acc = model.evaluate(self.test_images, self.test_labels)
        # print('test_acc:', test_acc)

if __name__ == "__main__":
    epoch = 1
    batch = 5
    model = YogaPose(num_classes=5)
    dataset = load_data(data_path=data_path, batch_size=batch)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    trainer = Trainer(model=model,
                  epochs=epoch,
                  batch=batch,
                  loss_fn=loss_function,
                  optimizer=optimizer)
    trainer.train(train_dataset=dataset,
                train_metric=train_acc_metric)