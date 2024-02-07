#!/usr/bin/python3
import keras
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from keras.utils import to_categorical
from tensorflow.keras.utils import Progbar
# load custom modules
from loader import *
from utils import *
from nets.unet import unet_model

# from models import model

# def train_model(train_generator, valid_generator, epoch):
#     model.fit(train_generator,
#             validation_data=valid_generator,
#             epochs=epoch)

# def eval_model(test_generator):
#     _, acc = model.evaluate(test_generator)

#     return acc

class Trainer:
    def __init__(self, model, epochs, batch, loss_fn, optimizer, valid_dataset=None):
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.valid_dataset = valid_dataset

    def train(self, train_dataset, train_metric, valid_metric=None):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                train_metric.update_state(y_batch_train, logits)
                if step % 10 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %d samples" % ((step + 1) * self.batch))
                    print(train_metric.result().numpy())

            train_acc = train_metric.result()
            print("Training metric over epoch: %.4f" % (float(train_acc),))
            train_metric.reset_states()

            if self.valid_dataset is not None and valid_metric is not None:
                self.evaluate(self.valid_dataset, valid_metric)

    def evaluate(self, valid_dataset, valid_metric):
        for x_batch_val, y_batch_val in valid_dataset:
            val_logits = self.model(x_batch_val, training=False)
            valid_metric.update_state(y_batch_val, val_logits)
        val_acc = valid_metric.result()
        valid_metric.reset_states()
        print("Validation metric: %.4f" % (float(val_acc),))