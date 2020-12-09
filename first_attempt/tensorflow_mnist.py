#!/usr/bin/env python
# coding: utf-8

# The mnist dataset contains ~70.000 handwritten digits assembled by http://yann.lecun.com/
# It is considered the "Hello World" of Machine Learning
# Each image of a digit is 28x28 pixels, or 784 pixels large
# The digits from 0-9 translate into 10 categories
# Images are flattened into vector 784x1, each entry corresponds to greyscale color value, 255=white, 0=black

import tensorflow as tf
import tensorflow_datasets as tfds

# import data, preprocess
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)  # already contained in tfds
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']  # training and testing are preset in lib

# create validation data subset (10% of training set)
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)  # cast to integer
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


def scale(image, label):  # scale input data to make result more numerically stable (between 0 and 1)
    image = tf.cast(image, tf.float32)
    image /= 255.  # divide each value by 255 to receive floats between 0 and 1
    return image, label  # return data in this structure to enable processing of function in dataset.map()


scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)

# shuffle data to enable batching
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

# Mini-Batch Gradient Descent for training model
BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(
    num_validation_samples)  # not necessary logically, but model expects data batched
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))  # make iterable for option as_supervised

# Outline the model
input_size = 784
output_size = 10
hidden_layer_size = 500

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # rectified linear unit
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # scc applies one-hot encoding

# Training the model
NUM_EPOCHS = 10

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)

# goal was validation accuracy >97%
# succeeded with validation accuracy of 98.7%, though fairly slow

######################## Output ########################
# Epoch 1/10
# 540/540 - 14s - loss: 0.3374 - accuracy: 0.9024 - val_loss: 0.2437 - val_accuracy: 0.9427
# Epoch 2/10
# 540/540 - 13s - loss: 0.1531 - accuracy: 0.9621 - val_loss: 0.1134 - val_accuracy: 0.9705
# Epoch 3/10
# 540/540 - 13s - loss: 0.1108 - accuracy: 0.9721 - val_loss: 0.1368 - val_accuracy: 0.9668
# Epoch 4/10
# 540/540 - 13s - loss: 0.0960 - accuracy: 0.9764 - val_loss: 0.0734 - val_accuracy: 0.9818
# Epoch 5/10
# 540/540 - 13s - loss: 0.0819 - accuracy: 0.9803 - val_loss: 0.0959 - val_accuracy: 0.9788
# Epoch 6/10
# 540/540 - 13s - loss: 0.0730 - accuracy: 0.9822 - val_loss: 0.0926 - val_accuracy: 0.9767
# Epoch 7/10
# 540/540 - 13s - loss: 0.0689 - accuracy: 0.9840 - val_loss: 0.0611 - val_accuracy: 0.9847
# Epoch 8/10
# 540/540 - 12s - loss: 0.0595 - accuracy: 0.9862 - val_loss: 0.1594 - val_accuracy: 0.9660
# Epoch 9/10
# 540/540 - 13s - loss: 0.0610 - accuracy: 0.9857 - val_loss: 0.0702 - val_accuracy: 0.9840
# Epoch 10/10
# 540/540 - 13s - loss: 0.0559 - accuracy: 0.9871 - val_loss: 0.0498 - val_accuracy: 0.9872

# Test model
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0: .2f}. Test accuracy: {1: .2f}%'.format(test_loss, test_accuracy * 100.))

######################## Output ########################
# 1/1 [==============================] - 0s 2ms/step - loss: 0.1010 - accuracy: 0.9797
# Test loss:  0.10. Test accuracy:  97.97%