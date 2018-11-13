'''Experiment 1'''
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG16
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

X_test = np.reshape(x_test, (-1,28*28)) #SVM


# prep data for pre train
train_data = []
for img in x_train:
    resized_image = cv2.resize(img, (64, 64))
    train_data.append(resized_image)
    
X = np.array(train_data).reshape(-1,64,64,1)

test_data = []
for img in x_test:
    resized_image = cv2.resize(img, (64, 64))
    test_data.append(resized_image)
    
X_test = np.array(test_data).reshape(-1,4096)

vgg_conv = VGG16(weights=None,input_shape = (64,64,1), include_top=True, classes=10)

y = []
for label in y_train:
    if label == 0:
        y.append([1,0,0,0,0,0,0,0,0,0])
    elif label == 1:
        y.append([0,1,0,0,0,0,0,0,0,0])
    elif label == 2:
        y.append([0,0,1,0,0,0,0,0,0,0])
    elif label == 3:
        y.append([0,0,0,1,0,0,0,0,0,0])
    elif label == 4:
        y.append([0,0,0,0,1,0,0,0,0,0])
    elif label == 5:
        y.append([0,0,0,0,0,1,0,0,0,0])
    elif label == 6:
        y.append([0,0,0,0,0,0,1,0,0,0])
    elif label == 7:
        y.append([0,0,0,0,0,0,0,1,0,0])
    elif label == 8:
        y.append([0,0,0,0,0,0,0,0,1,0])
    elif label == 9:
        y.append([0,0,0,0,0,0,0,0,0,1])
Y = np.array(y).reshape(-1,10)


vgg_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
H = vgg_conv.fit(X,Y, batch_size=32, epochs=1, validation_split=0.1)