import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_auc_score
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

import tensorflow as tf
from AADatasets import import_mnist, import_melanoom, import_dogcat, pre_processing, make_pre_train_classes

x_train, y_train, x_test, y_test = import_mnist(0.2, True,limit=100)
x_train2, y_train2, x_test2, y_test2 = import_melanoom(r"C:\Users\s147057\Documents\Python Scripts\ISIC-2017_Training_Data", 64, 64, 0.2, True, limit = 400, color = False)
# print(x_train2.shape, y_train2.shape, x_test2.shape, y_test2.shape)
x_train = pre_processing(x_train, 64, 64, 1)
x_test = pre_processing(x_test, 64, 64, 1)

x_train2 = pre_processing(x_train2, 64, 64, 1)
x_test2 = pre_processing(x_test2, 64, 64, 1)

y_train,numb_classes = make_pre_train_classes(y_train)
y_test,numb_classes = make_pre_train_classes(y_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape = x_train.shape[1:]))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('relu'))

model.compile(optimizer='adam', loss='categorical_crossentropy')
#callback = EarlyStopping("val_loss", patience=1, verbose=0, mode='auto')
model.fit(x_train, y_train, epochs=1, batch_size=5, validation_data=(x_test, y_test))

# # Import some data to play with
# X_train = np.array(x_train).reshape(-1,28*28)
# y = y_train
# n_classes = y.shape[1]

# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

#     # Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
# lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Activation('sigmoid'))
# model.compile(optimizer='adam', loss='categorical_crossentropy')
#callback = EarlyStopping("val_loss", patience=1, verbose=0, mode='auto')
model.fit(x_train2, y_train2, epochs=1, batch_size=5, validation_data=(x_test2, y_test2)) #verbrose? 
# Calculate total roc auc score
score = roc_auc_score(y_test2, model.predict(x_test2))
print(score)

if __name__ == '__main__':
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)