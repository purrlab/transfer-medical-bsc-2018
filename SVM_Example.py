''' 
Support vector Machine
'''

import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm, metrics

def get_mnist():
	mnist_data = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	X_train = np.reshape(x_train, (-1,28*28))
	X_train.shape
	return X_train, , y_train,x_test, y_test

def svm_setup():
	gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
	gamma_range = gamma_range.flatten()
	C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
	C_range = C_range.flatten()
	parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}
	svm_clsf = svm.SVC()
	grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)
	return grid_clsf

if __name__ == '__main__':
	X_train, , y_train,x_test, y_test = get_mnist()
	grid_clsf = svm_setup()
	grid_clsf.fit(X_train, y_train)