
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics
import keras.backend as K

from sklearn import metrics
from keras import backend as K

# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc
    
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def make_model(x, y, numb_classes):
	
	# make and compile vgg16 model with correct parameters
	vgg_conv = tf.keras.applications.VGG16(weights=None,input_shape = (x[0].shape), include_top=True, classes=numb_classes) #top??
	adm = tf.keras.optimizers.SGD(lr=0.008, momentum=0.0, decay=0.0, nesterov=False)
	vgg_conv.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])  #'auc'

	return vgg_conv

def train_model(model,X,Y,):
	# train models over AUC, for x epochs. make it loopable for further test. return plottable data
	H = model.fit(X,Y, batch_size=batch_size_manual, epochs=E, validation_split=0.1)

def config_desktop():
	## WHEN USING TF 1.5 or lower and GPU ###
	config = tf.ConfigProto()               #
	config.gpu_options.allow_growth = True  #
	session = tf.Session(config=config)     #
	#########################################

def main():
	# small example to test script
	pass

if __name__ == '__main__':
	main()