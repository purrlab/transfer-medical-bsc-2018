
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics

def load_model():
	#make a quick fucntion for the imagenet pre trained VGG16
	pass

def get_feature_vector(model, x, layer):
	# choose certain layer and make predictions and deliver vector
	model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer).output)
	if limit:	
		predictions = model.predict(x[:limit])
	else:
		predictions = model.predict(x)
	return predictions

def preform_svm(x,y,x_val,y_val):
	# preform the svm on the vector
	'''
	Simple support vector machine, Variable zijn nog aanpasbaar, maar nog niet mee geexperimenteerd. pakt automatisch een deel test en train. (afb moet vierkant zijn)
	input: afbeeldingen vector
	output: classifier en accuarcy, ook een voorbeeld.
	'''
	## SVM part ##
	clf = sklearn.svm.SVC(gamma=0.001,C=100)
	clf.fit(x,y)
	p = x[-1:]#.reshape(-1, 1)

	## accuacy quick check ##
	guess = []
	for x,y in zip(list(y_val),list(clf.predict(x_val))):
		if x == y:
			guess.append("True")
		else:
			guess.append("False")
	acc = guess.count("True")/len(guess)
	print('Accuarcy = ',acc)
	# plt.show()
	return clf

def main():
	# small example to test script
	pass
	
if __name__ == '__main__':
	main()