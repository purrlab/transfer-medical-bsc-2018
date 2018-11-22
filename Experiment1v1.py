'''Experiment 1'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics

def get_mnist_data():
	'''
	Creert data set van de mnist data.
	input: None
	output: train en test data set, each having their own list of classes and images.
	'''
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	return x_train, y_train, x_test, y_test

# X_test = np.reshape(x_test, (-1,28*28)) #SVM

def transform_img_list(X,img_size_x,img_size_y,demension):
	'''
	Door bijvoorbeeld te kleine afbeeldingen, wordt de data getransformeerd naar een ander formaat, later kan hier nog data generators aan toegevoegd worden
	input: lijst met afbeeldingen, en de afmeting van de gewenste afbeelding.
	output: Lijst met afbeeldingen getrasformeerd naar de afmeting.
	'''
	train_data = []
	for img in X:
	    resized_image = cv2.resize(img, (img_size_x, img_size_y))
	    train_data.append(resized_image)

	X = np.array(train_data).reshape(-1,img_size_x,img_size_y,demension)
	return X

def make_pre_train_classes(Y, numb_classes = None):
	'''
	zorgt voor classes op een manier dat een NN er mee kan werken, deze versie werkt op INTERGERS
	input: lijst met interger classes
	outpt: 0'en en 1'en lijst, even lang als classes aantal
	'''
	clas = list()
	if not numb_classes:
		numb_classes = max(Y)-min(Y)+1

	for label in Y:
	    new_list = numb_classes*[0]
	    new_list[label] = 1
	    clas.append(new_list)
	clas_np = np.array(clas).reshape(-1,numb_classes)
	return clas_np, numb_classes

def svm(x,y):
	'''
	Simple support vector machine, Variable zijn nog aanpasbaar, maar nog niet mee geexperimenteerd. pakt automatisch een deel test en train. (afb moet vierkant zijn)
	input: afbeeldingen vector
	output: classifier en accuarcy, ook een voorbeeld.
	'''
	split_value = int((len(x)*0.10))
	x,y = x[:-split_value],y[:-split_value]
	clf = sklearn.svm.SVC(gamma=0.001,C=100)
	clf.fit(x,y)
	p = x[-1:]#.reshape(-1, 1)
	print('prediction: ',clf.predict(p))
	plt.imshow(x[-1].reshape(-1, int(len(x[0])**0.5)), cmap=plt.cm.gray_r, interpolation ='nearest') ## werkt dit? wortel van afbeelding == size?
	plt.show()

	## waarom niet bij elkaar? acc()
	guess = []
	for x,y in zip(list(y[-split_value:]),list(clf.predict(x[-split_value:]))):
		if x == y:
			guess.append("True")
		else:
			guess.append("False")
	acc = guess.count("True")/len(guess)
	print('Accuarcy = ',acc)
	return clf

def acc(x,y,clf): # maak procentueel
	guess = []
	for x,y in zip(list(y[-10:]),list(clf.predict(x[-10:]))):
		if x == y:
			guess.append("True")
		else:
			guess.append("False")
	acc = guess.count("True")/len(guess)
	print('Accuarcy = ',acc)

def plot_epochs(H):
	'''
	Bij gebruik van vele epochs zijn deze plots handig
	input: NN
	output: Graphs
	'''
	# summarize history for accuracy
	plt.plot(H.history['acc'])
	plt.plot(H.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(H.history['loss'])
	plt.plot(H.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def Experiment_1():
	'''
	voert het eerste experiment uit, onder de onderstaande parameters, tevens geeft het een mooi overzicht van de handelingen.
	input: None
	output: None
	'''
	## Parameters ##
	img_size_x = 64
	img_size_y = 64
	demension = 1
	limit = 100
	batch_size_manual = 2
	E = 1

	data_size = [limit, img_size_x,img_size_y,demension] # idea?

	## Run Functions ##
	x_train, y_train, x_test, y_test = get_mnist_data()
	X = transform_img_list(x_train,img_size_x,img_size_y,demension)
	X_test = (transform_img_list(x_test,img_size_x,img_size_y,demension))#np.array#.reshape(-1,img_size_x*img_size_y)
	Y,numb_classes = make_pre_train_classes(y_train)
	vgg_conv = VGG16(weights=None,input_shape = (img_size_x,img_size_y,demension), include_top=True, classes=numb_classes) #top??
	vgg_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	H = vgg_conv.fit(X[:limit],Y[:limit], batch_size=batch_size_manual, epochs=E, validation_split=0.1)
	model = tf.keras.models.Model(inputs=vgg_conv.input, outputs=vgg_conv.get_layer('fc2').output)
	predictions = model.predict(X_test[:limit])
	clf = svm(predictions,y_test[:limit])




if __name__ == '__main__':
	Experiment_1()
	# make class of experiment. with parameters and technics class ---> exp1 = experiment_all(parameter_list)
