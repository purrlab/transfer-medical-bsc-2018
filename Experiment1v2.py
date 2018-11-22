'''Experiment 1'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics

# desktop needs this

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_mnist_data(split = False, norm = True):
	'''
	Creert data set van de mnist data.
	input: Normalize boolean
	output: train en test data set, each having their own list of classes and images.
	'''
	if type(split) != float:
		print("please enter 'float' for split")
	if type(norm) != bool:
		print("please enter 'boolean' for norm(alization)")

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x = np.append(x_test,x_train,axis=0)
	y = np.append(y_test,y_train,axis=0)
	if split:
		spl = int(split*len(x))
		x_train = x[spl:]
		y_train = y[spl:]
		x_test  = x[:spl]
		y_test  = y[:spl]
	else:
		x_train = x
		y_train = y
		x_test  = 0
		y_test  = 0

	if norm:
		x_train, x_test = x_train / 255.0, x_test / 255.0
	return x_train, y_train, x_test, y_test

# X_test = np.reshape(x_test, (-1,28*28)) #SVM

def transform_img_list(X,img_size_x,img_size_y,demension):
	'''
	Door bijvoorbeeld te kleine afbeeldingen, wordt de data getransformeerd naar een ander formaat, later kan hier nog data generators aan toegevoegd worden
	input: lijst met afbeeldingen, en de afmeting van de gewenste afbeelding.
	output: Lijst met afbeeldingen getrasformeerd naar de afmeting.
	'''
	if type(X) != np.ndarray and type(X) != list:
		print("Please enter a list or np.array")
	if img_size_x < 48 or img_size_y < 48:
		print("img_size is too small, VGG16 needs an image size larger than 48X48")

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
	try:
		int(Y[0])
	except Exception as e:
		raise e
	else:
		print('Please give classes in compatible type(), int, unti8')

	clas = list()
	if not numb_classes:
		numb_classes = max(Y)-min(Y)+1

	for label in Y:
	    new_list = numb_classes*[0]
	    new_list[int(label)] = 1
	    clas.append(new_list)
	clas_np = np.array(clas).reshape(-1,numb_classes)
	return clas_np, numb_classes

def svm(x,y):
	'''
	Simple support vector machine, Variable zijn nog aanpasbaar, maar nog niet mee geexperimenteerd. pakt automatisch een deel test en train. (afb moet vierkant zijn)
	input: afbeeldingen vector
	output: classifier en accuarcy, ook een voorbeeld.
	'''
	## SVM part ##
	split_value = int((len(x)*0.10))
	x,y = x[:-split_value],y[:-split_value]
	clf = sklearn.svm.SVC(gamma=0.001,C=100)
	clf.fit(x,y)
	p = x[-1:]#.reshape(-1, 1)
	print('prediction: ',clf.predict(p))
	plt.imshow(x[-1].reshape(-1, int(len(x[0])**0.5)), cmap=plt.cm.gray_r, interpolation ='nearest') ## werkt dit? wortel van afbeelding == size?

	## accuacy quick check ##
	guess = []
	for x,y in zip(list(y[-split_value:]),list(clf.predict(x[-split_value:]))):
		if x == y:
			guess.append("True")
		else:
			guess.append("False")
	acc = guess.count("True")/len(guess)
	print('Accuarcy = ',acc)
	# plt.show()
	return clf

def plot_epochs(H):
	'''
	Bij gebruik van vele epochs zijn deze plots handig
	input: NN
	output: Graphs d.m.v. plt.show()
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

def Random_score(y):
	'''
	Get a score for randomness, takes steps equal to amount of classes, sees if it contains all the classes.
	Could set score limit
	input: classes
	output: score of given data
	'''
	## checks if they are intergers based classes ##
	if type(Y[0]) != np.ndarray:
		back_to_num = list()
		for i in Y:
			back_to_num.append(list(i).index(1))

	score = 0
	numb_classes = max(y)-min(y)+1
	totaal = int(len(y)/numb_classes)
	score = 0
	for n in range(0,len(y[:totaal*numb_classes]),numb_classes):
		combi = (len(set(y[n:n+numb_classes])))
		score += combi/(totaal*numb_classes)
	print(score)
	return y, score

def count_classes():
	'''
	Counts classes, no matter wich type of class notation it is, array of 1 and 0 or interger.
	input: class list
	output: class dict, {class:count}
	'''
	## checks if they are intergers ##
	if type(Y[0]) == np.ndarray:
		back_to_num = list()
		for i in Y:
			back_to_num.append(list(i).index(1))

	d = dict()
	for n in back_to_num:
		if n in d:
			d[n] += 1
		else:
			d[n] = 1
	return d


def Experiment_1(parameters):
	'''
	voert het eerste experiment uit, onder de onderstaande parameters, tevens geeft het een mooi overzicht van de handelingen.
	"idee: elk deze sub kop wordt een eige file, parameterfile zal alles samen voegen en runnen."
	input: parameters
	output: None
	'''

	## Parameters ##
	batch_size_manual = parameters[1]
	E = parameters[2]
	data_size = parameters[0]
	split = parameters[4]
	norm = parameters[3]

	## Run Functions ##
	# Prep data
	x_train, y_train, x_test, y_test = get_mnist_data(split, norm)
	X = transform_img_list(x_train,data_size[1],data_size[2],data_size[3])
	X_test = (transform_img_list(x_test,data_size[1],data_size[2],data_size[3]))
	Y,numb_classes = make_pre_train_classes(y_train)
	# Pre train
	vgg_conv = tf.keras.applications.VGG16(weights=None,input_shape = (data_size[1],data_size[2],data_size[3]), include_top=True, classes=numb_classes) #top??

	adm = tf.keras.optimizers.SGD(lr=0.008, momentum=0.0, decay=0.0, nesterov=False)

	vgg_conv.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
	if limit:
		H = vgg_conv.fit(X[:data_size[0]],Y[:data_size[0]], batch_size=batch_size_manual, epochs=E, validation_split=0.1)
		model = tf.keras.models.Model(inputs=vgg_conv.input, outputs=vgg_conv.get_layer('fc2').output)
		#Featurize and SVM
		predictions = model.predict(X_test[:data_size[0]])
		clf = svm(predictions,y_test[:data_size[0]])
	else:
		print(X.shape)
		print(X_test.shape)

		H = vgg_conv.fit(X,Y, batch_size=batch_size_manual, epochs=E, validation_split=0.1)
		model = tf.keras.models.Model(inputs=vgg_conv.input, outputs=vgg_conv.get_layer('fc2').output)
		#Featurize and SVM
		print(X_test.shape)
		predictions = model.predict(X_test)
		clf = svm(predictions,y_test)
		clf = svm(predictions[:1000],y_test[:1000])
		plot_epochs(H)
	# plt.show() # when you uses the show fucntion somewhere halfway, activated it here, so the whole script can run propperly.



if __name__ == '__main__':
	## options ##
	img_size_x = 64
	img_size_y = 64
	demension = 1
	limit = False
	Batch_size = 32
	Epochs = 10
	norm = True
	split = 0.14285714285714285 # (1/7)

	## compress ##
	data_size = [limit, img_size_x,img_size_y,demension]
	parameters = [data_size,Batch_size,Epochs,norm,split]

	## RUN ##
	Experiment_1(parameters)


	### dataset diagnostics --> grafieken en counts en (nieuwe techieken)
