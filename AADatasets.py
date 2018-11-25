
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics
import os
import random
import pandas

def import_melanoom(img_size_x,img_size_y, norm, train_size, color = False):
	try: 
		DIR = r"C:\Users\Floris\Documents\Python Scripts\ISIC-2017_Training_Data"
		os.listdir(DIR)
		data_dir = DIR
		print('Desktop detected')
	except:
		DIR = r"C:\Users\s147057\Documents\Python Scripts\ISIC-2017_Training_Data"
		print('Laptop detected')
		data_dir = DIR

	### get data ground truth ###
	df = pandas.read_csv(r"C:\Users\S147057\Documents\Python scripts\ISIC-2017_Training_Part3_GroundTruth.csv")
	df = df.set_index("image_id")

	# data_dir=r"C:\Users\s147057\Documents\Python Scripts\ISIC-2017_Training_Data"
	training_data = []
	target_data = []

	L = os.listdir(data_dir)
	L.reverse() # omdraaien van de lijst zorgt voor meer diversiteit van classes. 
	for img in L[:-1]:
		if 'superpixels' in img:
			continue
		try:
			
			class_num = df.loc[img[0:-4],:]
			if class_num[0] == 1:
				class_num = [0,1,0]
			elif class_num[1] == 1:
				class_num = [0,0,1]
			else:
				class_num = [1,0,0]

			if color:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
				D = 3
				new_array = cv2.resize(img_array,(img_size_x, img_size_y))
			else:
				img_array = cv2.imread(os.path.join(data_dir,img), cv2.IMREAD_GRAYSCALE)
				D = 1
				new_array = cv2.resize(img_array,(img_size_x, img_size_y))
			new_array = cv2.resize(img_array,(img_size_x, img_size_y))
			training_data.append(new_array)
			target_data.append(class_num)
		except Exception as e:
			pass

	x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
	y = np.array(target_data).reshape(-1,3)

	if type(norm) != bool:
		print("please enter 'boolean' for norm(alization)")

	test_split = 0.25
	val_split = 0.1

	spl = int(test_split*len(x))
	X = x[spl:]
	Y = y[spl:]
	x_test  = x[:spl]
	y_test  = y[:spl]

	spl2 = int(val_split*len(X))
	x_val = x[:spl2]
	y_val = y[:spl2]
	x_train = X[spl2:]
	y_train = Y[spl2:]

	if norm:
		x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

	print(f"This melanoom dataset contains the following: \nTotal length Dataset = {len(x)} \nTotal length train set = {len(x_train)} \nTotal length val set = {len(x_val)} \nTotal length test set= {len(x_test)}")
	return x_train, y_train, x_val, y_val, x_test, y_test

def import_dogcat(img_size_x,img_size_y, norm, train_size, color = False):
	try: 
		DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		cat = list(os.listdir(DIR))
		data_dir = DIR
		print('Desktop detected')
	except:
		DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"
		cat = list(os.listdir(DIR))
		print('Laptop detected')
		data_dir = DIR

	training_data = list()
	training_class = list()

	for category in cat:
		path = os.path.join(data_dir, category)
		class_num = cat.index(category)
		path = os.path.join(data_dir, category)
		for img in os.listdir(path):
			try:
				if color:
					img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
					D = 3
					new_array = cv2.resize(img_array,(img_size_x, img_size_y))
				else:
					img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
					D = 1
					new_array = cv2.resize(img_array,(img_size_x, img_size_y))
				training_data.append(new_array)
				training_class.append(class_num)
			except Exception as e:
				pass
	zip_list = list(zip(training_data,training_class))
	random.shuffle(zip_list)
	training_data,training_class = zip(*zip_list)
	x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
	y = np.array(training_class).reshape(-1,1)

	if type(norm) != bool:
		print("please enter 'boolean' for norm(alization)")

	test_split = 0.25
	val_split = 0.1

	spl = int(test_split*len(x))
	X = x[spl:]
	Y = y[spl:]
	x_test  = x[:spl]
	y_test  = y[:spl]

	spl2 = int(val_split*len(X))
	x_val = x[:spl2]
	y_val = y[:spl2]
	x_train = X[spl2:]
	y_train = Y[spl2:]

	if norm:
		x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0



	print(f"This Dog_Cat dataset contains the following: \nTotal length Dataset = {len(x)} \nTotal length train set = {len(x_train)} \nTotal length val set = {len(x_val)} \nTotal length test set= {len(x_test)}")
	return x_train, y_train, x_val, y_val, x_test, y_test

def import_mnist(split, norm, limit = None):
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

	if limit > len(y_train):
		print("Limit is to high, limit is turned off")
		limit = None

	if norm:
		x_train, x_test = x_train / 255.0, x_test / 255.0

	if not limit:
		limit = len(y_train)

	return x_train, y_train, x_val, y_val, x_test, y_test


def train_val_test():
	#make consistent test set, and variable train sets
	pass

def pre_processing(X,img_size_x,img_size_y,demension):
	#def transform_img_list(X,img_size_x,img_size_y,demension):
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
	
	clas = list()
	if not numb_classes:
		numb_classes = int(max(list(Y))-min(list(Y))+1)

	for label in list(Y):
	    new_list = numb_classes*[0]
	    new_list[int(label)] = 1
	    clas.append(new_list)
	clas_np = np.array(clas).reshape(-1,numb_classes)
	return clas_np, numb_classes

def get_data(name_data,name_data2, vgg,img_size_x,img_size_y, norm, train_size, color = False):
	if name_data == 'mela':
		x_train, y_train, x_val, y_val, x_test, y_test = import_melanoom(img_size_x,img_size_y, norm, train_size, color)
		model = None
	elif name_data == 'catdog':
		x_train, y_train, x_val, y_val, x_test, y_test = import_dogcat(img_size_x,img_size_y, norm, train_size, color)
		model = None
	elif vgg:
		x_train, y_train, x_val, y_val, x_test, y_test = None,None,None,None,None,None
		model = tf.keras.applications.VGG16(weights=None,input_shape = (x[0].shape), include_top=True, classes=numb_classes) #top??
		adm = tf.keras.optimizers.SGD(lr=0.008, momentum=0.0, decay=0.0, nesterov=False)
		model.compile(loss='categorical_crossentropy', optimizer=adm)#, metrics=['accuracy'])  #'auc'
	else:
		print('There is no data set with that name')		

	if name_data2 == 'mela':
		x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = import_melanoom(img_size_x,img_size_y, norm, train_size, color)
	elif name_data2 == 'catdog':
		x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = import_dogcat(img_size_x,img_size_y, norm, train_size, color)
	else:
		print('Warning: No second set')
		x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = None,None,None,None,None,None
	print('Train, Val and test sets created')
	return x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model