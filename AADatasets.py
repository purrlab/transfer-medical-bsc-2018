
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

def import_melanoom(data_dir,img_size_x,img_size_y, split, norm, limit = False, color = False):
	### get data ground truth ###
	df = pandas.read_csv(r"C:\Users\Floris\Documents\Python scripts\ISIC-2017_Training_Part3_GroundTruth.csv")
	df = df.set_index("image_id")

	# data_dir=r"C:\Users\s147057\Documents\Python Scripts\ISIC-2017_Training_Data"
	training_data = []
	target_data = []
	x = 0
	D = 1
	for img in os.listdir(data_dir)[1:]:
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
		x+=1
		if limit:
			if x > limit: #len(df):
				break
	x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
	y = np.array(target_data).reshape(-1,1)

	if type(split) != float:
		print("please enter 'float' for split")
	if type(norm) != bool:
		print("please enter 'boolean' for norm(alization)")

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
	print('Classes are in binary')
	return x_train, y_train, x_test, y_test

def import_dogcat(data_dir,cat,img_size_x,img_size_y, split, norm, limit = False, color = False):
	training_data = list()
	training_class = list()

	for category in cat:
		path = os.path.join(data_dir, category)
		class_num = cat.index(category)
		path = os.path.join(data_dir, category)
		if limit:
			limit = limit/2
		else:
			limit = len(os.listdir(path))

		for img in os.listdir(path)[:limit]:
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

	if type(split) != float:
		print("please enter 'float' for split")
	if type(norm) != bool:
		print("please enter 'boolean' for norm(alization)")

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

	return x_train[:limit], y_train[:limit], x_test, y_test


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

def main():
	# small example to test script
	pass
if __name__ == '__main__':
	main() 