

## Import all scripts ##
from AADatasets import import_mnist, import_dogcat, import_melanoom, pre_processing, make_pre_train_classes
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
# from AnalyseData import AnalyseDataClass, plot_pre_train_result


def run_experiment(name_data,name_data2, limit, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, split):
	global desktop
	#run a set of parameters over the variables (make loopable)
	if 'mnist' == name_data:
		x_train, y_train, x_test, y_test = import_mnist(split, norm, limit)
	elif 'CatDog' == name_data:
		cat = ['Dog', 'Cat']
		if desktop:
			DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		else:
			DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"
		x_train, y_train, x_test, y_test = import_dogcat(DIR,cat,img_size_x,img_size_y, split, norm, limit, color = False)

	else:
		print('Choose another dataset')

	if 'mnist' == name_data2:
		x_train2, y_train2, x_test2, y_test2 = import_mnist(split, norm, limit = 1000)
	elif 'CatDog' == name_data2:
		cat = ['Dog', 'Cat']
		if desktop:
			DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		else:
			DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"
		x_train2, y_train2, x_test2, y_test2 = import_dogcat(DIR,cat,img_size_x,img_size_y, split, norm, limit = 1000, color = False)

	else:
		print('Choose another dataset')

	x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
	x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test, none = make_pre_train_classes(y_test)
	y_train2, none = make_pre_train_classes(y_train2)
	y_test2, none = make_pre_train_classes(y_test2)
	# Pre train
	model = make_model(x_train, y_train, numb_classes)
	model.save('CatAndDog.model')

	H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_test, y_test))
	vector = get_feature_vector(model, x_train2, layer = 'fc2')
	vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
	auc_svm(vector, y_train2, vector_val, y_test2)
	

def test_script(limit):
	x_train, y_train, x_test, y_test = import_mnist(0.10, True, limit)
	x_train = pre_processing(x_train, 28,28,1)
	x_test = pre_processing(x_test, 28,28,1)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test, none = make_pre_train_classes(y_test)
	print(x_train.shape)
	print(x_test.shape)
	x = auc_svm(x_train, y_train, x_test, y_test,plot = False)
	return x

# def display_results():
# 	# run some analistic functios
# 	plot_epochs(History)


if __name__ == '__main__':
	x,y,X,Y = import_melanoom(r"C:\Users\Floris\Documents\Python Scripts\ISIC-2017_Training_Data", 300, 300, 0.2, True, limit = 100, color = False)
	print(X[1])
	# Test for loop of auc
	'''
	auc_list = list()
	for x in range(1,10):
		auc = test_script(100*x)
		print(auc)
	'''

	# test exp 1 en 2
	''' 
	while True:
		A = input("Are you on a desktop? (y/n)   \n")
		if A == 'y':
			config_desktop()
			desktop = True
			break
		elif A == 'n':
			desktop = False
			break

	## experiment selection #######

	# run_experiment('mnist','mnist', limit=100, img_size_x=64, img_size_y=64, demension=1, Batch_size=2, Epochs=1, norm=True, split=0.142857)

	run_experiment('CatDog','mnist', limit=10000, img_size_x=64, img_size_y=64, demension=1, Batch_size=10, Epochs=20, norm=True, split=0.20) 
	''' 