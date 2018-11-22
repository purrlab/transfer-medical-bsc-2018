

## Import all scripts ##
from 2Datasets import import_mnist, import_dogcat, pre_processing, make_pre_train_classes
from 3PreTrain import make_model,train_model, config_desktop
from 4TransferLearn import get_feature_vector, preform_svm
# from AnalyseData import AnalyseDataClass, plot_pre_train_result


def run_experiment(name_data,name_data2, limit, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, split):
	global desktop
	#run a set of parameters over the variables (make loopable)
	if 'mnist' == name_data:
		x_train, y_train, x_test, y_test = import_mnist(split, norm)
	elif 'CatDog' == name_data:
		cat = ['Dog', 'Cat']
		if desktop:
			DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		else:
			DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"

		x_train, y_train, x_test, y_test = import_dogcat(DIR,cat,img_size_x,img_size_y, split, norm, limit = 100, color = False)
	else:
		print('Choose another dataset')

	if 'mnist' == name_data2:
		x_train2, y_train2, x_test2, y_test2 = import_mnist(split, norm)
	elif 'CatDog' == name_data2:
		cat = ['Dog', 'Cat']
		if desktop:
			DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		else:
			DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"

		x_train2, y_train2, x_test2, y_test2 = import_dogcat(DIR,cat,img_size_x,img_size_y, split, norm, limit = 100, color = False)
	else:
		print('Choose another dataset')
	x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
	x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test,numb_classes = make_pre_train_classes(y_test)
	# Pre train
	model = make_model(x_train, y_train, numb_classes)
	

	H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_test, y_test))
	vector = get_feature_vector(model, x_train2, layer = 'fc2')
	vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
	clf = preform_svm(vector, y_train2, vector_val, y_test2)
	

def test_script():
	# run this for quick test of function (super small data set --> cat dog 20 images)
	pass

# def display_results():
# 	# run some analistic functios
# 	plot_epochs(History)


if __name__ == '__main__':
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

	run_experiment('CatDog','mnist', limit=200, img_size_x=100, img_size_y=100, demension=1, Batch_size=2, Epochs=1, norm=True, split=0.10)
