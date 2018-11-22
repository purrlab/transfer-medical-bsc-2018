

## Import all scripts ##
from Datasets import import_mnist, import_dogcat, pre_processing, make_pre_train_classes
from PreTrain import make_model,train_model, config_desktop
from TransferLearn import get_feature_vector, preform_svm
# from AnalyseData import AnalyseDataClass, plot_pre_train_result


def run_experiment(name_data, limit, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, split):
	global desktop
	#run a set of parameters over the variables (make loopable)
	if name_data == 'mnist':
		x_train, y_train, x_test, y_test = import_mnist(split, norm)
	elif name_data == 'CatDog':
		cat = ['Dog', 'Cat']
		if desktop:
			DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
		else:
			DIR = r"C:\Users\s147057\Documents\Python scripts\PetImages"

		x_train, y_train, x_test, y_test = import_dogcat(DIR,cat,img_size_x,img_size_y, split, norm, limit = 100, color = False)
	else:
		print('Choose another dataset')

	x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	y_train,numb_classes = make_pre_train_classes(y_train)
	# Pre train
	model = make_model(x_train, y_train, numb_classes)
	
	if limit:
		H = model.fit(x_train[:limit],y_train[:limit], batch_size=Batch_size, epochs=Epochs, validation_split=0.1)
		vector = get_feature_vector(model, x_test, limit, layer = 'fc2')
		clf = preform_svm(vector, y_test[:limit])
	else:
		H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_split=0.1)
		vector = get_feature_vector(model, x_test, limit = None, layer = 'fc2')
		clf = preform_svm(vector, y_test)
	

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

	# run_experiment('mnist', limit=100, img_size_x=64, img_size_y=64, demension=1, Batch_size=2, Epochs=1, norm=True, split=0.142857)

	run_experiment('CatDog', limit=200, img_size_x=100, img_size_y=100, demension=1, Batch_size=2, Epochs=1, norm=True, split=0.10)
