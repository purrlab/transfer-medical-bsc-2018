

## Import all scripts ##
from AADatasets import import_mnist, import_dogcat, import_melanoom, pre_processing, make_pre_train_classes, get_data
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
# from AnalyseData import AnalyseDataClass, plot_pre_train_result


def run_experiment(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

	x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm, train_size)

	# if x_train != None:
	x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test, none = make_pre_train_classes(y_test)
	# else:
	# 	print("dataset 1 not found")

	# if x_train2 != None:
	x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
	x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
	# else:
	# 	print("dataset 2 not found")

	# if not model:
	model = make_model(x_train, y_train, numb_classes)

	H, score = train_model(model,x_train,y_train,x_test,y_test, Epochs, Batch_size)

	vector = get_feature_vector(model, x_train2, layer = 'fc2')
	vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
	x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
	return x, score

def run_experiment2(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

	x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm)

	# if x_train != None:
	x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test, none = make_pre_train_classes(y_test)
	# else:
	# 	print("dataset 1 not found")

	# if x_train2 != None:
	x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
	x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
	# else:
	# 	print("dataset 2 not found")
	results = []
	for i in range(0,len(train_size)):
		# if not model:
		model = make_model(x_train, y_train, numb_classes)
		H, score = train_model(model,x_train[:train_size[i]],y_train[:train_size[i]],x_test,y_test, Epochs, Batch_size)
		vector = get_feature_vector(model, x_train2, layer = 'fc2')
		vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
		x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
		results.append(x)

	return results

def run_experiment3(name_data,name_data2, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

	x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = get_data(name_data,name_data2 ,img_size_x,img_size_y, norm, color = True)

	# if x_train != None:
	# x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
	# x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
	# y_train,numb_classes = make_pre_train_classes(y_train)
	# y_test, none = make_pre_train_classes(y_test)
	# else:
	# 	print("dataset 1 not found")

	# if x_train2 != None:
	x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
	x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
	# else:
	# 	print("dataset 2 not found")
	results = []
	model = make_model(x_train2, 3, w = 'imagenet')
	# # if not model:
	# model = make_model(x_train, y_train, numb_classes)
	H, score = train_model(model,x_train2,y_train2,x_test2,y_test2, Epochs, Batch_size)
	# vector = get_feature_vector(model, x_train2, layer = 'fc2')
	# vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
	# x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)

	results = score

	return results

def test_script(limit):
	x_train, y_train, x_val, y_val, x_test, y_test = import_mnist(0.10, True, limit)
	x_train = pre_processing(x_train, 28,28,1)
	x_test = pre_processing(x_test, 28,28,1)
	y_train,numb_classes = make_pre_train_classes(y_train)
	y_test, none = make_pre_train_classes(y_test)
	print(x_train.shape)
	print(x_test.shape)
	x = auc_svm(x_train, y_train, x_test, y_test,plot = True)
	return x

# def display_results():
# 	# run some analistic functios
# 	plot_epochs(History)


if __name__ == '__main__':
	# test_script(1000)
	# print(X[1])
	# Test for loop of au
	'''
	auc_list = list()
	for x in range(1,10):
		auc = test_script(100*x)
		print(auc)
	'''

	# test exp 1 en 2

	## experiment selection #######


	# run_experiment('mnist','mnist', limit=100, img_size_x=64, img_size_y=64, demension=1, Batch_size=2, Epochs=1, norm=True, split=0.142857)

	# run_experiment('catdog','mela', vgg = False, img_size_x=64, img_size_y=64, demension=1, Batch_size=12, Epochs=1, norm=True, train_size = 1000) 
	import matplotlib.pyplot as plt
	# r = run_experiment2('catdog','mela', vgg = False, img_size_x=64, img_size_y=64, demension=1, Batch_size=12, Epochs=20, norm=True, train_size = [500,2500,5000,10000,15000,20000])
	# print(r) 
	# plt.plot(r, [500,2500,5000,10000,15000,20000])
	# plt.title('Dataset len and auc of svm')
	# plt.show()

	r = run_experiment3('none','mela', img_size_x=224, img_size_y=224, demension=3, Batch_size=12, Epochs=20, norm=False, train_size = [2000])
	print(r) 
	plt.plot(r, [500,2500,5000,10000,15000,20000])
	plt.title('Dataset len and auc of svm')
	plt.show()


	### LAPNOTES AUTOMATE --> write txt file. 