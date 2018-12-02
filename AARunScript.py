import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
## Import all scripts ##
from AADatasets import import_mnist, import_dogcat, import_melanoom, pre_processing, make_pre_train_classes, get_data, more_data
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
from AAlogic import run_experiment,run_experiment2,run_experiment3,test_script
from LabnotesDoc import doc
# from AnalyseData import AnalyseDataClass, plot_pre_train_result

params = {'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':"imagenet", "extra_data":500, "shuffle": True, "epochs": 50 , "val_size":300, "Batch_size": 10}

try:
	print("Try to import pickle")
	zippy = pickle.load(open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_2x500.p", "rb" ))
	print("succeed to import pickle")
	zippy = list(zippy)
	random.shuffle(zippy)
	x,y = zip(*zippy)
	x = np.array(x)
	y = np.array(y)
	p = True
except:
	print("Failed to import pickle")
	x,y = import_melanoom(params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"])
	x,y = more_data(x,y,params["extra_data"])
	x,y = more_data(x,y,params["extra_data"])
	p = False
	zip_melanoom = zip(x,y)
	pickle.dump( zip_melanoom, open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_2x500.p", "wb" ))
	zippy = list(zip_melanoom)
	random.shuffle(zippy)
	x,y = zip(*zippy)
	x = np.array(x)
	y = np.array(y)

model = make_model(x, y.shape[1], w = params['pretrain'])
H, score, model = train_model(model,x[params["val_size"]:],y[params["val_size"]:],x[:params["val_size"]],y[:params["val_size"]], params["epochs"], params["Batch_size"])

doc(params,{'score':score},H)

#save model to JSON
model_json = model.to_json()
with open(r"C:\Users\Floris\Documents\GitHub\reg_first.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"C:\Users\Floris\Documents\GitHubreg_first.h5")
print("Saved model to disk")





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
	
	# r = run_experiment2('catdog','mela', vgg = False, img_size_x=64, img_size_y=64, demension=1, Batch_size=12, Epochs=20, norm=True, train_size = [500,2500,5000,10000,15000,20000])
	# print(r) 
	# plt.plot(r, [500,2500,5000,10000,15000,20000])
	# plt.title('Dataset len and auc of svm')
	# plt.show()

	# r = run_experiment3('none','mela', img_size_x=224, img_size_y=224, demension=3, Batch_size=12, Epochs=20, norm=False, train_size = [2000])
	
	# r = run_experiment3('none','mela', img_size_x=224, img_size_y=224, demension=3, Batch_size=12, Epochs=20, norm=False, train_size = [2000])

	### LAPNOTES AUTOMATE --> write txt file. 