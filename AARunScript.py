import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
## Import all scripts ##
from AADatasets import import_mnist, import_dogcat, import_melanoom, pre_processing, make_pre_train_classes, get_data, more_data, equal_data_run
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
from AAlogic import run_experiment,run_experiment2,run_experiment3,test_script
from LabnotesDoc import doc
# from AnalyseData import AnalyseDataClass, plot_pre_train_result

# params = {'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':"imagenet", "extra_data":500, "shuffle": True, "epochs": 50 , "val_size":300, "Batch_size": 10}
# file_path = r"C:\Users\Floris\Documents\Python Scripts\ISIC-2017_Training_Data"

# try:
#     print("Try to import pickle")
#     zippy = pickle.load(open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_2x500.p", "rb" ))
#     print("succeed to import pickle")
#     zippy = list(zippy)
#     random.shuffle(zippy)
#     x,y = zip(*zippy)
#     x = np.array(x)
#     y = np.array(y)
#     p = True
# except:
#     print("Failed to import pickle")
#     x,y = import_melanoom(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"])
#     x,y = more_data(x,y,params["extra_data"])
#     x,y = more_data(x,y,params["extra_data"])
#     p = False
#     zip_melanoom = zip(x,y)
#     pickle.dump( zip_melanoom, open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_2x500.p", "wb" ))
#     zippy = list(zip_melanoom)
#     random.shuffle(zippy)
#     x,y = zip(*zippy)
#     x = np.array(x)
#     y = np.array(y)

# model = make_model(x, y.shape[1], w = params['pretrain'])
# H, score, model = train_model(model,x[params["val_size"]:],y[params["val_size"]:],x[:params["val_size"]],y[:params["val_size"]], params["epochs"], params["Batch_size"])

# doc(params,{'score':score},H)

# #save model to JSON
# model_json = model.to_json()
# with open(r"C:\Users\Floris\Documents\GitHub\reg_first.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(r"C:\Users\Floris\Documents\GitHubreg_first.h5")
# print("Saved model to disk")

params = {'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':None, "equal_data":True, "shuffle": True, "epochs": 50 , "val_size":400, "Batch_size": 32}
file_path = r"D:\ISIC\ISIC-2017_Training_Data"
config_desktop()

try:
    print("Try to import pickle")
    zippy = pickle.load(open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_EQUAL.p", "rb" ))
    print("succeed to import pickle")
    zippy = list(zippy)
    random.shuffle(zippy)
    x,y = zip(*zippy)
    x = np.array(x)
    y = np.array(y)

except:
    print("Failed to import pickle")
    x,y = import_melanoom(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"])
    if params["equal_data"] == True:
        x,y = equal_data_run(y,x)

    zip_melanoom = zip(x,y)
    pickle.dump( zip_melanoom, open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_EQUAL.p", "wb" ))
    zippy = list(zip_melanoom)
    random.shuffle(zippy)
    x,y = zip(*zippy)
    x = np.array(x)
    y = np.array(y)

model = make_model(x, y, w = params['pretrain'])
H, score, model = train_model(model,x[params["val_size"]:],y[params["val_size"]:],x[:params["val_size"]],y[:params["val_size"]], params["epochs"], params["Batch_size"])

result = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

doc(params,results,H)

#save model to JSON
model_json = model.to_json()
with open(r"C:\Users\Floris\Documents\GitHub\reg_first.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"C:\Users\Floris\Documents\GitHubreg_first.h5")
print("Saved model to disk")


## analitics ##