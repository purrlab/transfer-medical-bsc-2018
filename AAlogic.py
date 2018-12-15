import os
import cv2
import sys
import time
import pickle
import pandas
import random
import sklearn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import interp
from sklearn import metrics
from itertools import cycle
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score

from RunSource import *
from RunTarget import *
from AADatasets import *
from AAPreTrain import *
from LabnotesDoc import *
# from AARunScript import *
# from AAAnalyseData import *
from AATransferLearn import *


def test_all(params):

    print("\n")

    params_test = {"Data":'CatDog',
            'file_path':r"C:\Miniset",
            'pickle_path':r"C:\pickles\CatDog_mini",
            'model_path':r"C:\models\Epochs_",
            'doc_path':r"C:\Users\Flori\Documents\GitHub\t",
            'img_size_x':224,
            'img_size_y':224,
            'norm':True,
            'color':True,
            'pretrain':None,
            "equal_data":False, 
            "shuffle": True, 
            "epochs": 5 , 
            "val_size":3,
            "test_size":5, 
            "Batch_size": 1
            }

    print("File path of data:")
    print(os.path.isdir(params["file_path"]))
    print("File path of pickle:")
    print(os.path.isdir(params["pickle_path"][:-len(params["Data"])]))
    print("File path of mdoel:")
    print(os.path.isdir(params["model_path"]))
    print("File path of Documentation:")
    print(os.path.isdir(params["doc_path"][:-1]))
    print("\n")
    print("Size of Image:")
    print(img_size_x > 223 and img_size_y > 223)
    print("\n")
    print("Testing functions:")
    x,y,x_val,y_val,x_test,y_test = get_data(params_test)
    if x == None or y == None or x_val == None or y_val == None or x_test == None or y_test == None:
        print("get_data = OK")
    else:
        print("get_data = OK")

    model = make_model(x, y, w = 'imagenet')
    print("make_model = OK")
    weights = determen_weights(y)
    print("determen_weights = OK")
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"],weights)
    print("train_model = OK")
    predictions = get_feature_vector(model, x, layer = 'fc2')
    predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
    print("get_feature_vector = OK")
    score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
    print("auc_svm = OK")
    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss'],'data_name':data_name,'method':method,'style':style}

    doc(params,results,H,params["doc_path"])
    print("doc = OK")

# def run_experiment(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

#     x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm, train_size)

#     # if x_train != None:
#     x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
#     x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
#     y_train,numb_classes = make_pre_train_classes(y_train)
#     y_test, none = make_pre_train_classes(y_test)
#     # else:
#     #   print("dataset 1 not found")

#     # if x_train2 != None:
#     x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
#     x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
#     # else:
#     #   print("dataset 2 not found")

#     # if not model:
#     model = make_model(x_train, y_train, numb_classes)

#     H, score = train_model(model,x_train,y_train,x_test,y_test, Epochs, Batch_size)

#     vector = get_feature_vector(model, x_train2, layer = 'fc2')
#     vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
#     x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
#     return x, score

# def run_experiment2(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

#     x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm)

#     # if x_train != None:
#     x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
#     x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
#     y_train,numb_classes = make_pre_train_classes(y_train)
#     y_test, none = make_pre_train_classes(y_test)
#     # else:
#     #   print("dataset 1 not found")

#     # if x_train2 != None:
#     x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
#     x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
#     # else:
#     #   print("dataset 2 not found")
#     results = []
#     for i in range(0,len(train_size)):
#         # if not model:
#         model = make_model(x_train, y_train, numb_classes)
#         H, score = train_model(model,x_train[:train_size[i]],y_train[:train_size[i]],x_test,y_test, Epochs, Batch_size)
#         vector = get_feature_vector(model, x_train2, layer = 'fc2')
#         vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
#         x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
#         results.append(x)

#     return results

# def run_experiment3(name_data,name_data2, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

#     x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = get_data(name_data,name_data2 ,img_size_x,img_size_y, norm, color = True)

#     # if x_train != None:
#     # x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
#     # x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
#     # y_train,numb_classes = make_pre_train_classes(y_train)
#     # y_test, none = make_pre_train_classes(y_test)
#     # else:
#     #   print("dataset 1 not found")

#     # if x_train2 != None:
#     x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
#     x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
#     # else:
#     #   print("dataset 2 not found")
#     results = []
#     model = make_model(x_train2, 3, w = 'imagenet')
#     # # if not model:
#     # model = make_model(x_train, y_train, numb_classes)
#     H, score = train_model(model,x_train2,y_train2,x_test2,y_test2, Epochs, Batch_size)
#     # vector = get_feature_vector(model, x_train2, layer = 'fc2')
#     # vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
#     # x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)

#     results = score

#     return results

# def test_script():
#     params = {'img_size_x':48,'img_size_y':48,'norm':False,'color':False, 'pretrain':None, "equal_data":True, "shuffle": True, "epochs": 3 , "val_size":10,"test_size":20, "Batch_size": 2}
#     file_path = r"C:\Miniset"
#     doc_path =  r"C:\Users\Flori\Documents\GitHub\t"
#     #config_desktop()

#     x,y = import_dogcat(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"])
#     if params["equal_data"] == True:
#         x,y = equal_data_run(y,x)

#     zip_melanoom = zip(x,y)
#     pickle.dump( zip_melanoom, open( r"C:\Users\Flori\Documents\GitHub\pickles\test.p", "wb" ))
#     zippy = list(zip_melanoom)
#     random.shuffle(zippy)
#     x,y = zip(*zippy)
#     x = np.array(x)
#     y = np.array(y)
#     x_test,y_test,x,y = val_split(x,y, params["test_size"])
#     x_val,y_val,x,y = val_split(x,y, params["val_size"])



#     model = make_model(x, y, w = params['pretrain'])
#     H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])

#     results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

#     doc(params,results,H, doc_path)

# def menegola_plane():
#     params = {'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':None, "equal_data":True, "shuffle": True, "epochs": 50 , "val_size":200,"test_size":300, "Batch_size": 16, "mela_class":"three"}
#     file_path = r"C:\ISIC\ISIC-2017_Training_Data"
#     config_desktop()

#     try:
#         print("Try to import pickle")
#         zippy = pickle.load(open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_three.p", "rb" ))
#         print("succeed to import pickle")
#         zippy = list(zippy)
#         random.shuffle(zippy)
#         x,y = zip(*zippy)
#         x = np.array(x)
#         y = np.array(y)
#         x_test,y_test,x,y = val_split(x,y, params["test_size"])
#         x_val,y_val,x,y = val_split(x,y, params["val_size"])

#     except:
#         print("Failed to import pickle")    
#         x,y = import_melanoom(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"],classes = params["mela_class"])
#         if params["equal_data"] == True:
#             x,y = equal_data_run(y,x)

#         zip_melanoom = zip(x,y)
#         pickle.dump( zip_melanoom, open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_three.p", "wb" ))
#         zippy = list(zip_melanoom)
#         random.shuffle(zippy)
#         x,y = zip(*zippy)
#         x = np.array(x)
#         y = np.array(y)
#         x_test,y_test,x,y = val_split(x,y, params["test_size"])
#         x_val,y_val,x,y = val_split(x,y, params["val_size"])



#     model = make_model(x, y, w = params['pretrain'])
#     H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])

#     results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

#     doc(params,results,H)

#     #save model to JSON
#     model_json = model.to_json()
#     with open(r"D:\models\Epochs_50_melanoom_equal.json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights(r"D:\models\weights\Epochs_50_melanoom_equal.h5")
#     print("Saved model to disk")