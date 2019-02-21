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


def test_all():

    print("\n")

    params = {"Data":'CatDog',
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