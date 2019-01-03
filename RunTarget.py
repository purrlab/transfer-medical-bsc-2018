import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import tensorflow as tf
from AADatasets import *
from AAPreTrain import *
from AATransferLearn import *
from AAlogic import *
from LabnotesDoc import *

def run_target(params):
    x,y = get_data(params)

    # if params['norm']:
    #     x = x/x.max()

    x_test,y_test,x,y = val_split(x,y, params["test_size"])
    x_val,y_val,x,y = val_split(x,y, params["val_size"])
    
    config_desktop()

    model = make_model(x, y, params)

    if params["style"] == 'FT':
        weights = determen_weights(y)
        H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params, weights)
        predictions = get_feature_vector(model, x, layer = 'fc2')
        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
        results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    elif params["style"] =='SVM':
        predictions = get_feature_vector(model, x, layer = 'fc2')
        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
        results = {'score':score,'data_name':params["data_name"],'method':params["model"],'style':params["style"]}
        H=None

    doc(params,results,H,params["doc_path"])