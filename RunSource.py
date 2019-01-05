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


def run(params):
    config_desktop()
    x,y = get_data(params)
    
    x_test,y_test,x,y = val_split(x,y, params["test_size"])
    x_val,y_val,x,y = val_split(x,y, params["val_size"])

    model = make_model(x, y, params)
    # weights = determen_weights(y)
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params)
    
    # if params["style"] == 'FT':
    #     model = make_model(x, y, params)
    #     weights = determen_weights(y)
    #     H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params, weights)
    #     predictions = get_feature_vector(model, x, layer = 'fc2')
    #     predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
    #     score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
    #     results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    # elif params["style"] =='SVM':
    #     predictions = x.flatten().reshape(len(x), 224*224)
    #     predictions_test = x_test.flatten().reshape(len(x_test), 224*224)
    #     score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
    #     results = {'score':score,'data_name':params["data_name"],'method':params["model"],'style':params["style"]}
    #     H=None

    #save model to JSON
    model_json = model.to_json()
    # with open(f"{params['model_path']}{params['epochs']}_{params['Data']}.json", "w") as json_file:
    #     json_file.write(model_json)
    with open(f"NONE_{params['Data']}{params['data_name']}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights(f"{params['model_path']}{params['epochs']}_{params['Data']}_Weights.h5")
    # print("Saved model to disk")
    model.save_weights(f"NONE_{params['Data']}{params['data_name']}_Weights.h5")
    print("Saved model to disk")

    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    doc(params,results,H,params["doc_path"])