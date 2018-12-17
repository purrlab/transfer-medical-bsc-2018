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
    x,y,x_val,y_val,x_test,y_test = get_data(params)
    # x,y,x_val,y_val,x_test,y_test =x[:1000],y[:1000],x_val[:1000],y_val[:1000],x_test[:1000],y_test[:1000] #SMALL TEST
    # x, y = equal_data_run(y,x)

    model = make_model(x, y, w = params['pretrain'])
    weights = determen_weights(y)
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"], weights)

    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss'],'weights':weights}

    doc(params,results,H,params["doc_path"])

    #save model to JSON
    model_json = model.to_json()
    with open(f"{params['model_path']}{params['epochs']}_{params['Data']}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{params['model_path']}{params['epochs']}_{params['Data']}_Weights.h5")
    print("Saved model to disk")