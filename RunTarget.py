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


    s = list('a')
    d = list('a')
    m = list('a')
    s[0] = params["style"]
    d[0] = params["data_name"]
    m[0] = params["model"]

    super_script = True
    if super_script == True:
        for style in s:
            for data_name in d:
                x,y,x_val,y_val,x_test,y_test = get_data(params)
                for method in m:
                    config_desktop()


                    model = make_model(x, y, params)

                    if style == 'FT':
                        weights = determen_weights(y)
                        H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"],weights)
                        predictions = get_feature_vector(model, x, layer = 'fc2')
                        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
                        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
                        results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss'],'data_name':data_name,'method':method,'style':style}

                    elif style =='SVM':
                        predictions = get_feature_vector(model, x, layer = 'fc2')
                        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
                        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
                        results = {'score':score,'data_name':params["data_name"],'method':method,'style':style}
                        H=None

                    doc(params,results,H,params["doc_path"])