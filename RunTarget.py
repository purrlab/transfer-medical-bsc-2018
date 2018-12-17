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

    with open(params["model_path"], 'r') as json_file:
        loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    #Load weights into new model
    loaded_model.load_weights(f"{params['model_path'][:-5]}_Weights.h5")
    print("Loaded model from disk")

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

                    if method == "imagenet":
                        model = make_model(x, y, w = 'imagenet')

                    elif method == "kaggleDR":
                        model = loaded_model
                        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(model.output)
                        model  = tf.keras.models.Model(inputs=model.input, outputs=fine_tune)
                        opt = tf.keras.optimizers.SGD(lr=0.0009, momentum=0.01, decay=0.0, nesterov=True)
                        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

                    if style == 'FT':
                        H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])
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