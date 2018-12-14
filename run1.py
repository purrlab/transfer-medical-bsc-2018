from Run import RUN
import sys
arg = sys.argv[1]

model = ["imagenet","kaggleDR"]
data = ['two','three','two_combined']
style = ['FT', 'SVM']

x = int(arg[0])
y = int(arg[1])
z = int(arg[2])
r = int(arg[3])

RUN(model[x],data[y],style[z],r)

# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import time
# import tensorflow as tf
# from AADatasets import import_mnist, import_dogcat, import_melanoom,import_kaggleDR, pre_processing, make_pre_train_classes, get_data, more_data, equal_data_run, val_split
# from AAPreTrain import make_model,train_model, config_desktop
# from AATransferLearn import get_feature_vector, preform_svm, auc_svm
# from AAlogic import test_script, menegola_plane
# from LabnotesDoc import doc
# from numpy import float16 as NPfloat16
# from numpy import asarray as NPasarray

# style = "FT" #"SVM"
# data_name = "two_combined" #"two" "three"
# method = "imagenet" #"kaggleDR"

# params = {"Data":'ISIC',"RandomSeed":2,'img_size_x':224,'img_size_y':224,'norm':True,'color':False, 'pretrain':None, "equal_data":False, "shuffle": True, "epochs": 60 , "val_size":200,"test_size":300, "Batch_size": 16}
# file_path = r"C:\ISIC"
# pickle_path = r"C:\pickles\save_melanoom_color_"
# model_path = r"C:\models\Epochs_"
# doc_path =  r"C:\Users\Flori\Documents\GitHub\t"

# random.seed(params["RandomSeed"])


# print("Try to import pickle")
# zippy = pickle.load(open( f"{pickle_path}{data_name}.p", "rb" ))
# print("succeed to import pickle")
# zippy = list(zippy)
# random.shuffle(zippy)
# x,y = zip(*zippy)
# x = np.array(x)
# y = np.array(y)
# if params["norm"] == True:
#     x = x/255
#     x = NPasarray(x, dtype=NPfloat16)
# if not params["equal_data"]:
#     if data_name != "two":
#         x = x[:2000]
#         y = y[:2000]
#     else:
#         x = x[:1746]
#         y = y[:1746]
# x_test,y_test,x,y = val_split(x,y, params["test_size"])
# x_val,y_val,x,y = val_split(x,y, params["val_size"])


# config_desktop()

# if method == "imagenet":
#     model = make_model(x, y, w = 'imagenet')
#     print("model made")
# elif method == "kaggleDR":
#     with open(r"C:\models\Epochs_50_kaggleDR.json", 'r') as json_file:
#         loaded_model_json = json_file.read()
#     json_file.close()
#     model = tf.keras.models.model_from_json(loaded_model_json)
#     #Load weights into new model
#     model.load_weights(r"C:\models\Epochs_50_kaggleDR_Weights.h5")
#     print("Loaded model from disk")
#     for layer in model.layers[:-5]:
#         layer.trainable = False
#     fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(model.output)
#     model  = tf.keras.models.Model(inputs=model.input, outputs=fine_tune)
#     opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.01, decay=0.0, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# if style == 'FT':
#     H, s, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])
#     predictions = get_feature_vector(model, x, layer = 'fc2')
#     predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
#     score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
#     results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss'],'data_name':data_name,'method':method,'style':style}

# elif style =='SVM':
#     predictions = get_feature_vector(model, x, layer = 'fc2')
#     predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
#     score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
#     results = {'score':score,'data_name':data_name,'method':method,'style':style}
#     H=None

# doc(params,results,H,doc_path)
# print(method)
# print(style)
# print(data_name)