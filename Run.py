import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import tensorflow as tf
from AADatasets import import_mnist, import_dogcat, import_melanoom,import_kaggleDR, pre_processing, make_pre_train_classes, get_data, more_data, equal_data_run, val_split
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
from AAlogic import test_script, menegola_plane
from LabnotesDoc import doc

def RUN(method,data_name,style,seed_num):
    with open(r"C:\models\Epochs_50_kaggleDR.json", 'r') as json_file:
        loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    #Load weights into new model
    loaded_model.load_weights(r"C:\models\Epochs_50_kaggleDR_Weights.h5")
    print("Loaded model from disk")



    params = {"Data":'ISIC',"RandomSeed":seed_num,'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':None, "equal_data":False, "shuffle": True, "epochs": 50 , "val_size":300,"test_size":400, "Batch_size": 16}
    file_path = r"C:\ISIC"
    pickle_path = r"C:\pickles\save_melanoom_color_"
    model_path = r"C:\models\Epochs_"
    doc_path =  r"C:\Users\Flori\Documents\GitHub\t"

    s = list('a')
    d = list('a')
    m = list('a')
    s[0] = style
    d[0] = data_name
    m[0] = method

    random.seed(params["RandomSeed"])

    count=0
    super_script = True
    if super_script == True:
        for style in s:
            for data_name in d: #'two',,'two_combined']:  "imagenet", 'two' ,,'two_combined'
                try:
                    print("Try to import pickle")
                    zippy = pickle.load(open( f"{pickle_path}{data_name}.p", "rb" ))
                    print("succeed to import pickle")
                    zippy = list(zippy)
                    random.shuffle(zippy)
                    x,y = zip(*zippy)
                    x = np.array(x)
                    y = np.array(y)
                    x_test,y_test,x,y = val_split(x,y, params["test_size"])
                    x_val,y_val,x,y = val_split(x,y, params["val_size"])
                except:
                    print("Failed to import pickle")

                for method in m: #"imagenet",
                    config_desktop()
                    count +=1
                    if count ==0: ## if you want to cancel numbers
                        continue
                    if method == "imagenet":
                        model = make_model(x, y, w = 'imagenet')
                    elif method == "kaggleDR":
                        model = loaded_model
                        # for layer in model.layers[:-5]:
                        #     layer.trainable = False
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
                        results = {'score':score,'data_name':data_name,'method':method,'style':style}
                        H=None

                    doc(params,results,H,doc_path)
                    print(method)
                    print(style)
                    print(data_name)