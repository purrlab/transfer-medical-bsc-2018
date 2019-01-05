    
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sklearn
from sklearn import datasets, svm, metrics
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def make_model(x, y,params):

    if params['model'] == 'imagenet':
        classes = 1000
        vgg_conv = tf.keras.applications.VGG16(weights=params["model"],input_shape = (x[0].shape), include_top=True, classes=classes)
        fine_tune = tf.keras.layers.Dense(500, activation='relu')(vgg_conv.output)
        fine_tune = tf.keras.layers.Dense(350, activation='relu')(fine_tune)
        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(fine_tune)
        vgg_conv  = tf.keras.models.Model(inputs=vgg_conv.input, outputs=fine_tune)

    elif params['model'] == "None":
        c = int(y.shape[1])
        vgg_conv = tf.keras.applications.VGG16(weights=None,input_shape = (x[0].shape), include_top=True, classes=c)        

    else:
        print(params['model_path'][params['model']])
        with open(params['model_path'][params['model']], 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        #Load weights into new model
        loaded_model.load_weights(f"{params['model_path'][params['model']][:-5]}_Weights.h5")
        print("Loaded model from disk")
        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(loaded_model.output)
        vgg_conv  = tf.keras.models.Model(inputs=loaded_model.input, outputs=fine_tune)

    
    print("MODEL SUMMARY:")
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)
    print("END OF SUMMARY")
    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.01, decay=0, nesterov=True)
    vgg_conv.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  #'auc'categorical_crossentropy
    return vgg_conv

def train_model(model,x_train,y_train,x_val,y_val,x_test,y_test, params,weights_dict = None):
    # train models over AUC, for x epochs. make it loopable for further test. return plottable data
    Epochs = params["epochs"]
    Batch_size = params["Batch_size"]

    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    # stop2 = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, restore_best_weights=True)
    check = tf.keras.callbacks.ModelCheckpoint(r"C:\models\model_temp.h5", monitor='val_acc', verbose=1, save_best_only=True)
    if params['stop'] == 'no':
        H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True,  callbacks = [], class_weight = weights_dict)
    else:
        H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True,  callbacks = [stop], class_weight = weights_dict)#,,callbacks =[check]
    # model.load_weights(r"C:\models\model_temp.h5")
    score = roc_auc_score(y_test, model.predict(x_test))
    print(' AUC of model = ' ,score)
    return H, score, model

#option 1
def data_generator_large_files(pathes, batch_size):

    while True: #generators for keras must be infinite
        for path in pathes:
            x_train, y_train = prepare_data(path)

            totalSamps = x_train.shape[0]
            batches = totalSamps // batch_size

            if totalSamps % batch_size > 0:
                batches+=1

            for batch in range(batches):
                section = slice(batch*batch_size,(batch+1)*batch_size)
                yield (x_train[section], y_train[section])

def config_desktop():
    ## WHEN USING TF 1.5 or lower and GPU ###
    #                                       #
    config = tf.ConfigProto()               #               
    config.gpu_options.allow_growth = True  #
    session = tf.Session(config=config)     #
    #########################################

def main():
    # small example to test script
    pass

if __name__ == '__main__':
    main()