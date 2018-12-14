    
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics
# import keras.backend as K
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

# def auc(y_true, y_pred):   
#     ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
#     pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
#     pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
#     binSizes = -(pfas[1:]-pfas[:-1])
#     s = ptas*binSizes
#     return K.sum(s, axis=0)

def make_model(x, y, w = None):

    if w != None:
        classes = 1000
        vgg_conv = tf.keras.applications.VGG16(weights=w,input_shape = (x[0].shape), include_top=True, classes=classes)
        # for layer in vgg_conv.layers[:-5]:
        #     layer.trainable = False
        fine_tune = tf.keras.layers.Dense(500, activation='relu')(vgg_conv.output)
        fine_tune = tf.keras.layers.Dense(350, activation='relu')(fine_tune)
        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(fine_tune)
        vgg_conv  = tf.keras.models.Model(inputs=vgg_conv.input, outputs=fine_tune)
    else:
        c = int(y.shape[1])
        vgg_conv = tf.keras.applications.VGG16(weights=w,input_shape = (x[0].shape), include_top=True, classes=c)
        
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)
    opt = tf.keras.optimizers.SGD(lr=0.0009, momentum=0.01, decay=0.0, nesterov=True)
    #opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.90)
    vgg_conv.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  #'auc'
    return vgg_conv

def train_model(model,x_train,y_train,x_val,y_val,x_test,y_test, Epochs, Batch_size):
    # train models over AUC, for x epochs. make it loopable for further test. return plottable data

    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

    H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True, callbacks = [stop])
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
                
#Create and use:

# gen = dataGenerator(['xaa', 'xab', 'xac', 'xad'], 50)
# model.fit_generator(gen,
#                     steps_per_epoch = expectedTotalNumberOfYieldsForOneEpoch
#                     epochs = epochs)

# Option 2
# for epoch in range(20):
#     for path in ['xaa', 'xab', 'xac', 'xad']:
#         x_train, y_train = prepare_data(path)
#         model.fit(x_train, y_train, batch_size=50, epochs=epoch+1, initial_epoch=epoch, shuffle=True)

def config_desktop():
    ## WHEN USING TF 1.5 or lower and GPU ###
    config = tf.ConfigProto()               #
    config.gpu_options.allow_growth = True  #
    session = tf.Session(config=config)     #
    #########################################

def main():
    # small example to test script
    pass

if __name__ == '__main__':
    main()