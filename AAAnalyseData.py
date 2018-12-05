
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics

class AnalyseDataCLass():
    def __init__(data):
        self.data = data

    def Random_score(y):
        '''
        Get a score for randomness, takes steps equal to amount of classes, sees if it contains all the classes.
        Could set score limit
        input: classes
        output: score of given data
        '''
        ## checks if they are intergers based classes ##
        if type(Y[0]) != np.ndarray:
            back_to_num = list()
            for i in Y:
                back_to_num.append(list(i).index(1))

        score = 0
        numb_classes = max(y)-min(y)+1
        totaal = int(len(y)/numb_classes)
        score = 0
        for n in range(0,len(y[:totaal*numb_classes]),numb_classes):
            combi = (len(set(y[n:n+numb_classes])))
            score += combi/(totaal*numb_classes)
        print(score)
        return y, score

    def count_classes():
        '''
        Counts classes, no matter wich type of class notation it is, array of 1 and 0 or interger.
        input: class list
        output: class dict, {class:count}
        '''
        ## checks if they are intergers ##
        if type(Y[0]) == np.ndarray:
            back_to_num = list()
            for i in Y:
                back_to_num.append(list(i).index(1))

        d = dict()
        for n in back_to_num:
            if n in d:
                d[n] += 1
            else:
                d[n] = 1
        return d
    def get_shape():
            print(X_test.shape)


def plot_pre_train_result():
    # plot some features of the model that will help give guidance to result
        '''
    Bij gebruik van vele epochs zijn deze plots handig
    input: NN
    output: Graphs d.m.v. plt.show()
    '''
    # summarize history for accuracy
    plt.plot(H.history['acc'])
    plt.plot(H.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def show_accuracy():
    # check if prediction is rigth using few images (i guess 4)

def plot_results():
    # make a graph using data from forloops on different configs (acc/pre_train_size)

def main():
    # small example to test script
    
if __name__ == '__main__':
    main()