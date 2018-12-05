
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# from tensorflow.applications import VGG16
import sklearn
from sklearn import datasets, svm, metrics
import os
import random
import pandas
import time

def import_melanoom(DIR, img_size_x,img_size_y, norm, color = False, classes = "two"):
    '''
    doc string
    '''
    try: 
        data_frame = pandas.read_csv(r"C:\Users\Floris\Documents\Python scripts\ISIC-2017_Training_Part3_GroundTruth.csv")
        data_frame = data_frame.set_index("image_id")
        os.listdir(DIR)
        data_dir = DIR
        print('Found directory')
    except:
        print("Directory path not found")

    # data_dir=r"C:\Users\s147057\Documents\Python Scripts\ISIC-2017_Training_Data"
    training_data = []
    target_data = []

    L = os.listdir(data_dir)
    L.reverse() # omdraaien van de lijst zorgt voor meer diversiteit van classes. 
    size = 1372+374
    start = time.time()
    i = 0
    for img in L[:-1]:
        if 'superpixels' in img:
            continue
        try:
            
            class_num = data_frame.loc[img[0:-4],:]
            if classes == "two":
                if class_num[0] == 1:
                    class_num = [0,1]
                elif class_num[1] == 1:
                    continue
                    class_num = [0,1]
                else:
                    class_num = [1,0]
                c = 2
            elif classes == "three":
                if class_num[0] == 1:
                    class_num = [0,1,0]
                elif class_num[1] == 1:
                    class_num = [0,0,1]
                else:
                    class_num = [1,0,0]
                c = 3
            elif classes == "two_combined":
                if class_num[0] == 1:
                    class_num = [0,1]
                elif class_num[1] == 1:
                    class_num = [0,1]
                else:
                    class_num = [1,0]
                c = 2

            if color:
                D = 3
                img_array = cv2.imread(os.path.join(data_dir,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            else:
                img_array = cv2.imread(os.path.join(data_dir,img), cv2.IMREAD_GRAYSCALE)
                D = 1
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            training_data.append(new_array)
            target_data.append(class_num)
        except Exception as e:
            pass
        loading(size,i,start, "melanoom data import")
        i+=1

    print('\n')
    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(target_data).reshape(-1,c)

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")


    if norm:
        x = x/ 255.0

    print(f"This melanoom dataset contains the following: \nTotal length Dataset = {len(x)}")
    return x,y 

def import_kaggleDR(path, img_size_x, img_size_y, norm, color = False, limit = None):

    target_path = path + r"\trainLabels.csv\trainLabels.csv"
    train_path = path + r"\train"
    data_frame = pandas.read_csv(target_path)
    data_frame = data_frame.set_index("image")

    os_dir_list = os.listdir(train_path)
    print('Found directory, start import data')

    print("Directory path not found")
    
    
    training_data = list()
    target_data = list()
    if limit:
        size = limit
    else:
        size = len(os_dir_list)
    start = time.time()
    i = 0
    for img_name in os_dir_list:
        img_data = data_frame.loc[img_name[:-5],:]
        try:
            if color:
                img_array = cv2.imread(os.path.join(train_path,img_name), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                D = 3
            else:
                img_array = cv2.imread(os.path.join(train_path,img_name), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                D = 1
        except Exception as e:
            pass
        training_data.append(new_array)
        target_data.append(int(img_data))
        loading(size,i,start, "KaggleDR data import")
        i+=1
        if limit:
            if i > limit-1:
                break
    
    print('\n')
    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(target_data).reshape(-1)

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")

    y,c = make_pre_train_classes(y, numb_classes = 5)
    
    if norm:
        x = x/ 255.0

    print(f"This melanoom dataset contains the following: \nTotal length Dataset = {len(x)}")
    return x,y 

def import_dogcat(path, img_size_x,img_size_y, norm, color):
    #DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
    try: 
        cat = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')


    training_data = list()
    training_class = list()

    for category in cat:
        path = os.path.join(path, category)
        class_num = cat.index(category)
        path = os.path.join(path, category)
        for img in os.listdir(path):
            try:
                if color:
                    D = 3
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                    
                else:
                    D = 1
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                    
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                training_data.append(new_array)
                training_class.append(class_num)
            except Exception as e:
                pass
    zip_list = list(zip(training_data,training_class))
    random.shuffle(zip_list)
    training_data,training_class = zip(*zip_list)
    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(training_class).reshape(-1,1)

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")

    if norm:
        x = x/ 255.0

    print(f"This Dog_Cat dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

def import_mnist(split, norm, limit = None):
    '''
    Creert data set van de mnist data.
    input: Normalize boolean
    output: train en test data set, each having their own list of classes and images.
    '''

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x = np.append(x_test,x_train,axis=0)
    y = np.append(y_test,y_train,axis=0)

    if limit:
        x = x[:limit]
        y = y[:limit]

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")

    test_split = 0.25
    val_split = 0.1

    spl = int(test_split*len(x))
    X = x[spl:]
    Y = y[spl:]
    x_test  = x[:spl]
    y_test  = y[:spl]

    spl2 = int(val_split*len(X))
    x_val = x[:spl2]
    y_val = y[:spl2]
    x_train = X[spl2:]
    y_train = Y[spl2:]

    if norm:
        x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0



    print(f"This Mnist dataset contains the following: \nTotal length Dataset = {len(x)} \nTotal length train set = {len(x_train)} \nTotal length val set = {len(x_val)} \nTotal length test set= {len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test



def train_val_test():
    #make consistent test set, and variable train sets
    pass

def pre_processing(X,img_size_x,img_size_y,demension):
    #def transform_img_list(X,img_size_x,img_size_y,demension):
    '''
    Door bijvoorbeeld te kleine afbeeldingen, wordt de data getransformeerd naar een ander formaat, later kan hier nog data generators aan toegevoegd worden
    input: lijst met afbeeldingen, en de afmeting van de gewenste afbeelding.
    output: Lijst met afbeeldingen getrasformeerd naar de afmeting.
    '''
    if type(X) != np.ndarray and type(X) != list:
        print("Please enter a list or np.array")
    if img_size_x < 48 or img_size_y < 48:
        print("img_size is too small, VGG16 needs an image size larger than 48X48")

    train_data = []
    for img in X:
        resized_image = cv2.resize(img, (img_size_x, img_size_y))
        train_data.append(resized_image)

    X = np.array(train_data).reshape(-1,img_size_x,img_size_y,demension)
    return X

def make_pre_train_classes(Y, numb_classes = None):
    '''
    zorgt voor classes op een manier dat een NN er mee kan werken, deze versie werkt op INTERGERS
    input: lijst met interger classes
    outpt: 0'en en 1'en lijst, even lang als classes aantal
    '''
    try:
        int(Y[0])
    except Exception as e:
        raise e
    
    clas = list()
    if not numb_classes:
        numb_classes = int(max(list(Y))-min(list(Y))+1)

    for label in list(Y):
        new_list = numb_classes*[0]
        new_list[int(label)] = 1
        clas.append(new_list)
    clas_np = np.array(clas).reshape(-1,numb_classes)
    return clas_np, numb_classes

def get_data(name_data,name_data2, img_size_x,img_size_y, norm, color = False):
    if name_data == 'mela':
        x_train, y_train, x_val, y_val, x_test, y_test = import_melanoom(img_size_x,img_size_y, norm, color)
    elif name_data == 'catdog':
        x_train, y_train, x_val, y_val, x_test, y_test = import_dogcat(img_size_x,img_size_y, norm, color)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = None,None,None,None,None,None
        print('There is no data set with that name')        

    if name_data2 == 'mela':
        x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = import_melanoom(img_size_x,img_size_y, norm, color)
    elif name_data2 == 'catdog':
        x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = import_dogcat(img_size_x,img_size_y, norm, color)
    else:
        print('Warning: No second set')
        x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = None,None,None,None,None,None
    print('Train, Val and test sets created')
    return x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2


def count_classes(Y):
    '''
    Counts classes, no matter wich type of class notation it is, array of 1 and 0 or interger.
    input: class list
    output: class dict, {class:count}
    '''
    ## checks if they are intergers ##
    
    back_to_num = list()
    list_classes = list(Y)
    for i in list_classes:
        back_to_num.append(list(i).index(1))
        
    d = dict()
    for n in back_to_num:
        if n in d:
            d[n] += 1
        else:
            d[n] = 1
    return d

def loading(size,i,start, name):
    stop = time.time()
    part = int(((i+1)/size)*20)
    loading_bar = part*'-'+(20-part)*' '
    print(f"{name}: {i+1}/{size}: [{loading_bar[0:10]}{part*5}%{loading_bar[10:20]}] elapsed time: {int(stop-start)}",end='\r')

def more_data(x,y,r):
    x_new = x
    y_new = y
    start = time.time()
    for i in range(0,r):
        loading(r,i,start, "Data generator")
        datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, fill_mode = "nearest")
        img=datagen.random_transform(x[i])
        y_new = np.concatenate((y_new, np.array(y[i]).reshape(1,y.shape[1])))
        x_new = np.concatenate((x_new, np.array(img).reshape(1,img.shape[0], img.shape[1],img.shape[2])))
    print("\n")
    return x_new,y_new

def equal_data(x,y):
    d = count_classes(y)
    l = list()
    for key in d.keys():
        l.append(d[key])
    limit = max(l)
    
    
    bad = list()
    
    x_new = x
    y_new = y
    start = time.time()
    for i in range(0,len(y)):
        loading(len(y),i,start, "Data generator")
        
        if not d[list(y[i]).index(1)] < limit:
            bad.append(list(y[i]))
            continue
        if list(y[i]).index(1) in bad:
            continue
        
        datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, fill_mode = "nearest")
        img=datagen.random_transform(x[i])
        y_new = np.concatenate((y_new, np.array(y[i]).reshape(1,y.shape[1])))
        x_new = np.concatenate((x_new, np.array(img).reshape(1,img.shape[0], img.shape[1],img.shape[2])))
        d = count_classes(y_new)
        if not d[list(y[i]).index(1)] < limit:
            bad.append(list(y[i]).index(1))
    print("\n")
    return x_new,y_new


def equal_data_run(y_new,x_new):
    while True:
        d_old = count_classes(y_new)
        x_new, y_new = equal_data(x_new,y_new)
        d_new = count_classes(y_new)
        if d_old == d_new:
            break
    return x_new,y_new