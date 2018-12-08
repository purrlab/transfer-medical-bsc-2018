import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
## Import all scripts ##
from AADatasets import import_mnist, import_dogcat, import_melanoom, pre_processing, make_pre_train_classes, get_data, more_data, equal_data_run, val_split
from AAPreTrain import make_model,train_model, config_desktop
from AATransferLearn import get_feature_vector, preform_svm, auc_svm
from LabnotesDoc import doc

def run_experiment(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

    x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm, train_size)

    # if x_train != None:
    x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
    x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
    y_train,numb_classes = make_pre_train_classes(y_train)
    y_test, none = make_pre_train_classes(y_test)
    # else:
    #   print("dataset 1 not found")

    # if x_train2 != None:
    x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
    x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
    # else:
    #   print("dataset 2 not found")

    # if not model:
    model = make_model(x_train, y_train, numb_classes)

    H, score = train_model(model,x_train,y_train,x_test,y_test, Epochs, Batch_size)

    vector = get_feature_vector(model, x_train2, layer = 'fc2')
    vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
    x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
    return x, score

def run_experiment2(name_data,name_data2, vgg, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

    x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, model = get_data(name_data,name_data2, vgg ,img_size_x,img_size_y, norm)

    # if x_train != None:
    x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
    x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
    y_train,numb_classes = make_pre_train_classes(y_train)
    y_test, none = make_pre_train_classes(y_test)
    # else:
    #   print("dataset 1 not found")

    # if x_train2 != None:
    x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
    x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
    # else:
    #   print("dataset 2 not found")
    results = []
    for i in range(0,len(train_size)):
        # if not model:
        model = make_model(x_train, y_train, numb_classes)
        H, score = train_model(model,x_train[:train_size[i]],y_train[:train_size[i]],x_test,y_test, Epochs, Batch_size)
        vector = get_feature_vector(model, x_train2, layer = 'fc2')
        vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
        x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)
        results.append(x)

    return results

def run_experiment3(name_data,name_data2, img_size_x, img_size_y, demension, Batch_size, Epochs, norm, train_size):

    x_train, y_train, x_val, y_val, x_test, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = get_data(name_data,name_data2 ,img_size_x,img_size_y, norm, color = True)

    # if x_train != None:
    # x_train = pre_processing(x_train, img_size_x, img_size_y, demension)
    # x_test = pre_processing(x_test, img_size_x, img_size_y, demension)
    # y_train,numb_classes = make_pre_train_classes(y_train)
    # y_test, none = make_pre_train_classes(y_test)
    # else:
    #   print("dataset 1 not found")

    # if x_train2 != None:
    x_train2 = pre_processing(x_train2, img_size_x, img_size_y, demension)
    x_test2 = pre_processing(x_test2, img_size_x, img_size_y, demension)
    # else:
    #   print("dataset 2 not found")
    results = []
    model = make_model(x_train2, 3, w = 'imagenet')
    # # if not model:
    # model = make_model(x_train, y_train, numb_classes)
    H, score = train_model(model,x_train2,y_train2,x_test2,y_test2, Epochs, Batch_size)
    # vector = get_feature_vector(model, x_train2, layer = 'fc2')
    # vector_val = get_feature_vector(model, x_test2, layer = 'fc2')
    # x = auc_svm(vector, y_train2, vector_val, y_test2, plot= False)

    results = score

    return results

def test_script():
    params = {'img_size_x':48,'img_size_y':48,'norm':False,'color':False, 'pretrain':None, "equal_data":True, "shuffle": True, "epochs": 3 , "val_size":10,"test_size":20, "Batch_size": 2}
    file_path = r"C:\Miniset"
    doc_path =  r"C:\Users\Flori\Documents\GitHub\t"
    #config_desktop()

    x,y = import_dogcat(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"])
    if params["equal_data"] == True:
        x,y = equal_data_run(y,x)

    zip_melanoom = zip(x,y)
    pickle.dump( zip_melanoom, open( r"C:\Users\Flori\Documents\GitHub\pickles\test.p", "wb" ))
    zippy = list(zip_melanoom)
    random.shuffle(zippy)
    x,y = zip(*zippy)
    x = np.array(x)
    y = np.array(y)
    x_test,y_test,x,y = val_split(x,y, params["test_size"])
    x_val,y_val,x,y = val_split(x,y, params["val_size"])



    model = make_model(x, y, w = params['pretrain'])
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])

    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    doc(params,results,H, doc_path)

def menegola_plane():
    params = {'img_size_x':224,'img_size_y':224,'norm':False,'color':True, 'pretrain':None, "equal_data":True, "shuffle": True, "epochs": 50 , "val_size":200,"test_size":300, "Batch_size": 16, "mela_class":"three"}
    file_path = r"C:\ISIC\ISIC-2017_Training_Data"
    config_desktop()

    try:
        print("Try to import pickle")
        zippy = pickle.load(open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_three.p", "rb" ))
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
        x,y = import_melanoom(file_path, params['img_size_x'],params['img_size_y'], norm = params["norm"], color = params["color"],classes = params["mela_class"])
        if params["equal_data"] == True:
            x,y = equal_data_run(y,x)

        zip_melanoom = zip(x,y)
        pickle.dump( zip_melanoom, open( r"C:\Users\Floris\Documents\GitHub\pickles\save_melanoom_color_three.p", "wb" ))
        zippy = list(zip_melanoom)
        random.shuffle(zippy)
        x,y = zip(*zippy)
        x = np.array(x)
        y = np.array(y)
        x_test,y_test,x,y = val_split(x,y, params["test_size"])
        x_val,y_val,x,y = val_split(x,y, params["val_size"])



    model = make_model(x, y, w = params['pretrain'])
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params["epochs"], params["Batch_size"])

    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    doc(params,results,H)

    #save model to JSON
    model_json = model.to_json()
    with open(r"D:\models\Epochs_50_melanoom_equal.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(r"D:\models\weights\Epochs_50_melanoom_equal.h5")
    print("Saved model to disk")