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

def test_script(limit):
    x_train, y_train, x_val, y_val, x_test, y_test = import_mnist(0.10, True, limit)
    x_train = pre_processing(x_train, 28,28,1)
    x_test = pre_processing(x_test, 28,28,1)
    y_train,numb_classes = make_pre_train_classes(y_train)
    y_test, none = make_pre_train_classes(y_test)
    print(x_train.shape)
    print(x_test.shape)
    x = auc_svm(x_train, y_train, x_test, y_test,plot = True)
    return x