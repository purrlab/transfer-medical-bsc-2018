def import_blood(path, img_size_x,img_size_y, norm, color):
    #DIR = r"C:\Users\Floris\Documents\Python scripts\PetImages"
    try: 
        types = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')

    x = dic()
    y = dic()
    
    for type_set in types:
        training_data = list()
        training_class = list()
        path2 = os.path.join(path, type_set)
        cat = list(os.listdir(path2))
        numb_classes = len(cat)

        for category in cat:
            class_num = cat.index(category)
            path3 = os.path.join(path2, category)
            start = time.time()
            i = 0
            size = len(list(os.listdir(path3)))
            for img in os.listdir(path3):
                try:
                    if color:
                        D = 3
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_COLOR)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        
                    else:
                        D = 1
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        print("done")
                    training_data.append(new_array)
                    new_list = numb_classes*[0]
                    new_list[int(label)] = 1
                    training_class.append(new_list)
                    # if class_num == 0:
                    #     training_class.append([1,0,0,0])
                    # elif class_num == 1:
                    #     training_class.append([0,1,0,0])
                    # elif class_num == 2:
                    #     training_class.append([0,0,1,0])
                    # elif class_num == 3:
                    #     training_class.append([0,0,0,1])
                except Exception as e:
                    pass
                loading(size,i,start, "Chest data import")
                i+=1

            print("\n")

        zip_list = list(zip(training_data,training_class))
        random.shuffle(zip_list)
        training_data,training_class = zip(*zip_list)
        training_data = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
        training_class = np.array(training_class).reshape(-1,len(cat))
        x[type_set] = training_data
        y[type_set] = training_class

    print(f"This Chest dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

