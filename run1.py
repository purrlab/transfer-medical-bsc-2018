from RunTarget import run_target
import sys
arg = sys.argv[1]

model = ["imagenet","Chest","CatDog","KaggleDR"]
# data = ['two','three','two_combined']data[y]
style = ['FT', 'SVM']

x = int(arg[0]) 
y = int(arg[2])
z = int(arg[1])
r = int(arg[3])

# if y == 0:
#         params = {"Data":'Breast',
#                 "data_name":None,
#                 "style":style[z],
#                 "model":model[x],
#                 "file_path":r"C:\breast-ultrasound-image",
#                 "pickle_path":r"C:\pickles\Breast",
#                 "model_path":{"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
#                 "RandomSeed":r,
#                 "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
#                 'img_size_x':224,
#                 'img_size_y':224,
#                 'norm':False,
#                 'color':True, 
#                 'pretrain':None, 
#                 "equal_data":False, 
#                 "shuffle":True, 
#                 "epochs":50 , 
#                 "val_size":25,
#                 "test_size":50, 
#                 "Batch_size":6
#                 }
# if y == 1:
#         params = {"Data":'Blood',
#                 "data_name":None,
#                 "style":style[z],
#                 "model":model[x],
#                 "file_path":r"C:\blood-cells",
#                 "pickle_path":r"C:\pickles\Blood",
#                 "model_path":{"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
#                 "RandomSeed":r,
#                 "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
#                 'img_size_x':224,
#                 'img_size_y':224,
#                 'norm':False,
#                 'color':True, 
#                 'pretrain':None, 
#                 "equal_data":False, 
#                 "shuffle":True, 
#                 "epochs":50 , 
#                 "val_size":300,
#                 "test_size":500, 
#                 "Batch_size":32
#                 }
# params = {"Data":'Chest',
#         "data_name":None,
#         "style":style[z],
#         "model":model[x],
#         "file_path":r"C:\chest_xray",
#         "pickle_path":r"C:\pickles\Chest_int",
#         "model_path":{"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
#         "RandomSeed":r,
#         "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
#         'img_size_x':224,
#         'img_size_y':224,
#         'norm':False,
#         'color':True, 
#         'pretrain':None, 
#         "equal_data":False, 
#         "shuffle":True, 
#         "epochs":50 , 
#         "val_size":300,
#         "test_size":500, 
#         "Batch_size":32
#         }
params = {"Data":'KaggleDR',
        "data_name":None,
        "style":style[z],
        "model":model[x],
        "file_path":r"C:\KaggleDR",
        "pickle_path":r"C:\pickles\KaggleDR",
        "model_path":{"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
        "RandomSeed":r,
        "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':False,
        'color':True, 
        'pretrain':None, 
        "equal_data":False, 
        "shuffle":True, 
        "epochs":50 , 
        "val_size":5000,
        "test_size":5000, 
        "Batch_size":32
        }

#"pickle_path":r"C:\pickles\save_melanoom_color_",

run_target(params)