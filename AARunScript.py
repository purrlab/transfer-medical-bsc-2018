from RunSource import run
from RunTarget import run_target
r= 2
params = {"Data":'Nat',
        "style":'none',
        "model":'none',
        'file_path':r"C:\natural_images",
        'pickle_path':r"C:\pickles\Nat",
        'model_path':r"C:\models\Epochs_",
        "RandomSeed":r,
        "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':False,
        'color':True, 
        'pretrain':None, 
        "equal_data":False, 
        "shuffle":True, 
        "epochs":50, 
        "val_size":500,
        "test_size":1000, 
        "Batch_size":32
        }

run(params)

params = {"Data":'CatDog',
        'file_path':r"C:\PetImages",
        'pickle_path':r"C:\pickles\CatDog",
        'model_path':r"C:\models\Epochs_",
        'doc_path':r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':True,
        'color':True,
        'pretrain':None,
        "equal_data":False, 
        "shuffle": True, 
        "epochs": 50 , 
        "val_size":3000,
        "test_size":5000, 
        "Batch_size": 32
        }

params = {"Data":'ISIC',
        "data_name":['two_combined','three','two'],
        "style":['FT','SVM'],
        "model":None,
        "file_path":r"C:\ISIC",
        "pickle_path":r"C:\pickles\melanoom_color_NotEqual_",
        "model_path":r"C:\models\Epochs_5_kaggleDR.json",
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
        "val_size":100,
        "test_size":200, 
        "Batch_size":16
        }

params = {"Data":'KaggleDR',
        'file_path':r"C:\kaggleDR",
        'pickle_path':r"C:\pickles\kaggleDR",
        'model_path':r"C:\models\Epochs_",
        'doc_path':r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':True,
        'color':True,
        'pretrain':None,
        "equal_data":False, 
        "shuffle": True, 
        "epochs": 50, 
        "val_size":4000,
        "test_size":4000, 
        "Batch_size": 32
        }

params = {"Data":'Chest',
        'file_path':r"C:\chest_xray",
        'pickle_path':r"C:\pickles\Chest_int",
        'model_path':r"C:\models\Epochs_",
        'doc_path':r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':False,
        'color':True,
        'pretrain':None,
        "equal_data":False, 
        "shuffle": True, 
        "epochs": 50,
        "val_size":200,
        "test_size":200, 
        "Batch_size": 16
        }

# run(params)

model = ["imagenet","kaggleDR"]
data = ['two','three','two_combined']
style = ['FT', 'SVM']

x = 0
y = 0
z = 0
r = 2

params = {"Data":'ISIC',
        "data_name":data[y],
        "style":style[z],
        "model":model[x],
        "file_path":r"C:\ISIC",
        "pickle_path":r"C:\pickles\save_melanoom_color_",
        "model_path":r"C:\models\Epochs_40_CatDog.json",
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
        "val_size":300,
        "test_size":400, 
        "Batch_size":16
        }

params = {"Data":'Blood',
        "data_name":None,
        "style":"SVM",
        "model":"imagenet",
        "file_path":r"C:\blood-cells",
        "pickle_path":r"C:\pickles\Blood",
        "model_path":{"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
        "RandomSeed":2,
        "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
        'img_size_x':224,
        'img_size_y':224,
        'norm':False,
        'color':True, 
        'pretrain':None, 
        "equal_data":False, 
        "shuffle":True, 
        "epochs":10 , 
        "val_size":1500,
        "test_size":2000, 
        "Batch_size":32
        }
# params = {"Data":'Breast',
#         "data_name":None,
#         "style":'SVM',
#         "model":'kaggleDR',
#         "file_path":r"C:\breast-ultrasound-image",
#         "pickle_path":r"C:\pickles\Breast",
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
#         "val_size":25,
#         "test_size":50, 
#         "Batch_size":6
#         }

#run_target(params)

