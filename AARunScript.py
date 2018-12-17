from RunSource import run
from RunTarget import run_target


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

run(params)

model = ["imagenet","kaggleDR"]
data = ['two','three','two_combined']
style = ['FT', 'SVM']

x = 0
y = 0
z = 0
r = 1

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

# RunTarget.run_target(params)

