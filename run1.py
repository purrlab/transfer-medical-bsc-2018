from RunTarget import run_target
from RunSource import run
import sys
arg = sys.argv[1]

model = ["imagenet","Chest","CatDog","KaggleDR","Nat",'None']
data = ['two','three','two_combined']#data[y]
style = ['FT', 'SVM']

x = int(arg[0])
d = int(arg[1])
z = int(arg[2])
r = int(arg[3])
y = 4

#"pickle_path":r"C:\pickles\save_melanoom_color_",

# run_target(params)
if x ==0:
        params = {"Data":'ISIC',
                "data_name":data[d],
                "style":style[z],
                "model":model[y],
                "file_path":r"C:\ISIC",
                "pickle_path":r"C:\pickles\save_melanoom_color_",
                 "model_path":{'Nat':r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
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
                "val_size":200,
                "test_size":400, 
                "Batch_size":32,
                "stop":'yes'
                }

elif x == 1:
        params = {"Data":'Nat',
                "style":'none',
                "model":'none',
                'file_path':r"C:\natural_images",
                'pickle_path':r"C:\pickles\Nat",
                'model_path':r"C:\models\Epochs_50_Nat.json",
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
                "val_size":y*500,
                "test_size":1000, 
                "Batch_size":32,
                "stop":'yes'
                }

elif x == 2:
        params = {"Data":'Blood',
                "data_name":None,
                "style":style[z],
                "model":model[y],
                "file_path":r"C:\blood-cells",
                "pickle_path":r"C:\pickles\Blood",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
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
                "val_size":1500,
                "test_size":2000, 
                "Batch_size":32,
                "stop":'yes'
                }
elif x == 3:
         params = {"Data":'Chest',
                "data_name":None,
                "style":style[z],
                "model":model[y],
                "file_path":r"C:\chest_xray",
                "pickle_path":r"C:\pickles\Chest_int",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
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
                "val_size":600,
                "test_size":900, 
                "Batch_size":32,
                "stop":'yes'
                }
elif x == 4:
        params = {"Data":'KaggleDR',
                "data_name":None,
                "style":style[z],
                "model":model[y],
                "file_path":r"C:\KaggleDR",
                "pickle_path":r"C:\pickles\KaggleDR",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
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
                "Batch_size":32,
                "stop":'yes'
                }
run_target(params)
# run(params)