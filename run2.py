from RunTarget import run_target
from RunSource import run
import sys
arg = sys.argv[1]

model = ["imagenet","Chest","CatDog","KaggleDR","Nat"]
data = ['two','three','two_combined']#data[y]
style = ['FT', 'SVM']

d = int(arg[0])
x = 4 
y = int(arg[1])
z = int(arg[2])
r = int(arg[3])


params = {"Data":'Blood',
        "data_name":None,
        "style":style[z],
        "model":model[x],
        "file_path":r"C:\blood-cells",
        "pickle_path":r"C:\pickles\Blood",
        "model_path":{"nat":r"C:\models\Epochs_{}_Nat".format(((d*4)+2)),"KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
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
run_target(params)