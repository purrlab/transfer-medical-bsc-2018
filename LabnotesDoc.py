'''### LAPNOTES AUTOMATE --> write txt file with used params and datum and allllll ''' 

## oplost ##

# save result fig with date.
# save results and params in txt in a file.
# versie van fucntie vermelden.
import matplotlib.pyplot as plt
import numpy as np
import time

# fig.savefig('path/to/save/image/to.png')   # save the figure to file
# plt.close(fig)    # close the figure

# par = {"epochs":5, "size":10000, "model":"VGG"}
# res = {"AUC":[0.1,0.2,0.3,0.4,0.5], "acc":0.99, "loss":0.66}
# fig_str = "Results_1.fig"

def doc(par,res,H,doc_path):

    t = time.localtime()
    file_name= f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_RUNEXPERIMENT_RESULTS00{t.tm_hour}{t.tm_min}{t.tm_sec}.txt"

    f= open(r"{}ransfer-medical-bsc-2018\LAB\{}".format(doc_path,file_name),"w+")

    params = str()
    x = 1
    for item in par.keys():
        params = params + f"\n{x}." + str(item) + " = " + str(par[item])
        x+=1

    results = str()
    x = 1
    for item in res.keys():
        results = results + f"\n{x}." + str(item) + " = " + str(res[item])
        x+=1

    all_times = "Floris Fok \n" + str(t.tm_year)+"/" + str(t.tm_mon) +"/"+ str(t.tm_mday) +"      At: "+ str(t.tm_hour) +":" + str(t.tm_min)  +":"+ str(t.tm_sec)

    f.write(f"{all_times}\n \n \n The params where: {params} \n \n \n The results where: {results}")
    if H != None:
        fig = plt.figure(figsize=(3, 6))
        plt.plot(H.history['acc'])
        plt.plot(H.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig_str = f"{doc_path}{t.tm_year}_{t.tm_mon}_{t.tm_mday}_RUNEXPERIMENT_FigureACC0{t.tm_hour}{t.tm_min}.png"
        fig.savefig(fig_str, dpi=fig.dpi)

        fig = plt.figure(figsize=(3, 6))
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig_str2 = f"{doc_path}{t.tm_year}_{t.tm_mon}_{t.tm_mday}_RUNEXPERIMENT_FigureLOSS&ACC0{t.tm_hour}{t.tm_min}.png"
        fig.savefig(fig_str2, dpi=fig.dpi)
        f.write(f"\nFigure name: \n{fig_str}"+"\n" + f"{fig_str2}")
        #plt.show()