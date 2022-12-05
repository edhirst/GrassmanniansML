'''Miscellaneous Analysis of Grassmannian data'''
'''
Run cells:
Cell 1 ==> import libraries.
Cell 2 ==> import all the data (CV & NCV), and reformat the tableaux.
Cell 3 ==> sample tableaux from each dataset to plot as images.
Cell 4 ==> select the dataset to partition ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), count the partitions of a selected dataset according to tableaux rank.
Cell 5 ==> select the dataset to partition ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), partition the full dataset according to tableaux rank and n, to produce the N_{k,n,r} table.
'''
### Cell 1 ###
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval as LE

#%% ### Cell 2 ###
#Set-up data import
Datachoices = [0,1,2] #...select the indices of the filepaths to import data from (i.e. choose the data to import)
Datafiles =    ['./Data/CVData/CV_Gr312_Rank6.txt',  './Data/CVData/CV_Gr410_Rank6.txt',  './Data/CVData/CV_Gr412_Rank4.txt']
NCVdatafiles = ['./Data/NCVData/NCV_Gr312_Rank6.txt','./Data/NCVData/NCV_Gr410_Rank6.txt','./Data/NCVData/NCV_Gr412_Rank4.txt']
prefixes = [f[-7:-4] for f in Datafiles]

#Import Grassmannians
Data = []
for datapath in Datachoices:
    Data.append([])
    with open(Datafiles[datapath],'r') as file:
        for line in file.readlines():
            Data[-1].append(LE(line))
print('Dataset sizes: '+str(list(map(len,Data))))

#Pad data (all datasets to the same size)
max_height = max([max(map(len,dataset)) for dataset in Data])
max_width  = max([max(map(len,[tab[0] for tab in dataset])) for dataset in Data]) #...all tableaux rows the same length so can just consider first row
for dataset_idx in range(len(Data)):
    for tab_idx in range(len(Data[dataset_idx])):
        tab = np.array(Data[dataset_idx][tab_idx])
        new_tab = np.zeros((max_height,max_width),dtype=int)
        new_tab[:len(tab),:len(tab[0])] = tab
        Data[dataset_idx][tab_idx] = new_tab
    Data[dataset_idx] = np.array(Data[dataset_idx])

#Import NCV data
NCV = []
for datapath in Datachoices:
    NCV.append([])
    with open(NCVdatafiles[datapath],'r') as file:
        NCV[-1] = LE(file.read())
    NCV[-1] = np.array(NCV[-1])
print('NCV dataset sizes: '+str(list(map(len,NCV))))

del(dataset_idx,tab_idx,tab,new_tab,file,line,datapath)

########################################################
#%% ### Cell 3 ###
#Select tableaux to plot
number_tab = 10 #..select how many tableaux to plot from each dataset
choices = [np.random.choice(range(len(Data[i])),number_tab,replace=False) for i in range(len(Datafiles))] 

#Plot sample of SSYTs as images
for f in range(len(choices)):
    for i in choices[f]:
        plt.axis('off')
        plt.imshow(Data[f][i])
        #plt.savefig('./Images/'+str(prefixes[f])+'_'+str(i)+'.pdf')
del(f,i)

########################################################
#%% ### Cell 4 ###
G_choice = 0 #...choose the grassmannian to partition: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}

#Rank partition the full datasets 
r_parts = np.zeros(6)
for tab in Data[G_choice]:
    if   tab[0][1] == 0: r_parts[0]+=1
    elif tab[0][2] == 0: r_parts[1]+=1
    elif tab[0][3] == 0: r_parts[2]+=1
    elif tab[0][4] == 0: r_parts[3]+=1
    elif tab[0][5] == 0: r_parts[4]+=1
    else:                r_parts[5]+=1
print(r_parts)

########################################################
#%% ### Cell 5 ###
#Compute the counts table
G_choice = 0 #...choose the grassmannian to partition: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}
datainfo = [[3,12,6],[4,10,6],[4,12,4]]

#Initialise table
N = np.zeros((datainfo[G_choice][1]-datainfo[G_choice][0]+1,datainfo[G_choice][2]),dtype=int)

#Loop through SSYTs and count respective table entries
for tab in Data[G_choice]:
    #Identify rank
    smaller = False
    for i, e in enumerate(tab[0]):
        if e == 0: 
            r_idx = i-1
            smaller = True
            break
    if not smaller: r_idx = len(tab[0])-1
    #Identify n
    n_idx = max(tab.flatten())-datainfo[G_choice][0]
    #Update correct table entry
    N[n_idx,r_idx] += 1

print(N,end='\n\n')

#Convert N to table format
from copy import deepcopy as dc
Nfull = dc(N)
for i in range(len(N)):
    Nfull[i] = np.sum(N[:i+1],axis=0)
print(Nfull)

del(tab,i,e,n_idx,r_idx,smaller)
