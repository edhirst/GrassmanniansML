'''PCA of Grassmannian cluster algebra datasets'''
'''
Run cells:
Cell 1 ==> import libraries.
Cell 2 ==> import all the data (CV & NCV), and reformat the tableaux.
Select which investigation to run: (1) PCA all the data together (Cells 4,5), (2) PCA each dataset (Cell 6), (3) PCA the rank/n partitioned data (Cells 7,8,9), (4) PCA the CV vs NCV data (Cells 10,11).
(1) Cell 4  ==> select whether to use kernel PCA (if so first run Cell 3), fit the PCA
    Cell 5  ==> plot the PCA results.
(2) Cell 6  ==> PCA each of the datasets individually, and plot separately.
(3) Cell 7  ==> select the dataset to partition PCA ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), and perform the partitions.
    Cell 8  ==> perform the PCA, and sort points according to the partitioning.
    Cell 9  ==> select the partition to plot {01,} = {rank,n}, plot the partitioned PCA for the dataset.
(4) Cell 10 ==> select the dataset to PCA ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), select whether to use kernel PCA (if so first run Cell 3), perform the PCA.
    Cell 11 ==> plot the PCA.
'''
### Cell 1 ###
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval as LE
from sklearn.decomposition import PCA 
from sklearn.decomposition import KernelPCA  #...maps the data to a higher-dimensional space, and uses a non-linear metric (of choice) to separate the data, note a kernel-trick is used so as not to require computation of the high-dimensional embedding

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

#%% ### Cell 3 ###
#Sample the CV data (necessary when using kernel PCA, as memory problems with full datasets)
#Save the full data elsewhere to avoid reimporting
from copy import deepcopy as dc
Data_backup = dc(Data)

#Sample data
sample_size = 10000 #...set to 0 to use full dataset
if sample_size:
    for dataset_idx in range(len(Data)):
        Data[dataset_idx] = np.array([Data[dataset_idx][i] for i in np.random.choice(len(Data[dataset_idx]),sample_size,replace=False)])
del(dataset_idx)

########################################################
#%% ### Cell 4 ###
#PCA all the data together
kernel_check = False #...select whether to use Gaussian kernel (True), or simple linear (False)

#PCA transform the full dataset
all_data = np.concatenate((Data[0].reshape(Data[0].shape[0],-1),Data[1].reshape(Data[1].shape[0],-1),Data[2].reshape(Data[2].shape[0],-1))) 
if not kernel_check: pca = PCA(n_components=24) #...just use 2 components for plotting
else:                pca = KernelPCA(n_components=24,kernel='rbf') #...specify kernel used
pca.fit(all_data)
pcad_datasets = [pca.transform(Data[0].reshape(Data[0].shape[0],-1)),pca.transform(Data[1].reshape(Data[1].shape[0],-1)),pca.transform(Data[2].reshape(Data[2].shape[0],-1))]
print(pca.explained_variance_ratio_)

#%% ### Cell 5 ###
#Plot the PCA
plt.figure('Full PCA')
for d_idx in range(len(Data)):
    plt.scatter(pcad_datasets[d_idx][:,0],pcad_datasets[d_idx][:,1],alpha=0.1,label=r'$\mathbb{C}$[Gr('+prefixes[d_idx][0]+','+prefixes[d_idx][1:]+')]')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')

leg=plt.legend(loc='best')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
#plt.savefig('./KPCA_all10k_###.png')
#del(all_data,leg,lh)

########################################################
#%% ### Cell 6 ###
#PCA the data separately
#PCA transform the datasets
pcad_datasets = []
for d_idx in range(len(Data)):
    pca = PCA(n_components=Data[d_idx].shape[1]*Data[d_idx].shape[2]) 
    pcad_datasets.append(pca.fit_transform(Data[d_idx].reshape(Data[d_idx].shape[0],-1)))
    print('Explained Variance ratio for dataset '+str(prefixes[d_idx])+':\n'+str(pca.explained_variance_ratio_)+' (i.e. normalised eigenvalues)') #...note components gives rows as eigenvectors

#Plot the PCA transformed datasets
for d_idx in range(len(Data)):
    plt.figure('Dataset: '+str(prefixes[d_idx]))
    #plt.title('Dataset '+str(prefixes[d_idx]))
    plt.scatter(pcad_datasets[d_idx][:,0],pcad_datasets[d_idx][:,1],alpha=0.1)
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    #plt.savefig('./PCA_dataset_'+str(prefixes[d_idx])+'.png') #...too many points to save as a pdf
del(d_idx)    
    
########################################################
#%% ### Cell 7 ###
#PCA subdatasets according to rank (columns) or max entries (n), and colour separately
G_choice = 0 #...choose the grassmannian to partition: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}
datainfo = [[3,12,6],[4,10,6],[4,12,4]]

#Partition the dataset
r_partitioned, n_partitioned = [[] for i in range(datainfo[G_choice][2])], [[] for i in range(datainfo[G_choice][0],datainfo[G_choice][1]+1)]
for tab in Data[G_choice]:
    n_partitioned[max(tab.flatten())-datainfo[G_choice][0]].append(tab)
    #Identify where columns padded to establish a lower rank
    smaller = False
    for i, e in enumerate(tab[0]):
        if e == 0: 
            r_partitioned[i-1].append(tab)
            smaller = True
            break
    if not smaller: r_partitioned[len(tab[0])-1].append(tab)
#Convert to arrays
r_partitioned = [np.array(i) for i in r_partitioned]
n_partitioned = [np.array(i) for i in n_partitioned]

#%% ### Cell 8 ###
#Perform the PCA
pca = PCA(24)
pca.fit(Data[G_choice].reshape(Data[G_choice].shape[0],-1))
pcad_rpartitions = [pca.transform(i.reshape(i.shape[0],-1)) for i in r_partitioned]
pcad_npartitions = [pca.transform(i.reshape(i.shape[0],-1)) for i in n_partitioned]

#%% ### Cell 9 ###
#Select which partition to plot
partition_plot = 0 #...{0,1} = {rank,n}

#Plot the partitioned PCA (change list choice to r or k partitions)
plt.figure('PCA partitions')
if not partition_plot:
    for p_idx in range(len(pcad_rpartitions)-1,-1,-1):
        plt.scatter(pcad_rpartitions[p_idx][:,0],pcad_rpartitions[p_idx][:,1],alpha=0.1,label=p_idx+1)
else:
    for p_idx in range(len(pcad_npartitions)-1,-1,-1):
        plt.scatter(pcad_npartitions[p_idx][:,0],pcad_npartitions[p_idx][:,1],alpha=0.1,label=p_idx+datainfo[G_choice][0])
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.xlim(-5,25)
leg=plt.legend(loc='lower right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
#plt.savefig('./PCA###_#partition.png')

########################################################
#%% ### Cell 10 ###
#PCA the CV vs NCV
G_choice = 0         #...select the grassmannian to pca: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}
kernel_check = False #...select whether to use Gaussian kernel (True), or simple linear (False)

#PCA transform the full dataset
all_data = np.concatenate((Data[G_choice].reshape(Data[G_choice].shape[0],-1),NCV[G_choice].reshape(NCV[G_choice].shape[0],-1))) 
if not kernel_check: pca = PCA(n_components=24) #...just use 2 components for plotting
else:                pca = KernelPCA(n_components=2,kernel='rbf') #...specify kernel used
pca.fit(all_data)
pcad_datasets = [pca.transform(Data[G_choice].reshape(Data[G_choice].shape[0],-1)),pca.transform(NCV[G_choice].reshape(NCV[G_choice].shape[0],-1))]
print(pca.explained_variance_ratio_)

#%% ### Cell 11 ###
#Plot the PCA
plt.figure('PCA')
plt.scatter(pcad_datasets[0][:,0],pcad_datasets[0][:,1],alpha=0.1,label='CV')
plt.scatter(pcad_datasets[1][:,0],pcad_datasets[1][:,1],alpha=0.1,label='NCV')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
leg=plt.legend(loc='best')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.savefig('./PCA'+str(prefixes[G_choice])+'_allCVvsNCV.png')
#del(all_data,d_idx,leg,lh)
