'''K-Means clustering of Grassmannian cluster algebra datasets'''
'''
Run cells:
Cell 1 ==> import libraries.
Cell 2 ==> import all the data (CV & NCV), and reformat the tableaux.
Select the investigation to perform, with variable investigation ({0,1,2} = {CV vs NCV, all Grs, single Gr}), select the number of clusters to use (if 0 will do elbow method), select which dataset to use if appropriate ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}).
Cell 3 ==> perform the selected k-means clustering
Run misclustering analysis dependend on investigation choice:
    (0) Cell 4 ==> CV vs NCV 
    (1) Cell 5 ==> all Grassmannians
    ...in either case run the respective cell once, note down from the output counts which index corresponds to the correct cluster in each case, and reset the respective 'll' check indices in each case, run the cell again to extract the misclustered tableaux (note this cannot be automated as the clustering process is random and hence inconsistent).
Cell 6 ==> partition the misclustered tableaux by rank for analysis (can be run after either investigation 0 or 1).
'''
### Cell 1 ###
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval as LE
from sklearn.cluster import KMeans
from collections import Counter

#%% ### Cell 2 ###
#Set-up data import
Datachoices = [0,1,2] #...select the indices of the filepaths to import data from (i.e. choose the data to import)
Datafiles = ['./Data/SmallRank6ModulesGr312.txt','./Data/SmallRank6ModulesGr410.txt','./Data/SmallRank4ModulesGr412.txt']
NCVdatafiles = ['./Data/NCV312_10000.txt','./Data/NCV410_10000.txt','./Data/NCV412_10000.txt']
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
#Select the desired investigation: {0,1,2} = {CV vs NCV, all Grs, single Gr}
investigation = 0 

#Perform K-Means
preset_number_clusters = 0   #...set to chosen number of clusters, or to 0 to determine optimal number of clusters
max_number_clusters = 15     #...select the maximum number of clusters to consider in the elbow method
G_choice = 0                 #...select the grassmannian to k-means cluster: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}

#Run on CV vs NCV for selected Grassmannian
if investigation == 0:
    all_data = np.concatenate((Data[G_choice].reshape(Data[G_choice].shape[0],-1),NCV[G_choice].reshape(NCV[G_choice].shape[0],-1))) 
#Run of Grassmannian datasets
if investigation == 1:
    all_data = np.concatenate((Data[0].reshape(Data[0].shape[0],-1),Data[1].reshape(Data[1].shape[0],-1),Data[2].reshape(Data[2].shape[0],-1))) 
#Run on a single Grassmannian dataset
if investigation == 2:
    all_data = Data[G_choice].reshape(Data[G_choice].shape[0],-1)


#Run K-Means Clustering
if preset_number_clusters:
    #Perform K-Means clustering (use preset number of clusters)
    kmeans = KMeans(n_clusters=preset_number_clusters).fit(all_data)  
else:
    #Plot scaled inertia distribution to determine optimal number of clusters
    inertia_list = []
    single_clust_inertia = KMeans(n_clusters=1).fit(all_data).inertia_
    for k in range(1,max_number_clusters+1):
        scaled_inertia = KMeans(n_clusters=k).fit(all_data).inertia_ / single_clust_inertia + 0.01*(k-1)
        inertia_list.append(scaled_inertia)
        
    #Determine optimal number of clusters
    k_optimal = list(range(1,max_number_clusters+1))[inertia_list.index(min(inertia_list))]
    print('Optimal number of clusters: '+str(k_optimal))
    
    #Perform plotting
    plt.figure('K-Means Inertia')
    plt.scatter(list(range(1,max_number_clusters+1)),inertia_list)
    plt.xlabel('Number of Clusters')
    plt.xticks(range(max_number_clusters+1))
    plt.ylabel('Scaled Inertia')
    plt.ylim(0,1.05)
    plt.grid()
    plt.tight_layout()
    plt.savefig('./KMeansInertia_rf_'+str(G_choice)+'.pdf')
    
    #Perform K-Means clustering (use computed optimal number of clusters)
    kmeans = KMeans(n_clusters=k_optimal).fit(all_data)   

#Compute clustering over the data
transformed_full_data = kmeans.transform(all_data)                        #...data transformed to list distance to all centres
kmeans_labels = np.argmin(transformed_full_data,axis=1)                   #...identify the closest cluster centre to each datapoint
full_data_inertia = np.sum([min(x)**2 for x in transformed_full_data])    #...compute the inertia over the full dataset
cluster_sizes = Counter(kmeans_labels)                                    #...compute the frequencies in each cluster
print('\nCluster Centres: '+str(kmeans.cluster_centers_.flatten())+'\nCluster sizes: '+str(cluster_sizes)+'\n\nInertia: '+str(full_data_inertia)+'\nNormalised Inertia: '+str(full_data_inertia/len(all_data.flatten()))+'\nNormalised Inertia / range: '+str(full_data_inertia/(len(all_data.flatten())*(max(all_data.flatten())-min(all_data.flatten())))))

#%% ### Cell 4 ### (investigation 0)
#Count how many of each CV/NCV tableaux in each cluster, and record those misclustered
Cluster_counts = [np.zeros(preset_number_clusters,dtype='int'),np.zeros(preset_number_clusters,dtype='int')]
misclustered = [[] for i in range(preset_number_clusters)]
for idx, ll in enumerate(kmeans_labels[:len(Data[G_choice])]):
    Cluster_counts[0][ll] += 1
    if ll == 1: misclustered[0].append(Data[G_choice][idx]) #...this index is hardcoded from a previous run to know what cluster each GR should be in
for idx, ll in enumerate(kmeans_labels[len(Data[G_choice]):]):
    Cluster_counts[1][ll] += 1
    if ll == 0: misclustered[1].append(NCV[G_choice][idx]) #...this index is hardcoded from a previous run to know what cluster each GR should be in

print(Cluster_counts)

#%% ### Cell 5 ### (investigation 1)
#Count how many of each Grassmannian in each cluster, and record those misclustered
Cluster_counts = [np.zeros(preset_number_clusters,dtype='int') for dset in Datachoices]
misclustered = [[] for i in range(preset_number_clusters)]
for idx, ll in enumerate(kmeans_labels[:len(Data[0])]):
    Cluster_counts[0][ll] += 1
    if ll != 1: misclustered[0].append(Data[0][idx]) #...this index is hardcoded from a previous run to know what cluster each GR should be in
for idx, ll in enumerate(kmeans_labels[len(Data[0]):len(Data[0])+len(Data[1])]):
    Cluster_counts[1][ll] += 1
    if ll != 2: misclustered[1].append(Data[1][idx]) #...this index is hardcoded from a previous run to know what cluster each GR should be in
for idx, ll in enumerate(kmeans_labels[len(Data[0])+len(Data[1]):]):
    Cluster_counts[2][ll] += 1
    if ll != 0: misclustered[2].append(Data[2][idx]) #...this index is hardcoded from a previous run to know what cluster each GR should be in
    
print(Cluster_counts)

#%% ### Cell 6 ###
#Rank partition the misclustered data
misclustered_rank = np.zeros(6)
for tab in misclustered[0]:
    if   tab[0][1] == 0: misclustered_rank[0]+=1
    elif tab[0][2] == 0: misclustered_rank[1]+=1
    elif tab[0][3] == 0: misclustered_rank[2]+=1
    elif tab[0][4] == 0: misclustered_rank[3]+=1
    elif tab[0][5] == 0: misclustered_rank[4]+=1
    else:                misclustered_rank[5]+=1

print(misclustered_rank)
print(sum(misclustered_rank))
