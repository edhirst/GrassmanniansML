'''Code to import, reformat, and ML the Grassmannian cluster variables as SSYT'''
'''
Run cells:
Cell 1 ==> import libraries.
Cell 2 ==> import all the data (CV & NCV), reformat the tableaux, and sample the CV data so the datasets are balanced for ML.
Then choose the investigation: 
(1) Binary classify CV vs NCV SSYT (CV means a cluster variable, NCV not a cluster variable)
    Cell 3 ==> select which of the 3 datasets to run the ML for ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), format the data for k-fold cross-validation.
    Choose which architecture to train & test:
        Cell 4 ==> SVM
        Cell 5 ==> NN
(2) Multiclassification between the 3 Grassmannians
    Cell 6 ==> format the data for k-fold cross-validation.
    Cell 7 ==> train & test the NN architecture.
Cell 8 ==> partition the misclassified tableaux by rank for analysis (can be run after any architecture train & test).
'''
### Cell 1 ###
#Import libraries
import numpy as np
from ast import literal_eval as LE
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import matthews_corrcoef as MCC

#%% ### Cell 2 ###
#Set-up data import
Datachoices = [0,1,2] #...select the indices of the filepaths to import data from (i.e. choose the data to import)
CVdatafiles =  ['./Data/CVData/CV_Gr312_Rank6.txt',  './Data/CVData/CV_Gr410_Rank6.txt',  './Data/CVData/CV_Gr412_Rank4.txt']
NCVdatafiles = ['./Data/NCVData/NCV_Gr312_Rank6.txt','./Data/NCVData/NCV_Gr410_Rank6.txt','./Data/NCVData/NCV_Gr412_Rank4.txt']
prefixes = [f[-7:-4] for f in CVdatafiles]

#Import Grassmannians
CV = []
for datapath in Datachoices:
    CV.append([])
    with open(CVdatafiles[datapath],'r') as file:
        for line in file.readlines():
            CV[-1].append(LE(line))
print('Dataset sizes: '+str(list(map(len,CV))))
del(file,line,datapath)

#Sample data
sample_size = 10000 #...set to 0 to use full dataset
if sample_size:
    #Save the full data elsewhere to avoid reimporting
    from copy import deepcopy as dc
    Data_backup = dc(CV)
    #Sample SSYTs
    for dataset_idx in range(len(CV)):
        CV[dataset_idx] = [CV[dataset_idx][i] for i in np.random.choice(len(CV[dataset_idx]),sample_size,replace=False)]

#Pad data (all datasets to the same size)
max_height = max([max(map(len,dataset)) for dataset in CV])
max_width  = max([max(map(len,[tab[0] for tab in dataset])) for dataset in CV]) #...all tableaux rows the same length so can just consider first row
for dataset_idx in range(len(CV)):
    for tab_idx in range(len(CV[dataset_idx])):
        tab = np.array(CV[dataset_idx][tab_idx])
        new_tab = np.zeros((max_height,max_width),dtype=int)
        new_tab[:len(tab),:len(tab[0])] = tab
        CV[dataset_idx][tab_idx] = new_tab
    CV[dataset_idx] = np.array(CV[dataset_idx])
del(dataset_idx,tab_idx,tab,new_tab)

#Import NCV data
NCV = []
for datapath in Datachoices:
    NCV.append([])
    with open(NCVdatafiles[datapath],'r') as file:
        NCV[-1] = LE(file.read())
    NCV[-1] = np.array(NCV[-1])
print('NCV dataset sizes: '+str(list(map(len,NCV))))
del(file,datapath)

########################################################
#%% ### Cell 3 ###
#Binary classify the CV vs NCV with SVM & NN
#Format the data
G_choice = 0 #...select which dataset to run the ML for: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}
X = np.concatenate((CV[G_choice],NCV[G_choice]),axis=0)
X = X.reshape(X.shape[0],-1) #...flatten the matrices
Y = np.concatenate((np.ones(len(CV[G_choice])),np.zeros(len(NCV[G_choice]))))

#Zip data together & shuffle
data_size = len(Y)
ML_data = [[X[index],Y[index]] for index in range(data_size)]
np.random.shuffle(ML_data)
k = 5   #... k = 5 => 80(train) : 20(test) splits approx.
s = int(np.floor(data_size/k)) #...number of datapoints in each validation split

#Separate for cross-validation
X_train, X_test, Y_train, Y_test = [], [], [], []
for i in range(k):
    X_train.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Y_train.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    X_test.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Y_test.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

del(ML_data,X,Y)

#%% ### Cell 4 ###
#Create, Train, & Test SVM
#Define the learning measure lists
accs, mccs, cms = [], [], []
misclassifications = [[[],[]] for i in range(k)]

#Define & Train Neural Network directly on the data
for svm_idx in range(k):
    print('SVM:',svm_idx+1)
    svm_clf = svm.SVC(C=1,kernel='linear')
    svm_clf.fit(X_train[svm_idx], Y_train[svm_idx]) 
    print('...trained')
    #Calculate predictions directly with Neural Network
    Y_pred_svm = svm_clf.predict(X_test[svm_idx])
    cms.append(CM(Y_test[svm_idx],Y_pred_svm,normalize='all'))
    accs.append(np.sum(Y_pred_svm == Y_test[svm_idx])/len(Y_test[svm_idx]))
    mccs.append(MCC(Y_test[svm_idx],Y_pred_svm))
    #Save the misclassifications
    for tab_idx in range(len(Y_test[svm_idx])):
        if Y_test[svm_idx][tab_idx] != Y_pred_svm[tab_idx]:
            misclassifications[svm_idx][int(Y_test[svm_idx][tab_idx])].append(X_test[svm_idx][tab_idx])

#Reformat misclassifications
misclassifications = [[np.array(i[0],dtype=int),np.array(i[1],dtype=int)] for i in misclassifications]
misclassifications = [[i[0].reshape(i[0].shape[0],4,6),i[1].reshape(i[1].shape[0],4,6)] for i in misclassifications]
  
#Output averaged learning measures with standard errors
print('Average measures:')
print('Accuracy:',sum(accs)/k,'\pm',np.std(accs)/np.sqrt(k))
print('MCC:',sum(mccs)/k,'\pm',np.std(mccs)/np.sqrt(k))
cms=np.array(cms)
print('CM:\n',np.mean(cms,axis=0),'\n\pm\n',np.std(cms,axis=0)/np.sqrt(k))
print('Misclassifications:',[[len(i[0]),len(i[1])] for i in misclassifications])

#%% ### Cell 5 ###
#Create, Train, & Test NN Classifier
#Define the learning measure lists
accs, mccs, cms = [], [], []
misclassifications = [[[],[]] for i in range(k)]

#Define & Train Neural Network directly on the data
for nn_idx in range(k):
    print('Network:',nn_idx+1)
    nn_clf = MLPClassifier((16,32,16))
    nn_clf.fit(X_train[nn_idx], Y_train[nn_idx]) 
    print('...trained')
    #Calculate predictions directly with Neural Network
    Y_pred_nn = nn_clf.predict(X_test[nn_idx])
    cms.append(CM(Y_test[nn_idx],Y_pred_nn,normalize='all'))
    accs.append(np.sum(Y_pred_nn == Y_test[nn_idx])/len(Y_test[nn_idx]))
    mccs.append(MCC(Y_test[nn_idx],Y_pred_nn))
    #Save the misclassifications
    for tab_idx in range(len(Y_test[nn_idx])):
        if Y_test[nn_idx][tab_idx] != Y_pred_nn[tab_idx]:
            misclassifications[nn_idx][int(Y_test[nn_idx][tab_idx])].append(X_test[nn_idx][tab_idx])

#Reformat misclassifications
misclassifications = [[np.array(i[0],dtype=int),np.array(i[1],dtype=int)] for i in misclassifications]
misclassifications = [[i[0].reshape(i[0].shape[0],4,6),i[1].reshape(i[1].shape[0],4,6)] for i in misclassifications]
  
#Output averaged learning measures with standard errors
print('Average measures:')
print('Accuracy:',sum(accs)/k,'\pm',np.std(accs)/np.sqrt(k))
print('MCC:',sum(mccs)/k,'\pm',np.std(mccs)/np.sqrt(k))
cms=np.array(cms)
print('CM:\n',np.mean(cms,axis=0),'\n\pm\n',np.std(cms,axis=0)/np.sqrt(k))
print('Misclassifications:',[[len(i[0]),len(i[1])] for i in misclassifications])

########################################################
#%% ### Cell 6 ###
#Classification between the Grassmannians
#Format the data
X = np.concatenate((CV[0],CV[1],CV[2]))
X = X.reshape(X.shape[0],-1)
Y = np.concatenate((np.zeros(len(CV[0])),np.ones(len(CV[1])),np.full(len(CV[2]),2)))

#Zip data together & shuffle
data_size = len(Y)
ML_data = [[X[index],Y[index]] for index in range(data_size)]
np.random.shuffle(ML_data)
k = 5   #... k = 5 => 80(train) : 20(test) splits approx.
s = int(np.floor(data_size/k)) #...number of datapoints in each validation split

#Separate for cross-validation
X_train, X_test, Y_train, Y_test = [], [], [], []
for i in range(k):
    X_train.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Y_train.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    X_test.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Y_test.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

del(ML_data,X,Y)

#%% ### Cell 7 ###
#Create, Train, & Test NN Classifier
#Define the learning measure lists
accs, mccs, cms = [], [], []
misclassifications = [[[[] for i in range(3)] for i in range(3)] for i in range(k)]

#Define & Train Neural Network directly on the data
for nn_idx in range(k):
    print('Network:',nn_idx+1)
    nn_clf = MLPClassifier((16,32,16))
    nn_clf.fit(X_train[nn_idx], Y_train[nn_idx]) 
    print('...trained')
    #Calculate predictions directly with Neural Network
    Y_pred_nn = nn_clf.predict(X_test[nn_idx])
    cms.append(CM(Y_test[nn_idx],Y_pred_nn,normalize='all'))
    accs.append(np.sum(Y_pred_nn == Y_test[nn_idx])/len(Y_test[nn_idx]))
    mccs.append(MCC(Y_test[nn_idx],Y_pred_nn))
    #Save the misclassifications
    for tab_idx in range(len(Y_test[nn_idx])):
        if Y_test[nn_idx][tab_idx] != Y_pred_nn[tab_idx]:
            misclassifications[nn_idx][int(Y_test[nn_idx][tab_idx])][int(Y_pred_nn[tab_idx])].append(X_test[nn_idx][tab_idx].tolist())
  
#Output averaged learning measures with standard errors
print('Average measures:')
print('Accuracy:',sum(accs)/k,'\pm',np.std(accs)/np.sqrt(k))
print('MCC:',sum(mccs)/k,'\pm',np.std(mccs)/np.sqrt(k))
cms=np.array(cms)
print('CM:\n',np.mean(cms,axis=0),'\n\pm\n',np.std(cms,axis=0)/np.sqrt(k))
print('Misclassifications:\n',np.array([[len(i) for j in k for i in j] for k in misclassifications]).reshape(k,3,3))

########################################################
#%% ### Cell 8 ###
#Rank partition the misclassified data
misclassified_rank = np.zeros(6)
for arch_idx in range(k):
    for error_idx in range(2):
        for tab in misclassifications[arch_idx][error_idx]:
            if   tab[0][1] == 0: misclassified_rank[0]+=1
            elif tab[0][2] == 0: misclassified_rank[1]+=1
            elif tab[0][3] == 0: misclassified_rank[2]+=1
            elif tab[0][4] == 0: misclassified_rank[3]+=1
            elif tab[0][5] == 0: misclassified_rank[4]+=1
            else:                misclassified_rank[5]+=1
        
        print('k '+str(arch_idx+1)+', error '+str(error_idx+1)+': '+str(misclassified_rank))
