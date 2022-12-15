'''Saliency Analysis of NN performance CV vs NCV'''
'''
Run cells:
Cell 1 ==> import libraries.
Cell 2 ==> import all the data (CV & NCV), and reformat the tableaux.
Cell 3 ==> select which of the 3 datasets to run the ML for ({0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}), set-up the data accordingly for the ML.
Cell 4 ==> run the ML using NNs (now from tensorflow but with the same hyperparameters), note results are negligibly different.
Cell 5 ==> perform and output the gradient saliency analysis for the NN.
...Cells 6-9 ==> further functionality to extract the most salient features, and run symbolic regression on them (no suitably accuracte equation can be found).
'''
### Cell 1 ###
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from ast import literal_eval as LE

#%% ### Cell 2 ###
#Data import
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

#%% ### Cell 3 ###
#Data set-up
#Format the data
G_choice = 0 #...select which dataset to run the ML for: {0,1,2} = {Gr(3,12)r6, Gr(4,10)r6, Gr(4,12)r4}
X = np.concatenate((CV[G_choice],NCV[G_choice]),axis=0)
X = X.reshape(X.shape[0],-1) #...flatten the matrices
Y = np.concatenate((np.ones(len(CV[G_choice])),np.zeros(len(NCV[G_choice]))))

#Zip data together & shuffle
data_size = len(Y)
s = int(np.floor(data_size/5)) #...number of datapoints in each validation split
ML_data = [[X[index],Y[index]] for index in range(data_size)]
np.random.shuffle(ML_data)

#Separate for cross-validation
X_train = np.array([datapoint[0] for datapoint in ML_data[s:]],dtype=float)
Y_train = np.array([datapoint[1] for datapoint in ML_data[s:]],dtype=float)
X_test  = np.array([datapoint[0] for datapoint in ML_data[:s]],dtype=float)
Y_test  = np.array([datapoint[1] for datapoint in ML_data[:s]],dtype=float)

del(ML_data,X,Y)

#%% ### Cell 4 ###
#Run ML
#Define NN hyper-parameters
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation
number_of_epochs = 20           #...number of times to run training data through NN
size_of_batches = 200            #...number of datapoints the NN sees per iteration of optimiser (high batch means more accurate param updating, but less frequently) 
layer_sizes = [16,32,16]     #...number and size of the dense NN layers

#Define lists to record training history and learning measures
hist_data = []               #...training data (output of .fit(), used for plotting)

#Setup NN
model = keras.Sequential()
for layer_size in layer_sizes:
    model.add(keras.layers.Dense(layer_size, activation=act_fn))
    #model.add(keras.layers.Dropout(0.1)) #...dropout layer to reduce chance of overfitting to training data
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
#Train NN
hist_data.append(model.fit(X_train, Y_train, batch_size=size_of_batches, epochs=number_of_epochs, shuffle=True, validation_split=0., verbose=0))
#Test NN
predictions = np.ndarray.flatten(np.round(model.predict(X_test)))
accuracy = 1-np.mean(np.absolute(Y_test-predictions)) 
print('Accuracy:',accuracy)

#%% ### Cell 5 ###
#Gradient Saliency
image = tf.Variable(X_test)
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(image)
    predictions = model(image)
    loss = predictions
    
#Compute the gradient of the loss wrt the input image (and convert to numpy)
gradient = tape.gradient(loss, image)
gradient = gradient.numpy()
avg_grad = np.absolute(np.mean(gradient,axis=0)).reshape((4,6))
print('Average Gradients:\n',avg_grad)
plt.axis('off')
plt.imshow(avg_grad)
#plt.savefig('./SaliencyImage_'+str(G_choice)+'.pdf')

########################################################
#%% ### Cell 6 ###
#Extract the data entries which were most salient (bottom-left and top-right)
G_choice = 0
if G_choice == 0: bl_idx = 2 #...when using Gr(3,12) ignore trivially padded last row
else: bl_idx = 3
CV_parts, NCV_parts = [], []

for tab in CV[G_choice]:
    bl = tab[bl_idx][0]
    if   tab[0][1] == 0: tr = tab[0][0]
    elif tab[0][2] == 0: tr = tab[0][1]
    elif tab[0][3] == 0: tr = tab[0][2]
    elif tab[0][4] == 0: tr = tab[0][3]
    elif tab[0][5] == 0: tr = tab[0][4]
    else: tr = tab[0][5]
    CV_parts.append([bl,tr])
CV_parts = np.array(CV_parts)
  
for tab in NCV[G_choice]:
    bl = tab[bl_idx][0]
    if   tab[0][1] == 0: tr = tab[0][0]
    elif tab[0][2] == 0: tr = tab[0][1]
    elif tab[0][3] == 0: tr = tab[0][2]
    elif tab[0][4] == 0: tr = tab[0][3]
    elif tab[0][5] == 0: tr = tab[0][4]
    else: tr = tab[0][5]
    NCV_parts.append([bl,tr])
NCV_parts = np.array(NCV_parts)

#%% ### Cell 7 ###
#Plot cross correlations of these entries - note swapping the scatter order shows there is actually substantial overlap (most entries)
plt.figure('Entry correlations')
plt.scatter(CV_parts[:,0], CV_parts[:,1], alpha=0.1,color='blue',label='CV' )
plt.scatter(NCV_parts[:,0],NCV_parts[:,1],alpha=0.1,color='red', label='NCV')
plt.xlabel('bottom-left')
plt.ylabel('top-right')
plt.grid()
plt.tight_layout()
leg=plt.legend(loc='best')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
#plt.savefig('./.pdf')

########################################################
#%% ### Cell 8 ###
#Symbolic Regression on these entries
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sympy import sympify

#Set-up Data
X2 = np.vstack((CV_parts,NCV_parts))
Y2 = np.concatenate((np.ones(len(CV[G_choice])),np.zeros(len(NCV[G_choice]))))
X2_train, X2_test, Y2_train, Y2_test = tts(X2, Y2, test_size=0.2, random_state=2)

#%% ### Cell 9 ###
#Define and fit the Sregressor
#Choose functions from ['add','sub','mul','div','neg','sqrt','log','abs','inv','max','min','sin','cos','tan']
SR = SymbolicRegressor(population_size=5000, function_set=['add','sub','mul','div','abs'], metric='mean absolute error', generations=30, stopping_criteria=0.01, const_range=(-10,10),
                       p_crossover=0.75, p_subtree_mutation=0.025, p_hoist_mutation=0.015, p_point_mutation=0.02,
                       max_samples=1, verbose=1, parsimony_coefficient=0.015)#, random_state=1), #...usually metric='mean absolute error', but code below to make metric=mape work

SR.fit(X2_train, Y2_train)

#Test the Sregressor
Y2_pred = SR.predict(X2_test)
Score = SR.score(X2_test, Y2_test)
print('R^2:\t',Score)
print('MAE:\t',MAE(Y2_test,Y2_pred))
print('MAPE:\t',MAPE(Y2_test,Y2_pred))

#Output the final equation ---> needs sympy
converter = {
    'add' : lambda x, y : x + y,
    'sub' : lambda x, y : x - y,
    'mul' : lambda x, y : x*y,
    'div' : lambda x, y : x/y,
    'neg' : lambda x    : -x,
    'sqrt': lambda x    : x**0.5,
    'log' : lambda x    : log(x),
    'abs' : lambda x    : abs(x),
    'inv' : lambda x    : 1/x,
    'max' : lambda x    : max(x),
    'min' : lambda x    : min(x),
    'sin' : lambda x    : sin(x),
    'cos' : lambda x    : cos(x),
    'tan' : lambda x    : tan(x)
}
Eq = sympify(str(SR._program), locals=converter) #Eq = str(SR._program)
print('Equation:',Eq)
