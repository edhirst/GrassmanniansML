# GrassmanniansML
Generation, data analysis, and machine learning of Grassmannian cluster variables via Young tableaux.  
   
The `Data` folder contains the respective generated cluster variables, as semi-standard Young tableaux, as well as the equivalent non-cluster variable data of tableaux which are not cluster variables.   
...download and unzip the files before running analysis & machine learning.    
   
The python script `HPC_Generation.py` generates cluster variables as semi-standard Young tableaux stochastically via mutation, saving those of ranks under consideration. The file is set-up for parallelisation on a hpc cluster, saving subfiles intermittently which are later combined (taking the union of all variables).  
   
The scripts `ML.py`, `PCA.py`, and `KMeans.py` perform the respective supervised and unsupervised machine learning used to analyse these datasets.   
...ensure the local filepaths are correct for each of the datasets for importing, instructions are given in each script.   
   
The `MiscellaneousAnalysis.py` script contains additional analysis used for results in this research.  




