'''Generating grassmannians tableaux'''
'''Note this is a sagemath script, generation parameters are set in the final section'''
#Import libraries
import sys
import numpy as np
import itertools
from numpy import matrix 
from itertools import combinations as comb
from operator import add
from typing import List
from sage.combinat.shuffle import ShuffleProduct
#from multiprocess import Pool
from sage.parallel.multiprocessing_sage import Pool
 
####################################################################
#Define relevant functions

def TableauToMatrixTakeRows(a): 
    # a[i] are rows of the tableau a
    # transform a tableau to matrix form
    m=len(a)
    n=len(a[0])
    r=Matrix(m,n)
    for i in range(m):
        for j in range(n):
            r[i,j]=a[i][j]
    return r

def PromotionOfTableauNTimes(N,T1,n):
    # promotion of a tableau N times
    r=[T1]
    T=T1
    for i in range(N-1):
        T=T.promotion(n-1)
        r.append(T)
    return r 

def PromotionOfTableauNTimesInMatrix(N,T1,n):
    # promotion of tableau n times, T1 is in matrix form
    t1=MatrixTakeRows(T1)
    t2=SemistandardTableau(t1)
    r1=PromotionOfTableauNTimes(N,t2,n)
    r=[]
    for i in r1:
        r.append(TableauToMatrixTakeRows(i))
        
    return r 

def PluckerToMinimalAff(a1):
    # translate a Plucker coordinate to highest weight monomial
    r=[]
    a=sorted(a1)
    n=len(a)
    for i in range(n-1,0,-1):
        r.append(a[i]-a[i-1]-1)
    r.append(a[0]-1)

    return r

def InitialCluster(rank,n):  
    # initial cluster from Gr(rank, n)
    sizeColumn=n-rank
    k=sizeColumn    
    k1=rank
    p1=k*rank+1
    mat=Matrix(p1,p1)
    for i in range(p1): 
        i1=i+1
        if i1==1: 
            mat[i,i+k+1]=1
            mat[i, p1-1]=1
            mat[i,i+k]=-1
            mat[i, i+1]=-1
        elif i1>=2 and i1<=k-1: 
            mat[i,i+1]=-1
            mat[i,i+k]=-1
            mat[i,i-1]=1
            mat[i,i+k+1]=1
        elif i1==k: 
            mat[i,i-1]=1
            mat[i,i+k]=-1
        elif i1>k and i1<(rank-1)*k and i1 % k==1: 
            mat[i,i-k]=1
            mat[i,i+k+1]=1
            mat[i,i+1]=-1
            mat[i,i+k]=-1
        elif i1>k and i1<(rank-1)*k and i1 % k >=2 and i1 % k<=k-1:
            mat[i,i-k-1]=-1
            mat[i,i+1]=-1
            mat[i,i+k]=-1
            mat[i,i-k]=1
            mat[i,i-1]=1
            mat[i,i+k+1]=1
        elif i1>=2*k and i1<=(rank-1)*k and i1 % k==0:
            mat[i,i-k-1]=-1
            mat[i,i+k]=-1
            mat[i,i-k]=1
            mat[i,i-1]=1
        elif i1>(rank-1)*k and i1<p1 and i1 % k==1: 
            mat[i,i-k]=1
            mat[i,i+1]=-1
        elif i1>=(rank-1)*k+2 and i1<rank*k:
            mat[i,i-k-1]=-1
            mat[i,i+1]=-1
            mat[i,i-k]=1
            mat[i,i-1]=1
        elif i1==rank*k:
            mat[i,i-k-1]=-1
            mat[i,i-k]=1
            mat[i,i-1]=1
        elif i1==p1:
            mat[i,0]=-1

    vertices0=[]

    for j in range(k1-1,-1,-1):
        for i in range(k1,k1+k):
            t1=list(range(1,j+1))
            t2=list(range(i-k1+j+2,i+2))
            t3=t1+t2
            vertices0.append(t3)

    vertices0.append(list(range(1,k1+1)))  
    verticesTableaux = [] # Tableaux are represented by matrices
    for i in range(len(vertices0)):
        verticesTableaux.append([0, [vertices0[i]], i]) # [vertices0[i]] is an one column tableau
    mat1 = Matrix(p1,p1)
    for i in range(p1):
        for j in range(p1):
            mat1[i,j]=mat[i,j]
    clusterVariables=[] 
    vertices1 = [verticesTableaux, clusterVariables] # vertices1[1] store cluster variables, vertices1[0] store variables on quiver
    r=[mat, vertices1]
    
    return r


def TableauExpansionsInMatrixHalf(l,b,c): 
    # l is tableau in matrix form, b is the content of l, c is a list of numbers
    # replace a_1<...<a_m in l by c_1<...<c_m
    r=[]
    m=l.nrows()
    n=l.ncols()
    r=Matrix(m,n)
    for i in range(m):
        for j in range(n):
            t1=b.index(l[i,j])+1
            r[i,j]=c[t1-1]
    return r

def TableauExpansionsInMatrix(l,n): 
    # l is tableau in matrix form
    # replace a_1<...<a_m in l by c_1<...<c_m
    r1=ContentOfTableau(l)
    m=len(r1)
    r2=list(itertools.combinations(list(range(1,n+1)), m)) 
    r=[]
    for i in r2:
        t1=TableauExpansionsInMatrixHalf(l,r1,i)
        r.append(t1)
    return r

def TableauExpansionsInMatrixList(l,n): 
    # l is a list of tableaux in matrix form
    r=[]
    for i in l:
        r=r+TableauExpansionsInMatrix(i,n)
    r=removeDuplicates2(r)  
    
    return r

def ContentOfTableau(l): 
    # l is tableau
    # compute the content of l, with multiplicities
    r=[]
    for i in l:
        for j in i:
            r.append(j)
    #r=np.unique(r,axis=0)
    r=removeDuplicatesListOfLists(r)
    r=sorted(r)
    return r

def immutabilize(m):
    M = copy(m)
    M.set_immutable()
    return M

def ChangeListOfMatricesToSetOfMatrices(S):
    r={immutabilize(i) for i in S}
    return r

def removeAnElementInList(i, l):
    r=[]
    for j in range(len(l)):
        if (j!=i):
            r.append(l[j])
    
    return r

def removeDuplicates(l):
    # remove duplicates
    # it is slow whn l is large
    r=[]
    for i in l:
        if (i in r)==False:
            r.append(i)
    return r

def removeDuplicates2(l): 
    # remove duliplictes
    # vary fast, for matrices
    t1=ChangeListOfMatricesToSetOfMatrices(l)
    r=list(dict.fromkeys(t1))
    return r

def removeDuplicatesListOfLists(l): 
    # very fast
    l.sort()
    r=list(l for l,_ in itertools.groupby(l))

    return r

def SetDifference2(a,b):
    # take different of two lists a, b
    t1=ChangeListOfMatricesToSetOfMatrices(a)
    t2=ChangeListOfMatricesToSetOfMatrices(b)
    r=t1.difference(t2)
    return r

def SetDifferenceListDifference(A,B): 
    # A-B, can have duplicate elements
    # take different of two lists A, B, count multiplicites
    r=[]
    r1=list(set(A))
    for i in r1:
        t1=A.count(i)-B.count(i)
        #print(t1)
        for j in range(1,t1+1):
            r.append(i)
            
    return r

def TableauToMatrix(a):
    # transform a tableau to matrix form
    m=len(a)
    n=len(a[0])
    r=Matrix(n,m)
    for i in range(n):
        for j in range(m):
            r[i,j]=a[j][i]
    return r

def MatrixTakeRows(a):
    # take rows of a matrix to get a list
    n=a.nrows()
    m=a.ncols()
    r=[]
    for i in range(n):
        t1=a[[i],list(range(m))]
        t2=[]
        for j in range(m):
            t2.append(t1[0,j])
        r.append(t2)
    return r

def MatrixTakeRowsList(a):
    # function MatrixTakeRows for a list of matrices
    r=[]
    for i in a:
        r.append(MatrixTakeRows(i))
    return r

def TableauDivision(a,b):
    # division of two tableaux a, b, that is removing b from a
    t1=TableauToMatrix(a)
    t2=TableauToMatrix(b)  
    r1=MatrixTakeRows(t1)
    r2=MatrixTakeRows(t2)
    r3=[]
    for i in range(len(r1)):
        r3.append(sorted(SetDifferenceListDifference(r1[i],r2[i])))
    r=[]
    for i in range(len(r3[0])):
        t1=[]
        for j in range(len(r3)):
            t1.append(r3[j][i])
        r.append(t1)
        
    return r

def UnionOfTwoTableaux(a,b):
    t1=a+b
    t2=TableauToMatrix(t1)
    r=[]
    for i in range(t2.nrows()):
        r1=[]
        for j in range(t2.ncols()):
            r1.append(t2[i,j])
        r.append(sorted(r1))
        
    r2=TableauToMatrix(r);
    r=[]
    for i in range(r2.nrows()):
        r1=[]
        for j in range(r2.ncols()):
            r1.append(r2[i,j])
        r.append(sorted(r1))
    return r
        
def PowerOfTableaux(a,n):
    r=[]
    if a!=[] and a!=[[]]:
        for i in range(1,n+1):
            r=UnionOfTwoTableaux(r,a) 
    else:
        r=a 
    return r

def CartanMatrixSelfDefined(typ, rank):
    if typ=='E' and rank==6:
        r=Matrix([[2,0,-1,0,0,0],[0,2,0,-1,0,0],[-1,0,2,-1,0,0],[0,-1,-1,2,-1,0],[0,0,0,-1,2,-1],[0,0,0,0,-1,2]]) # this is the Cartan Matrix in Sage of type E6
    else:
    
        r = Matrix(rank, rank)
        n = rank
        for i in range(n):
            if i + 1 <= n-1:
                r[i, i + 1] = -1
            if 0 <= i - 1:
                r[i, i - 1] = -1
            r[i, i] = 2

        if typ == 'B' or typ == 2:
            r[n-1, n - 2] = -2
        elif typ == 'C' or typ == 3:
            r[n - 2, n-1] = -2
        elif typ == 'D' or typ == 4:
            if n == 2:
                r[0, 1] = 0
                r[1, 0] = 0
            elif 3 <= n:
                r[n - 3, n - 2] = -1
                r[n - 3, n-1] = -1
                r[n - 2, n - 3] = -1
                r[n-1, n - 3] = -1
                r[n - 2, n-1] = 0
                r[n-1, n - 2] = 0
        elif typ == 'E' or typ == 5:
            for k in [[2, 4], [4, 2]]:
                r[k[0], k[1]] = -1
            for k in [[3, 4], [4, 3]]:
                r[k[0], k[1]] = 0
        elif typ == 'F' or typ == 6:
            r[1, 2] = -2
        elif typ == 'G' or typ == 7:
            r[0, 1] = -3
     
    return r 

def compareWeightsTableaux(P1,P2,typ,rank): 
    # a,b are tableaux
    t1=WeightOfTableau(P1)
    t2=WeightOfTableau(P2)
    r=compareWeights2(t1,t2,typ, rank)

    return r

def WeightOfTableau(a): 
    # a[i] are columns of the tableau a
    m=len(a)
    n=len(a[0])
    r=[]
    for i in range(1,n+1):
        r.append(0)
    for i in range(m):
        t1=PluckerToMinimalAff(a[i])
        #r=list(np.array(r)+np.array(t1))
        r=list(map(add, r, t1))
        
    return r

def compareWeights(a, b, typ, rank): 
    # compare two weights
    r=1                             # r=1 means a>=b          
    l=a-b
    c=CartanMatrixSelfDefined(typ, rank)
    for i in range(rank): 
        p=0
        for j in range(rank):
            t1=(transpose(c)^(-1))[j,i]
            p=p+l[j,0]*t1
        if p<0: 
            r=-1              # r=-1 means a is not >= b, it is possible that a<b or a,b are not comparable
            break 
            
    if r==-1:
        for i in range(rank):
            p=0
            for j in range(rank):
                t1=(transpose(c)^(-1))[j,i]
                p=p+l[j,0]*t1
            if p>0: 
                r=0
                break
    return r

def compareWeights2(a,b,typ,rank): 
    # a,b are lists
    n=len(a)
    t1=Matrix(n,1)
    for i in range(n): 
        t1[i,0]=a[i] 
    t2=Matrix(n,1)
    for i in range(n): 
        t2[i,0]=b[i] 
    r=compareWeights(t1,t2,typ,rank)

    return r

def matrixMutation(mat,  k):  
    # matrix mutates at k
    size=mat.nrows()
    r=Matrix(size,size)
    for i in range(size):
        for j in range(size):
            r[i,j]=mat[i,j]
    
    for i in range(size): 
        for j in range(size): 
            
            if k==i or k==j:
                r[i,j]=-mat[i, j]    
            else: 
                r[i, j] = mat[i, j]+1/2*(abs(mat[i,k])*mat[k,j]+mat[i,k]*abs(mat[k,j]))
     
    return r

def ExtendSetOfTableauxToContainPromotions(l,n): 
    # l is a list of tableaux 
    # extend the set l to include their promotions
    r=[]
    for i in l:
        t1=PromotionOfTableauNTimes(n,i,n)
        r=r+t1 
    r=np.unique(r,axis=0)

    return r

def ExtendSetOfTableauxToContainPromotionsInMatrix(l,n): 
    # l is a list of tableaux in matrix form
    # extend the set l to include their promotions
    r=[]
    for i in l:
        t1=PromotionOfTableauNTimesInMatrix(n,i,n)
        r=r+t1 
    r=removeDuplicates2(r)
    
    return r

def TableauxToListOfTimesOfOccurrenceOfNumbers(a):
    # compute occurrences of numbers in tableau a
    r=[]
    n=a.nrows() 
    m=a.ncols() 
    r1=[]
    for i in range(n):  
        for j in range(m): 
            r1.append(a[i,j]) 
    for k in range(1,max(r1)+1):
        t1=0
        for i in r1:
            if i==k:
                t1=t1+1 
        r.append(t1)  
    return r

def TableauxToListOfTimesOfOccurrenceOfNumbersLengthN(a,N):
    r=[]
    n=a.nrows() 
    m=a.ncols() 
    r1=[]
    for i in range(n):  
        for j in range(m): 
            r1.append(a[i,j]) 
    for k in range(1,N):
        t1=0
        for i in r1:
            if i==k:
                t1=t1+1 
        r.append(t1)
    return r
    
def TableauxToListOfTimesOfOccurrenceOfNumbersLengthNWithContentLessOrEquN(a,N): 
    # compute the occurrences of numbers in i for those i in a such that the numbers in i is less or equal to N
    r=[]
    n=a.nrows() 
    m=a.ncols() 
    r1=[]
    for i in range(n):  
        for j in range(m): 
            r1.append(a[i,j])       
    if max(r1)<=N:
        for k in range(1,N):
            t1=0
            for i in r1:
                if i==k:
                    t1=t1+1 
            r.append(t1)
    return r
    
def TableauxToListOfTimesOfOccurrenceOfNumbersTableauIsList(a):
    t1=TableauToMatrix(a)
    r=TableauxToListOfTimesOfOccurrenceOfNumbers(t1)
 
    return r

def TableauxToListOfTimesOfOccurrenceOfNumbersLengthNTableauIsList(a,N):
    t1=TableauToMatrix(a)
    r=TableauxToListOfTimesOfOccurrenceOfNumbersLengthN(t1,N)
 
    return r


def computeEquationsForModulesTableaux(variable2, mat, k, typ, rank): 
    # mutation of Grassmannian cluster variables
    # variable2=(variables on quiver, cluster variables obtained so far)
    variable1=variable2[0]
    clusterVariables=variable2[1] 
    size=mat.nrows() 
    newVariable=[]
    newVariable2=[]
    variable=variable1

    for i in range(size):
        if mat[i, k]>0:
            newVariable=UnionOfTwoTableaux( newVariable, PowerOfTableaux(variable[i][1], mat[i,k]) )
 
    for i in range(size): 
        if mat[i, k]<0:
            newVariable2= UnionOfTwoTableaux( newVariable2, PowerOfTableaux(variable[i][1], -mat[i,k]) )
 
    variable[k][0]=variable[k][0]+1
    t1=compareWeightsTableaux(newVariable, newVariable2,typ,rank)
 
    if t1==1: 
        variable[k][1]=TableauDivision(newVariable, variable[k][1])
    else:
        variable[k][1]=TableauDivision(newVariable2, variable[k][1])  
        
    clusterVariables=TableauToMatrix(variable[k][1]) 
    
    r=[variable, clusterVariables]

    return r


def ll_perms(lli,typ,rank,max_column,n,repeat): 
    #Function for multiprocessing
    b1=[]
    IC=InitialCluster(rank,n)
    mat1=IC[0]
    vertices1=IC[1]
    ll=list(map(lambda x: x - 1, lli))
    
    mutationSequence=[]
    for j1 in [1..repeat]: # repeat the same sequence of mutations, it will give more cluster variables
        mutationSequence=mutationSequence+ll 
        
    for j in range(len(mutationSequence)): 
        vertices1 = computeEquationsForModulesTableaux(vertices1, mat1, mutationSequence[j],typ,rank)
        mat1 = matrixMutation(mat1, mutationSequence[j]) 

        if vertices1[1].ncols()>max_column:
            vertices1 = computeEquationsForModulesTableaux(vertices1, mat1, mutationSequence[j],typ,rank) # if encounter a cluster variable with too large number of columns, we mutate again to remove it
            mat1 = matrixMutation(mat1, mutationSequence[j]) 
        else:
            b1.append(vertices1[1]) 
            
    b1=removeDuplicates2(b1)
    b1=ExtendSetOfTableauxToContainPromotionsInMatrix(b1,n)
    b1=TableauExpansionsInMatrixList(b1, n)
    
    return b1

####################################################################

if __name__ == '__main__':
    #Define generation hyperparams
    rank, n = 4, 12   #...for Gr(rank, n)
    max_column = 4    #...obtain only tableaux with number of columns less or equal to max_column
    max_step = 36     #...this number controls the length of random mutation sequence, in order to obtain all cluster variables with number of columns less or equal to a fixed number, we need to put the number max_step sufficiently large
    checkpoint = 600  #...if after check_point steps, the number elements in b2 is not increasing, then stop
    repeat=23
    fp1='SmallRank'+str(max_column)+'ModulesGr'+str(rank)+str(n)+'_'+str(sys.argv[1])+'.txt' 
    
    #Run generation
    b2=[]
    typ=1
    k=rank
    sizeColumn=n-k
    ll0=[] 
    for i in range(1,k):
        for j in range(1,n-k):
            ll0.append((i-1)*sizeColumn+j)
    num,sn,sn1=0,0,0
    
    while True:  #...have split the ComputeClusterVariablesInGrkn function up into part we wish to parallelise 'll_perms' and the remainder
        sn=sn+1
                
        #Generate a list of permutations, then run above generation function with them on different cores
        lls = [np.random.permutation(ll0) for iii in range(max_step)]
        
        b5=[]
        with Pool() as p: #...map below action to as many cores as available
            bb = p.starmap(ll_perms,[(lls[i], typ, rank, max_column, n, repeat) for i in range(len(lls))])

        b5=[] #...concatenate list of all b1s for all permutations on different cores and add to b2
        for i in bb:
            b5=b5+i
            b5=removeDuplicates2(b5)
            
        b6=list(SetDifference2(b5,b2))
        print(len(b5), len(b6), len(b2))
                
        if b6 != []:
            b2=b2+b6
            F1 = open(fp1,'a+') 
            for j in b6:
                j1=MatrixTakeRows(j)
                F1.write(str(j1))
                F1.write('\n')
            F1.close()
        
        #Break loop when all probably generated
        if sn%checkpoint==1:
            print(sn, num, len(b2))
            if len(b2)==num:
                break
            else:
                num=len(b2)
       
    
