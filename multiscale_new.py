## Importing the libraries
import numpy as np
from matplotlib.pyplot import *
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy import random, linalg
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

#################################################### BASIC FUNCTIONS ####################################################


### Single scale model

def epsilon_0(Data,smax,delta):
    ## Input:
    ## Data: X,y records to be modeled
    ## smax: Max scale we may wish to explore
    ## delta: upper bound for vartheta_s
    
    ## Output:
    ## e0: epsilon_0
    
    d = Data.shape[1]-1
    Dist = squareform(pdist(Data[:,0:d], 'euclidean'))
    max_d = np.amax(Dist)
    T = 2*(max_d/2)**2
    ####
    epsilon = T/(2**0)
    G = np.exp(-((Dist**2)/epsilon))
    norms = LA.norm(G, axis=0)
    arg0 = np.argmin(norms)
    vartheta0 = min(norms)
    #### 
    epsilon = T/(2**smax)
    G = np.exp(-((Dist**2)/epsilon))
    norms = LA.norm(G, axis=0)
    args = np.argmin(norms)
    varthetas = min(norms)
    e0 = delta*(varthetas/vartheta0)
    return e0


def single_scale_forward(eps,Data,Dist,T,s,t): 
    ## Input: 
    ## 1: eps: epsilon tolerance for the inner product
    ## 2: Data: X,y
    ## 3: Dist: the distance matrix
    ## 4: T: a constant of the order of square of radius of data
    ## 5: s: Current scale
    ## 6: t: Current target
    
    ### Output:
    ## 1: B: the basis functions for the current scale
    ## 2: C: the coordinate of projection on this basis
    ## 3: sparse: The sparse representation collected over time
    ## 4: delta: a record for the reduction in MSE
    
    
    ### Adding the first function
    r = t.copy()
    epsilon = T/(2**s)
    G = np.exp(-((Dist**2)/epsilon))
    norms = LA.norm(G, axis=0)
    corr = (G.T.dot(r))**2/(norms**2)
    index = np.argmax(corr)
    zj = np.abs(G[index,:].dot(r))/(norms[index]**2)
    if zj >= eps:
        B = G[:,index].reshape(-1,1)
        sparse = np.concatenate((Data[index,:].reshape(1,-1),s*np.ones([1,1])), axis = 1)
        In = np.linalg.inv(B.T.dot(B))
        C = np.linalg.lstsq(B,t,rcond=None)[0]
        r = t - B.dot(C)
        delta = [np.linalg.norm(r)**2/B.shape[0]]
    else:
        B = []
        C = []
        sparse = []
        delta = []
        return B,C,sparse,delta
        
    ### adding subsequent functions    
    corr = (G.T.dot(r))**2/(norms**2)
    index = np.argmax(corr)
    zj = np.abs(G[index,:].dot(r))/(norms[index]**2)
    while zj >= eps:
        ### adding the good function
        b = G[:,index].reshape(-1,1)
        B,C,In = coordinate_update(In,B,b,t)
        sparse_temp = np.concatenate((Data[index,:].reshape(1,-1),s*np.ones([1,1])), axis = 1)
        sparse = np.concatenate((sparse,sparse_temp),axis=0)
        ### preparing for next iteration
        r = t - B.dot(C)
        delta.append(np.linalg.norm(r)**2/B.shape[0])
        corr = (G.T.dot(r))**2/(norms**2)
        index = np.argmax(corr)
        zj = np.abs(G[index,:].dot(r))/(norms[index]**2)
    return B,C,sparse,delta





def single_scale_backward(eps,vartheta,B,C,sparse,delta,t):
    ## Input:
    ## 1: eps: The same hyperparameter as single scale forward
    ## 2: B: The basis set collected in single scale forward
    ## 3: C: The coordinate of projection for single scale forward
    ## 4: sparse: The current sparse set at scale s
    ## 5: delta: the record for reduction in MSE from single scale forward
    ## 6: The target for the current scale
    
    ### Output
    ## 1: B: Cleaned basis set
    ## 2: C: Updated coordinate of projection
    ## 3: sparse: updated sparse representation
    mse_start = delta[-1]
    
    while True:
        norms = LA.norm(B, axis=0)
        w = norms*C
        index = np.argmin(np.abs(w))
        B1,C1,sparse1 = coordinate_delete(t,B,sparse,index)
        if len(C1) == 0:
            r = t
        else:
            r = t - B1.dot(C1)
        delta.append(np.linalg.norm(r)**2/B.shape[0])
        if abs(delta[-1]-mse_start) <= (eps*vartheta)**2/(B.shape[0]):
            #print('deleted')
            B = B1
            C = C1
            sparse = sparse1
        else:
            break
            
    return B,C,sparse
            
    
#### Updating the coordinate of projection


def coordinate_delete(t,B,sparse,index):
    ## Input
    ## 1: t: the target for the current scale
    ## 2: B: the current basis set
    ## 3: sparse: the current sparse set
    ## 4: index: the index to be updated
    
    ## Output:
    ## 1: B1: the updated basis
    ## 2: C1: updated coordinate of projection
    ## 3: sparse: updated sparse representation
    
    B1 = np.delete(B,index,1)
    sparse = np.delete(sparse,index,0) 
    
    if B1.shape[1]>0:
        C1 = np.linalg.lstsq(B1,t,rcond=None)[0]
    else:
        B1 = []
        C1 = []
    return B1,C1,sparse

def coordinate_update(In,B_s,b_s,r):
    ## Input
    ## 1: In: Current inverse
    ## 2: B_s: Current basis
    ## 3: b_s: function to be added
    ## 4: r: Current target
    
    ## Output:
    ## 1: B_s: Updated basis set
    ## 2: C_s: Updated coordinate set
    ## 3: Inv: Updated inverse
    
    b = B_s.T.dot(b_s)
    c = b_s.T.dot(b_s)
    
    Inv = np.zeros([In.shape[0]+1,In.shape[0]+1])
    f = In.dot(b)
    p = 1/(c-b.T.dot(f))
    Inv[0:In.shape[0],0:In.shape[0]] = In + p*f.dot(f.T)
    Inv[:-1,-1] = -1*p*f.flatten()
    Inv[-1,:-1] = -1*p*f.flatten().T
    Inv[-1,-1] = p
    
    y1 = B_s.T.dot(r)
    y2 = b_s.T.dot(r)
    y = np.concatenate((y1,y2),axis=0)
    
    C_s = Inv.dot(y)
    B_s = np.concatenate((B_s,b_s),axis = 1)
    return B_s,C_s,Inv
    



#### Main multiscale algorithm through K-fold CV

def Multiscale_train(Data,eps,maxs):
    ## Input:
    ## 1 Data: n-dimensional dataset under consideration
    ## 2 eps: The tolerance for the inner product
    
    ## Output:
    ## 1: sparse: the final sparse representation
    ## 2: Bs: the final basis set
    ## 3: Cs: the coordinate of projection
    ## 4: f: The final produced approximation
    ## 5: T: the normalization used in the exponential kernel
    
    ## Initialization
    s = 0  ## current scale
    sparse_s = []  ## current sparse representation
    f = 0  ## current approximation
    
    ## details of the dataset
    d = Data.shape[1]-1 ## getting the dimensionality of the dataset
    n = Data.shape[0] ## number of samples
    X = Data[:,0:-1]
    y = Data[:,-1].copy()
    r = Data[:,-1].copy()
    
    ## Getting the diameter of the dataset
    Dist = squareform(pdist(Data[:,0:d], 'euclidean'))
    max_d = np.amax(Dist)
    T = 2*(max_d/2)**2;
    
    ## computing delta and gamma
    epsilon = T/(2**s)
    G = np.exp(-((Dist**2)/epsilon))
    norms = LA.norm(G, axis=0)
    vartheta = min(norms)
    Delta = (eps*vartheta)**2/(Dist.shape[0])
    gamma = (eps*(vartheta**2))/np.linalg.norm(r)
    
    while s<=maxs:
        ## Single_scale approximation
        B_s,C_s,sparse_s,delta1 = single_scale_forward(eps,Data,Dist,T,s,r)
        if len(sparse_s)>0:
            B_s,C_s,sparse_s = single_scale_backward(eps,vartheta,B_s,C_s,sparse_s,delta1,r)
            
            
        if len(C_s) > 0:
            f_s = B_s.dot(C_s)

            if s == 0:
                Cs = C_s
                Bs = B_s
                sparse = sparse_s
            else:
                Cs = np.concatenate((Cs,C_s),axis=0)
                Bs = np.concatenate((Bs,B_s),axis = 1)
                sparse = np.concatenate((sparse,sparse_s),axis = 0)

            r = r - f_s
            f = f + f_s
            

        s = s + 1
        
        
        epsilon = T/(2**s)
        G = np.exp(-((Dist**2)/epsilon))
        norms = LA.norm(G, axis=0)
        vartheta = min(norms)
        eps1 = (gamma*(np.linalg.norm(r)))/(vartheta**2)
        eps2 = np.sqrt(Delta*Dist.shape[0])/vartheta
        eps = max(eps1,eps2)
    return sparse,Bs,Cs,f,T


def Multiscale_test(test_d,sparse,Cs,T,maxs):
    ## Input:
    ## test_d: testing dataset n x (d+1), also contains y
    ## sparse: sparse representation
    ## Cs: the coordinates for sparse representation
    ## T: normalizing constant
    ## maxs: maximum scale till which we want to test the approximation
    
    ## Output:
    ## error: MSE for prediction till the given scale
    
    xpred = test_d[:,:-1]
    d = xpred.shape[1]
    Dist = distance.cdist(xpred,sparse[:,0:d],'euclidean')
    Pred = np.zeros(Dist.shape)
    for i in range(Dist.shape[1]):
        epsilon = T/(2**sparse[i,-1])
        Pred[:,i] = np.exp(-((Dist[:,i]**2)/epsilon))
    
    scales = sorted(set(sparse[:,-1]))
    Af = {}
    err = []
    for s in np.arange(maxs+1):
        ind = np.where(sparse[:,-1] <= s)[0]
        Af[s] = Pred[:,ind].dot(Cs[ind])
        err.append(mean_squared_error(test_d[:,-1].reshape(-1,1),Af[s]))
        
    error = np.array(err)
    return error



def KfoldCV(D,k,eps,maxs):
    ## Input
    ## D: Full dataset (X,y)
    ## k: number of data splits for cv
    ## eps: epsilon_0
    ## maxs: Highest scale to explore to find the best scale
    
    ## Output
    ## sparse: sparse representation at the optimal scale
    ## Bs: Basis at the optimal scale
    ## Cs: Coordinate at the optimal scale
    ## f: Approximation on the full data
    ## T: Normalizing constant
    ## meanerror: record of testing errors for prediction at different scales
    

    ## shuffling data
    arr = np.arange(D.shape[0])
    random.seed(0)
    np.random.shuffle(arr)
    Data = D[arr,:]
    

    kf = KFold(n_splits=k)
    X = Data[:,:-1]
    d = X.shape[1]
    y = Data[:,-1].reshape(-1,1)
    
    Dist = squareform(pdist(Data[:,0:d], 'euclidean'))
    max_d = np.amax(Dist)
    T = 2*(max_d/2)**2;
    
    
    count =0
    error_mat = np.zeros([maxs+1,k])
    for train, test in kf.split(X):

        ## training data
        train_X = X[train]
        train_y = y[train].reshape(-1,1)
        #print(train_X.shape,train_y.shape)
        train_d = np.concatenate((train_X,train_y),axis = 1)
        sparse,Bs,Cs,f,T = Multiscale_train(train_d,eps,maxs)

        ## testing data
        test_X = X[test]
        test_y = y[test].reshape(-1,1)
        test_d = np.concatenate((test_X,test_y),axis = 1)
        error = Multiscale_test(test_d,sparse,Cs,T,maxs)
        
        ## saving
        #print(error.shape)
        error_mat[:,count] = error.flatten()
        count = count + 1
        print(count)

    meanerror = np.mean(error_mat,axis = 1)
    ss = np.argmin(meanerror)
    
    ### Predicting using best termination scale
    sparse,Bs,Cs,f,T = Multiscale_train(D,eps,ss)
    
    return sparse,Bs,Cs,f,T,meanerror




### Prediction

def predict(xpred,sparse,T,Cs):
    ## Input:
    ## 1: xpred: the locations for predictions
    ## 2: sparse: the sparse repreentation
    ## 3: T: the normalization used in the exponential kernel (T can be computed again, so not necessary here)
    ## 4: Cs: coordinate of projection
    
    ## Output:
    ## 1: Af: the approximation produced
    d = xpred.shape[1]
    Dist = distance.cdist(xpred,sparse[:,0:d],'euclidean')
    Pred = np.zeros(Dist.shape)
    for i in range(Dist.shape[1]):
        epsilon = T/(2**sparse[i,-1])
        Pred[:,i] = np.exp(-((Dist[:,i]**2)/epsilon))
    
    Af = Pred.dot(Cs)
    return Af


#################################################### BASIC FUNCTIONS ####################################################
