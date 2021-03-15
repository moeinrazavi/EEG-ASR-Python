#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import scipy.signal as sg
import scipy.special as sc
import scipy.linalg as sl
import scipy.sparse.linalg as ssl
from psutil import virtual_memory

import autograd.numpy as autonp

from pymanopt.manifolds import Grassmann, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import TrustRegions, SteepestDescent, ConjugateGradient, NelderMead, nelder_mead



# In[6]:


def geometric_median(X_gm,tol=1.e-5,y = [],max_iter=500):
    # Calculate the geometric median for a set of observations (mean under a Laplacian noise distribution)
    # This is using Weiszfeld's algorithm.
    #
    # In:
    #   X : the data, as in mean
    #   tol : tolerance (default: 1.e-5)
    #   y : initial value (default: median(X))
    #   max_iter : max number of iterations (default: 500)
    #
    # Out:
    #   g : geometric median over X

    if not y:
        y = np.median(X_gm, axis=0)

    for _ in range(max_iter):
        invnorms = 1/np.sqrt(np.sum((X_gm-y)**2,axis=1)).reshape(-1,1)
        oldy = y
        y = np.sum(X_gm*invnorms,0)/np.sum(invnorms,0)

        if (np.linalg.norm(y-oldy)/np.linalg.norm(y) < tol):
            break
        
    return y


# In[7]:


def block_geometric_median(X_bgm,blocksize=1,tol=1.e-5,y = [],max_iter=500):
    # Calculate a blockwise geometric median (faster and less memory-intensive 
    # than the regular geom_median function).
    #
    # This statistic is not robust to artifacts that persist over a duration that
    # is significantly shorter than the blocksize.
    #
    # In:
    #   X : the data (#observations x #variables)
    #   blocksize : the number of successive samples over which a regular mean 
    #               should be taken
    #   tol : tolerance (default: 1.e-5)
    #   y : initial value (default: median(X))
    #   max_iter : max number of iterations (default: 500)
    #
    # Out:
    #   g : geometric median over X
    #
    # Notes:
    #   This function is noticably faster if the length of the data is divisible by the block size.
    #   Uses the GPU if available.
    # 
    
    if blocksize > 1:
        [o,v] = np.shape(X_bgm)                    #observations & #variables
        r = np.mod(o,blocksize)                #rest in last block
        b = int((o-r)/blocksize)               #blocks
        if r > 0:
            X1 = np.reshape(np.sum(np.reshape(X_bgm[:(o-r),:], (blocksize,b*v)),0), (b,v))
            X2 = np.sum(X_bgm[(o-r):,:]*(blocksize/r),0)
            if r == 1:
                X2 = np.reshape(X2,(1,-1))
            X_bgm = np.concatenate((X1,X2),axis=0)
            
        else:
            X_bgm = np.reshape(np.sum(np.reshape(X_bgm, (blocksize,b*v)),0),(b,v))

    y = geometric_median(X_bgm,tol,y,max_iter)/blocksize;
    
    return y


# In[8]:


def positive_definite_karcher_mean(x):
    
    """
    Compute the centroid as Karcher mean of points x belonging to the manifold
    man.
    """
    
    k = len(x)
    n = x.shape[-1]
    
    man = PositiveDefinite(n)
    

    def objective(y):  # weighted Frechet variance
        acc = 0
        for i in range(k):
            acc += man.dist(y, x[i]) ** 2
        return acc / 2

    def gradient(y):
        g = man.zerovec(y)
        for i in range(k):
            g -= man.log(y, x[i])
        return g

    # TODO: manopt runs a few TR iterations here. For us to do this, we either
    #       need to work out the Hessian of the Frechet variance by hand or
    #       implement approximations for the Hessian to use in the TR solver.
    #       This is because we cannot implement the Frechet variance with
    #       theano and compute the Hessian automatically due to dependency on
    #       the manifold-dependent distance function.
    solver = SteepestDescent(maxiter=15)
    problem = Problem(man, cost=objective, grad=gradient, verbosity=0)
    return solver.solve(problem)


# In[9]:


def asr_process_r(Data_ASR_Process,SamplingRate,State,WindowLength=0.1,LookAhead=[],StepSize=4,MaxDimensions=1,MaxMemory=[],UseGPU=False):
    
    
    if not LookAhead: 
        LookAhead = WindowLength/2
    
    if not MaxMemory:
        mem = virtual_memory()
        MaxMemory = mem.total/(2**21)   # total physical memory available
    
    if MaxDimensions < 1:
        MaxDimensions = np.round(Data_ASR_Process.shape[0]*MaxDimensions+1e-8)
        
    
    C,S = Data_ASR_Process.shape
    N = np.round(WindowLength*SamplingRate+1e-8).astype(int)
    P = np.round(LookAhead*SamplingRate+1e-8).astype(int)
    T = State["T"]; M = State["M"]; A = State["A"]; B = State["B"]
    if not len(State["carry"]) > 0:
        State["carry"] = np.tile(2*Data_ASR_Process[:,0].reshape(-1,1),(1,P)) - Data_ASR_Process[:,np.mod(np.arange((P+1),1,-1)-1,S)]
    
    Data_ASR_Process = np.concatenate((State["carry"],Data_ASR_Process),axis=1)
    Data_ASR_Process[np.where(~np.isfinite(Data_ASR_Process))] = 0
    
    # split up the total sample range into k chunks that will fit in memory
    splits = np.ceil((C*C*S*8*8 + C*C*8*S/StepSize + C*S*8*2 + S*8*5) / (MaxMemory*1024*1024 - C*C*P*8*3))
    if splits > 1:
        print('Now cleaning data in %i blocks' %splits)
        
    for i in np.arange(1,splits+1):
        range_split = np.arange(1+np.floor((i-1)*S/splits), np.min((S,np.floor(i*S/splits)))+1).astype(int)
        if len(range_split) > 0:
            # get spectrally shaped data X for statistics computation (range shifted by lookahead)
            # and also get a subrange of the data (according to splits)
            X = []
            for i in range(len(Data_ASR_Process)):
                [data_temp,State["iir"][i]] = sg.lfilter(B,A,Data_ASR_Process[i, range_split+P-1],axis=-1,zi = State["iir"][i])
                X.append(data_temp)
                
            X = np.array(X)
            # return the filtered but othrerwise unaltered data for debugging
            Y = X
            # move it to the GPU if applicable

            ## the Riemann version uses the sample covariance matrix here:
            SCM = (1/S) * (np.dot(X, X.T))     # channels x channels
            # if we have a previous covariance matrix, use it to compute the average to make
            # the current covariance matrix more stable
            if not State["cov"]:
                
                # we do not have a previous matrix to average, we use SCM as is
                Xcov = SCM
            else:
                A = np.zeros((2,C,C))
                A[0,:,:] = SCM
                A[1,:,:] = State["cov"]
                Xcov = positive_definite_karcher_mean(A)
                
            update_at = np.arange(StepSize,X.shape[1]+StepSize-1+1e-8,StepSize)
            update_at[np.where(update_at > X.shape[1])] = X.shape[1]
            # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
            if not len(State["last_R"]) > 0:
                update_at = np.concatenate(([1],update_at), axis=0)
                State["last_R"] = np.eye(C)

            # function from manopt toolbox, adapted to this use case. manopt needs to be in the path
            
            [V, D] = rasr_nonlinear_eigenspace(Xcov, C)
            # use eigenvalues in descending order
            order = np.argsort(np.diag(D).flatten())
            D = np.diag(D).flatten()[order]
            # to sort the eigenvectors, here the vectors computed on the manifold
            V = V[:,order]

            # determine which components to keep (variance below directional threshold or not admissible for rejection)
            keep = D < np.sum(np.dot(T,V)**2,axis=0)
            keep[np.where(np.arange(1,C+1) < (C-MaxDimensions))] = True
            trivial = all(keep)

            # update the reconstruction matrix R (reconstruct artifact components using the mixing matrix)
            if not trivial:
                R = np.real(np.dot(np.dot(M,np.linalg.pinv(keep.reshape(-1,1)*np.dot(V.T,M))),V.T))
            else:
                R = np.eye(C)

            # do the reconstruction in intervals of length stepsize (or shorter at the end of a chunk)
            last_n = 0
            for j in range(len(update_at)):
                # apply the reconstruction to intermediate samples (using raised-cosine blending)
                n = update_at[j]
                if (not trivial or not State["last_trivial"]):
                    subrange = range_split[np.arange(last_n,n).astype(int)]
                    blend = (1-np.cos(np.pi*np.arange(1,(n-last_n)+1)/(n-last_n)))/2
                    Data_ASR_Process[:,subrange-1] = blend*np.dot(R,Data_ASR_Process[:,subrange-1]) + (1-blend)*np.dot(State["last_R"],Data_ASR_Process[:,subrange-1])
                
                last_n = n
                State["last_R"] = R
                State["last_trivial"] = trivial
                
        if splits > 1:
            print(".")
    if splits > 1:
        print("\n")
        
    # carry the look-ahead portion of the data over to the state (for successive calls)
    State["carry"] = np.concatenate((State["carry"], Data_ASR_Process[:,np.arange((-1-P+1),0)]), axis = 1)
    State["carry"] = State["carry"][:,np.arange((-1-P+1),0)]
    State["cov"] = Xcov

    # finalize outputs
    outdata = Data_ASR_Process[:,:-P]
    outstate = State
        
    return outdata, outstate, Y


# In[10]:


def rasr_nonlinear_eigenspace(L, k, alpha=1):
# Example of nonlinear eigenvalue problem: total energy minimization.
#
# L is a discrete Laplacian operator: the covariance matrix
# alpha is a given constant for optimization problem
# k determines how many eigenvalues are returned 
#
# This example is motivated in the paper
# "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
# Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
# SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
#


# This file is part of Manopt and is copyrighted. See the license file.
#
# Main author: Bamdev Mishra, June 19, 2015.
# Contributors:
# Sarah Blum, 8/2018: changed the function to be included in Riemannian ASR:
#   additional outputs are needed: namely the eigenvectors and eigenvalues 
   
    n = L.shape[0]
    assert L.shape[1]==n,'L must be square.'
        
    # Grassmann manifold description
    # (1) Instantiate a manifold
    manifold = Grassmann(n, k)

    # Cost function evaluation
    # (2) Define the cost function (here using autograd.numpy)
    def cost(X):
        rhoX = np.sum(X**2, axis=1)
        val = 0.5*np.trace(np.dot(X.T,np.dot(L,X))) + (alpha/4)*(np.dot(rhoX,np.linalg.solve(L,rhoX)))

        return val

    # Euclidean gradient evaluation
    # Note: Manopt automatically converts it to the Riemannian counterpart.
    def egrad(X):
        rhoX = np.sum(X**2, axis=1)
        g = np.dot(L,X) + alpha*np.dot(np.diag(np.linalg.solve(L,rhoX)),X)

        return g

    # Euclidean Hessian evaluation
    # Note: Manopt automatically converts it to the Riemannian counterpart.

    def ehess(X, U):
        rhoX = np.sum(X**2, axis=1)
        rhoXdot = 2*np.sum(X*U, axis=1)
        h = np.dot(L,U) + alpha*np.dot(np.diag(np.linalg.solve(L,rhoXdot)),X) + alpha*np.dot(np.diag(np.linalg.solve(L,rhoXdot)),U)

        return h


    # Initialization as suggested in above referenced paper.
    # randomly generate starting point for svd
    X = np.random.randn(n, k)
    [U,S,V] = np.linalg.svd(X, full_matrices=False)
    X = np.dot(U,V.T)
    [S0, U0] = ssl.eigs(L + alpha*np.diag(np.linalg.solve(L,np.sum(X**2, axis=1))),k)
    S0 = np.diag(S0)[:k,:k]
    U0 = U0[:k,:k]
    X0 = U0;

    # Call manoptsolve to automatically call an appropriate solver.
    # Note: it calls the trust regions solver as we have all the required
    # ingredients, namely, gradient and Hessian, information.

    # (3) Instantiate a Pymanopt solver
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, ehess=ehess)

    solver = TrustRegions()

    # let Pymanopt do the rest
    Xopt = solver.solve(problem, X0)

    
    
    return Xopt, S0


# In[11]:


def histc(X_histc, bins):
    map_to_bins = np.digitize(X_histc,bins)
    r = np.zeros((len(bins), X_histc.shape[1])).astype(int)
    for i in range(r.shape[1]):
        temp_r = np.zeros(len(bins)).astype(int)
        for j in map_to_bins[:,i]:
            temp_r[j-1] += 1
        r[:,i] = temp_r
    return [r, map_to_bins]


# In[12]:


def asr_calibrate_r(Data_ASR_Calib,SamplingRate,RejectionCutoff=3,Blocksize=10,FilterB=[],FilterA=[],
                    WindowLength=0.1,WindowOverlap=0.5,MaxDropoutFraction=0.1,MinCleanFraction=0.3):
    
    C, S = Data_ASR_Calib.shape
    C = int(C)
    S = int(S)
    if (not FilterA or not FilterB):
    # yulewalk filter coefficients
        if SamplingRate == 100:
            FilterB=np.array([0.931423352864164,-1.002368381496344,-0.412535986201832,0.763156747632749,0.416043039291041,
                        -0.654913103869222,-0.037258351804688,0.191626845875261,0.046241197159235])
            FilterA=np.array([1,-0.454422018030376,-1.000703868293683,0.537492552133784,0.490501336099142,
                        -0.486106287935112,-0.199598649069948,0.183004842073003,0.045767854923467])
        elif SamplingRate == 125:
            FilterB=np.array([1.087423167955396, -1.836435553816374, 0.573976014496824, 0.361020603610170, 0.059271456186474,
                        0.076763175985072, -0.498304757808424, 0.276872948140515, -0.006930792028036])
            FilterA=np.array([1.000000000000000, -0.983952187817050, -0.520232502560362, 0.603540557711479, 0.116893105621457,
                        -0.029126160924775, -0.282359853603720, 0.040784793357921, 0.103437108246108])
        elif SamplingRate == 128:
            FilterB=np.array([1.102730163916508,-2.002562181361187,0.894211951648128,0.154997952422709,0.019236690448804,
                        0.178289777027871,-0.528030669649869,0.291354060340750,-0.026220980252637])
            FilterA=np.array([1,-1.104204204642322,-0.331955852860656,0.580294622110738,-0.001036001391563,
                        0.038216709192505,-0.260992803442537,0.029871905776108,0.093504469295919])
        elif SamplingRate == 200:
            FilterB=np.array([1.448948332580248,-2.669251476480278,2.081397062073106,-0.973667887704943,0.105460506035284,
                        -0.188910169231467,0.611133163659234,-0.361648301307510,0.183431306077673])
            FilterA=np.array([1,-0.991323609939397,0.315956314546932,-0.070834748167754,-0.055879382207115,
                        -0.253961902647894,0.247305661525119,-0.042047843747311,0.007745571833446])
        elif SamplingRate == 250:
            FilterB=np.array([1.731333108542578,-4.168133532956979,5.373799008441696,-5.572125643438828,4.701226513165113,
                        -3.342087996552435,1.950454887249073,-0.766909658912065,0.233281060974834])
            FilterA=np.array([1,-1.638494927666595,1.739878142990544,-1.836386578834554,1.392417753679789,
                    -0.953780426622192,0.505158779550744,-0.159504514603054,0.054527839984798])
        elif SamplingRate == 256:
            FilterB=np.array([1.758701314177013,-4.326762439445861,5.799988003101626,-6.239662546354797,5.376807904688266,
                        -3.793821889337512,2.164910809522656,-0.859139256986372,0.256936112562797])
            FilterA=np.array([1,-1.700803963930180,1.923283039105883,-2.082692972692993,1.598263874255742,
                        -1.073585418393008,0.567971922565269,-0.188618149976820,0.057295411599726])
        elif SamplingRate == 300:
            FilterB=np.array([1.915392067643352,-5.774842110492657,9.186476485910097,-10.735035661935790,9.642367243772222,
                        -6.618193969953830,3.421942149417380,-1.262297656999261,0.296842301936336])
            FilterA=np.array([1,-2.314370332205515,3.222256732737830,-3.603052770431895,2.964515484407207,
                        -1.884261584068347,0.922245586875739,-0.310325170364825,0.063458644989631])
        elif SamplingRate == 500:
            FilterB=np.array([2.313352008695781,-11.947122300909630,29.106716649329467,-43.755017100720280,44.338576745229710,
                        -30.996552384654628,14.620988302087529,-4.274341240037765,0.598255358379289])
            FilterA=np.array([1,-4.689332908445407,10.598998670108621,-14.969151810137603,14.332035839974353,
                        -9.492431706917795,4.242589961898615,-1.171560097517919,0.153804842771760])
        elif SamplingRate == 512:
            FilterB=np.array([2.327547563613706,-12.216647848598150,30.163278905828193,-45.800984202084980,46.726126301108200,
                        -32.779685819676630,15.462334961255571,-4.501977968530456,0.624273348167554])
            FilterA=np.array([1,-4.782737894425858,10.978069623662467,-15.679518788820260,15.128197866758910,
                        -10.063207983453083,4.501469063651326,-1.239410087328936,0.161472751068846])
        
    row_ind_inf, col_ind_inf = np.where(np.isinf(Data_ASR_Calib))
    Data_ASR_Calib[row_ind_inf, col_ind_inf] = 0
    # apply the signal shaping filter and initialize the IIR filter state
    zinit = sg.lfilter_zi(FilterB,FilterA)
    iirstate = []
    
    for i in range(len(Data_ASR_Calib)):
        [Data_ASR_Calib[i],iirstate_i] = sg.lfilter(FilterB,FilterA,Data_ASR_Calib[i],axis=-1,zi = zinit)
        iirstate.append(iirstate_i)
    
    Data_ASR_Calib = Data_ASR_Calib.T
    if np.isinf(Data_ASR_Calib).any():
        print('The IIR filter diverged on your data. Please try using either a more conservative filter or removing some bad sections/channels from the calibration data.')
        
    # calculate the sample covariance matrices U (averaged in blocks of blocksize successive samples)
    U = np.zeros((len(np.arange(1,S+1e-8,Blocksize)),C*C))

    for k in np.arange(1,Blocksize+1):
        block_range = np.arange(k,(S+k-1)+1e-8,Blocksize).astype(int)
        block_range[block_range>S] = S
        U = U + np.reshape(np.reshape(Data_ASR_Calib[block_range-1,:],(-1,1,C))*np.reshape(Data_ASR_Calib[block_range-1,:],(-1,C,1)),np.shape(U))
    
    # get the mixing matrix M
    M = sl.sqrtm(np.real(np.reshape(block_geometric_median(U/Blocksize), (C,C))))

    # window length for calculating thresholds
    N = np.round(WindowLength*SamplingRate+1e-8)

    # get the threshold matrix T
    print('Determining per-component thresholds...')
    
    [V,D] = rasr_nonlinear_eigenspace(M, C)
    D = np.diagonal(D)
    
    order = np.argsort(D)
    V = V[:,order]
    Data_ASR_Calib = np.abs(np.dot(Data_ASR_Calib,V))
    mu = np.zeros(C)
    sig = np.zeros(C)
    for c in np.arange(C-1,-1,-1):
        # compute RMS amplitude for each window...
        rms = Data_ASR_Calib[:,c]**2
        rms = np.sqrt(np.sum(rms[(np.round(np.arange(1,S-N+1e-8,N*(1-WindowOverlap))+1e-8)+np.arange(N).reshape(-1,1) - 1).astype(int)],axis=0)/N)
        # fit a distribution to the clean part
        
        [mu[c],sig[c],_,_] = fit_eeg_distribution(rms,MinCleanFraction,MaxDropoutFraction)

    T = np.diag(mu + RejectionCutoff*sig)*V.T
    print('done.')
    
    State={"M":M,"T":T,"B":FilterB,"A":FilterA,"cov":[],"carry":[],"iir":iirstate,"last_R":[],"last_trivial":True}

    
    return State


# In[13]:


def fit_eeg_distribution(X_fit_EEG, MinCleanFraction = 0.25, MaxDropoutFraction = 0.1, FitQuantiles = [0.022, 0.6],
                         StepSizes = [0.01, 0.01], ShapeRange = np.arange(1.7, 3.5+1e-8, 0.15)):
    
    X_fit_EEG = np.array(sorted(X_fit_EEG))
    n = len(X_fit_EEG);
    zbounds = []
    rescale = []
    for b in range(len(ShapeRange)):
        zbounds.append(np.sign(np.array(FitQuantiles)-1/2)*sc.gammaincinv(1/ShapeRange[b], np.sign(np.array(FitQuantiles)-1/2)*(2*np.array(FitQuantiles)-1))**(1/ShapeRange[b]))
        rescale.append(ShapeRange[b]/(2*sc.gamma(1/ShapeRange[b])))
    
    # determine the quantile-dependent limits for the grid search
    lower_min = np.min(FitQuantiles)          # we can generally skip the tail below the lower quantile
    max_width = np.diff(FitQuantiles)[0]      # maximum width is the fit interval if all data is clean
    min_width = MinCleanFraction*max_width    # minimum width of the fit interval, as fraction of data
    # get matrix of shifted data ranges
    X_fit_EEG = X_fit_EEG[list(np.arange(1,np.round(n*max_width+1e-8)+1).astype(int))].reshape(-1,1) + np.round(n*np.arange(lower_min,lower_min+MaxDropoutFraction+StepSizes[0],StepSizes[0])+1e-8)
    X1 = X_fit_EEG[0,:]
    X_fit_EEG = X_fit_EEG-X1
    opt_val = np.inf
    # for each interval width...
    for m in np.round(n*np.arange(max_width,min_width-1e-8,-StepSizes[1])+1e-8).astype(int):
        # scale and bin the data in the intervals
        nbins = np.round(3*np.log2(1+m/2)+1e-8).astype(int)
        H = X_fit_EEG[:m,:]*nbins/X_fit_EEG[m-1,:]
        logq = np.log(histc(H,[*np.arange(nbins)] + [np.inf])[0] + 0.01)

        # for each shape value...
        for b in range(len(ShapeRange)):
            bounds = zbounds[b]
            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0]+np.arange(0.5,nbins-0.5+1e-8)/nbins*np.diff(bounds)
            p = np.exp(-abs(x)**ShapeRange[b])*rescale[b]
            p = p.reshape(-1,1)/np.sum(p)

            # calc KL divergences
            kl = np.sum(p*(np.log(p)-logq[:-1,:]), axis=0) + np.log(m)

            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = ShapeRange[b]
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx]+X_fit_EEG[m-1,idx]]

    # recover distribution parameters at optimum
    alpha = ((opt_lu[1]-opt_lu[0])/np.diff(opt_bounds))[0]
    mu = (opt_lu[0]-opt_bounds[0]*alpha)
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sigma = np.sqrt((alpha**2)*sc.gamma(3/beta)/sc.gamma(1/beta))
    print("mu: ", mu, "sig: ", sigma)
    
    return mu, sigma, alpha, beta


# In[14]:


def clean_windows(Signal_Clean_Windows, SRate, MaxBadChannels = 0.2, PowerTolerances = [-3.5, 5], WindowLength = 1,
                  WindowOverlap = 2/3, MaxDropoutFraction = 0.1,
                  MinCleanFraction = 0.25, TruncateQuantile = [0.022, 0.6],
                  StepSizes = [0.01, 0.01], ShapeRange = np.arange(1.7,3.5+1e-8, 0.15)):
    
    if (MaxBadChannels > 0 and MaxBadChannels < 1):
        MaxBadChannels = np.round(len(Signal_Clean_Windows)*MaxBadChannels+1e-8).astype(int)
    
    C, S = np.shape(Signal_Clean_Windows)
    
    N = WindowLength*SRate
    wnd = np.arange(N).astype(int)
    offsets = np.round(np.arange(1, S-N+1e-8, N*(1-WindowOverlap))+1e-8).astype(int)
    ind = offsets + wnd.reshape(-1,1) - 1
    print("Determining time window rejection thresholds...")
    
    wz = np.zeros((C, ind.shape[1]))
    for c in np.arange(C-1, -1, -1):
        # compute RMS amplitude for each window...
        X = Signal_Clean_Windows[c,:]**2
        X = np.sqrt(np.sum(X[ind], axis = 0)/N)
        # robustly fit a distribution to the clean EEG part
        [mu,sig,_,_]= fit_eeg_distribution(X,MinCleanFraction, MaxDropoutFraction,
                                           TruncateQuantile, StepSizes, ShapeRange)
        # calculate z scores relative to that
        wz[c,:] = (X - mu)/sig
        
    print('Done.')
    
    swz = np.sort(wz, axis=0)
    # determine which windows to remove
    remove_mask = np.full((np.shape(swz)[1]), False, dtype=bool)
    if np.max(PowerTolerances) > 0:
        remove_mask[np.where(swz[-MaxBadChannels-1,:] > np.max(PowerTolerances))[0]] = True
    if np.min(PowerTolerances) < 0:
        remove_mask[np.where(swz[MaxBadChannels,:] < np.min(PowerTolerances))[0]] = False
        
    removed_windows = np.where(remove_mask)[0]
    
    # find indices of samples to remove
    removed_samples = np.tile(offsets[removed_windows].reshape(-1,1),len(wnd)) + np.tile(wnd.reshape(-1,1),len(removed_windows)).T
    
    # mask them out
    sample_mask = np.full(S, True, dtype=bool)
    sample_mask[removed_samples.flatten()] = False;
    print('Keeping %.1f%% (%.0f seconds) of the data.\n' %(100*(np.mean(sample_mask)),np.count_nonzero(sample_mask)/SRate))
    # determine intervals to retain
    retain_data_intervals = np.where(np.diff([False]+ list(sample_mask)+ [False]))[0].reshape(-1,2)
    
    
    retain_data_intervals[:,1] = retain_data_intervals[:,1]-1

    return Signal_Clean_Windows, sample_mask, retain_data_intervals
    


# In[15]:


def clean_asr(Signal_Clean_ASR, SRate, Cutoff = 5, WindowLength = [],
              StepSize = [], MaxDimensions = 2/3,
              ReferenceMaxBadChannels = 0.075, ReferenceTolerances = [-3.5, 5.5],
              ReferenceWindowLength = 1, CleanWindows = False, UseGPU = False, UseRiemannian = True, MaxMem = []):
    if not WindowLength: WindowLength = np.max((0.5,1.5*len(Signal_Clean_ASR)/SRate))
    
    # first determine the reference (calibration) data
    print("Finding a clean section of the data...")
    if CleanWindows:
        Signal_Clean_ASR, sample_mask, retain_data_inervals = clean_windows(Signal_Clean_ASR,SRate,ReferenceMaxBadChannels,ReferenceTolerances,ReferenceWindowLength)
        ref_section = Signal_Clean_ASR[:, np.where(sample_mask)[0]]
    else:
        ref_section = Signal_Clean_ASR
    # calibrate on the reference data
    print("Estimating calibration statistics; this may take a while...")
    if UseRiemannian:
        state = asr_calibrate_r(ref_section,SRate,Cutoff)
#     else:
#         state = asr_calibrate(ref_section,SRate,Cutoff)
        
    if not StepSize:
        StepSize = np.floor(SRate*WindowLength/2)
        
    # extrapolate last few samples of the signal
    temp = 2*Signal_Clean_ASR[:,-1].reshape(-1,1) - Signal_Clean_ASR[:,np.arange((-1-1),-1-np.round(WindowLength/2*SRate+1e-8)-1,-1).astype(int)]
    sig = np.concatenate((Signal_Clean_ASR,temp),axis=1)
    # process signal using ASR
    if UseRiemannian:
        [Signal_Clean_ASR,state,_] = asr_process_r(sig,SRate,state,WindowLength,WindowLength/2,StepSize,MaxDimensions,MaxMem,UseGPU)
#     else:
#         [Signal,state] = asr_process(sig,signal.srate,state,windowlen,windowlen/2,stepsize,maxdims,maxmem,usegpu)
    Signal_Clean_ASR = np.delete(Signal_Clean_ASR,np.arange(state["carry"].shape[1]).astype(int), axis=1)
    return Signal_Clean_ASR


# In[ ]:




