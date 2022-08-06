# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 14:22:39 2022

@author: talk2
"""

import numpy as np
import math as mth
import random
from collections import deque


def GeneticAlgorithm(fun, funInEq, funEq, IntIndx, lb, ub):
    n = len(lb); popsize = 50; penalty = 100; ubmlb = ub - lb; r = [0];
    Xreal = np.zeros((n, popsize*2)); Xbin = []; NotIntIndx = []; 
    
    for i in range(n):
        isin = i in IntIndx;
        if(not isin):
            NotIntIndx.append(i);
    
    for j in range(popsize*2):
        Xbin.append('rand');
    
    for i in range(n):
        l = mth.ceil(mth.log2(ubmlb[i]));
        if (l == 0):
            l = 1;
        if(i in IntIndx):
            r.append(r[i] + l);
        else:
            r.append(r[i] + l + 10);
    
    def Objfun(j):
        Of = fun(Xreal[:, j]); 
        ineq = np.max(0, funInEq(Xreal[:, j])).sum();
        eq = np.abs(funEq(Xreal[:, j])).sum();
        return Of + penalty*(ineq + eq);
    
    def rand(j):
        for i in IntIndx:
            Xreal[i, j] = random.randint(lb[i], ub[i])
        for i in NotIntIndx:
            Xreal[i, j] = lb[i] + random.random()*ubmlb[i];
        
    def real2bin(j):
        Xbin[j] = '';
        for i in NotIntIndx:
            nb = r[i + 1] - r[i]; den = 2**nb;
            xres = mth.floor((Xreal[i, j] - lb[i])*den/ubmlb[i]);
            binary = bin(xres)[2:].zfill(nb);
            Xbin[j] += binary;  
        for i in IntIndx:
            nb = r[i + 1] - r[i];
            xres = Xreal[i, j]  - lb[i];
            binary = bin(xres)[2:].zfill(nb);
            Xbin[j] += binary;  
            
    def bin2real(j):
        for i in NotIntIndx:
            str = Xbin[j][r[i]:r[i + 1]];
            nb = r[i + 1] - r[i]; den = 2**nb - 1;
            xres = int(str, 2);
            Xreal[i, j] = lb[i] + xres*ubmlb[i]/den; 
        for i in IntIndx:
            str = Xbin[j][r[i]:r[i + 1]];
            nb = r[i + 1] - r[i]; den = 2**nb;
            xres = int(str, 2);
            Xreal[i, j] = int(mth.ceil(lb[i] + xres*ubmlb[i]/den)); 
            
    def Reproduce(i, j, k, Order):
        crp = random.randint(1, r[-1]-1);
        Xbin[Order[popsize + k*2]] = Xbin[Order[i]][:crp] + Xbin[Order[j]][crp:];
        Xbin[Order[popsize + k*2 + 1]] = Xbin[Order[j]][:crp] + Xbin[Order[i]][crp:];
    
    def Mutate(i):
        k = random.randint(1, r[-1]-1); kp1 = k+1;
        Xbin[i] = Xbin[i][:k] + str((int(Xbin[i][k]) + 1) % 2) + Xbin[i][kp1:];
        
    ans = GeneticAlgorithmWorkHorse(Objfun, real2bin, bin2real, Reproduce, Mutate, rand, n, Xreal, Xbin, popsize, lb, ub, ubmlb);
    return ans;


def GeneticAlgorithmWorkHorse(Objfun, real2bin, bin2real, Reproduce, Mutate, rand, nvar, Xreal, Xbin, popsize, lb, ub, ubmlb):
    imax = int(round(popsize/2)); popsize2 = popsize*2; w = -1; AvegVals = deque(); k1 = 0.5; k2 = 0.5; k3 = 0.5; k4 = 0.5;
    pi = mth.pi; Fval = np.zeros(popsize*2); Fits = np.zeros(popsize2); Bfval = mth.inf; sol = np.zeros(nvar); Wfval = -mth.inf;
    
    # Fitness function
    def FitFun(f):
        fitfun = 0.5*(mth.atan(-f)/pi + 1);
        return fitfun;
    
    # Function pointer for sorting
    def takeSecond(elem):
        return elem[1];
    for i in range(popsize):
        # Generate Random Solution and Convert2Binary
        rand(i); real2bin(i);
        # Evaluate Solution
        Fval[i] =  Objfun(i);
        # Compare Solution to Best Solution Ever Found
        if (Fval[i] < Bfval):
            Bfval = Fval[i];
            sol = Xreal[:,i];
        # Compare Solution to Worst Solution Ever Found
        if (Fval[i] > Wfval):
            Wfval = Fval[i]; w = i; 

    # Populate the popsize to 2 popsize
    for i in range(popsize, popsize2):
        Xreal[:,i] = Xreal[:,w]; 
        Xbin[i] = Xbin[w]; 
        Fval[i] = Fval[w];

    # Ranking Initial Population
    Pair = []; Order = [];
    for i in range(popsize2):
        Pair.append((i, Fval[i]));

    Pair.sort(key=takeSecond);
    for i in range(len(Fval)):
        Order.append(Pair[i][0]);
        Fits[i] = FitFun(Pair[i][1]);

    fb = Fits[:popsize].sum() / popsize; fmax = Fits[0]; AvegVals.append(fb);
    iter = 0; maxiter = 800; IsMinimized = 0; Tol = 1e-6;  
    while (iter < maxiter):
        # Selection/Reproduction (Elitism)
        for k in range(imax):
            i = random.randint(0, popsize); j = random.randint(0, popsize);
            fp = max(Fits[i], Fits[j]); pc = k3;
            if(fp >= fb):
                pc = k1 * (fmax - fp) / (fmax - fb);
            if(random.random() < pc):
                Reproduce(i, j, k, Order);

        # Convert to Real, Evaluate and Enqueue
        for i in range(popsize2):
            bin2real(i); Fval[i] = Objfun(i);
            if (Fval[i] < Bfval):
                Bfval = Fval[i]; sol = Xreal[:,i];


        # Sort to keep the next generation at the beginning
        Pair = []; Order = []; 
        for i in range(len(Fval)):
            Pair.append((i, Fval[i]));

        Pair.sort(key=takeSecond);
        for i in range(len(Fval)):
            Order.append(Pair[i][0]);
            Fits[i] = FitFun(Pair[i][1]);

        fb = Fits[:popsize].sum() / popsize; fmax = Fits[0]; AvegVals.append(fb);

        # Genetic Alteration (Mutation)
        for i in range(popsize):
            fi = Fits[i]; pm = k4;
            if(fi >= fb):
                pm = k2 * (fmax - fi) / (fmax - fb);
            if (random.random() < pm):
                Mutate(Order[i]);  

        # Check convergence
        if (iter > nvar):
            AvegmN = AvegVals.popleft();
            IsMinimized = mth.fabs(fb - AvegmN) < Tol;
            if (IsMinimized):
                break;

        iter += 1;
    print("iter = " + str(iter))
    return sol;

mp = -6.5511;
Lb = -3*np.ones(2); Ub = Lb+6;
def peak(v):
    x = v[0]; y = v[1]; x2 = x**2; y2 = y**2; xp1 = x + 1; yp1 = y + 1; x3 = x**3;
    y5 = y**5; xm1 = 1 - x; xm1_2 = xm1**2; yp1_2 = yp1**2; xp1_2 = xp1**2;
    f = 3 * xm1_2 * mth.exp(-x2 - yp1_2) - 10 * (x / 5 - x3 - y5) * mth.exp(-x2 - y2) - 1.0 / 3 * mth.exp(-xp1_2 - y2);
    return f;
def funInEq(v):
    f = 0;
    return f;
def funEq(v):
    f = 0;
    return f;
CorrecCount = 0;
for n in range(50):
    sol = GeneticAlgorithm(peak, funInEq, funEq, [], Lb, Ub);
    
    print(sol); fv = peak (sol); print(fv);
    if(mth.fabs((fv-mp)/mp) < 1e-3):
        CorrecCount += 1;
print("CorrecCount = " + str(CorrecCount));