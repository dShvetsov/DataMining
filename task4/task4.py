# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:15:34 2016

@author: dshvetsov
"""

import GradientBoosting
import numpy as np
import random
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

class SemiSupervisedGradientBoosting : 
    def __init__(self, max_depth=3, n_estimators=10, learning_rate=0.1,
                 min_samples_leaf=4, n_neighbors=5, n_components=2) :
        self.GB = GradientBoosting.GradientBoosting(max_depth, n_estimators,
                                   learning_rate, min_samples_leaf)
        self.Transformator = LocallyLinearEmbedding(n_neighbors, n_components)
        
    def fit_predict(self,Xl, y, Xu) :
        print 'start collapse space'
        delimeter = Xl.shape[0]        
        X_all = np.vstack((Xl, Xu))
        X_all = self.Transformator.fit_transform(X_all)
        X_l_t = X_all[:delimeter]
        X_u_t = X_all[delimeter:]
        del X_all
        print 'start compute simalirity'
        Sim = GradientBoosting.Simalirity(X_l_t, X_u_t)
        print 'end compute simalirity'        
        del X_l_t, X_u_t        
        #Xl = X_all[:delimeter]
        #Xu = X_all[delimeter:]
        print 'end collapse space succesfully'
        return self.GB.fit_predict(Xl, y, Xu, Sim)
        
    def predict(self,X) : 
        return self.GB.predict(X)
    def score (self, X, y) : 
        return self.GB.score(X, y)

def conc(a, b):
    h = (a == b).astype(float)
    return h.sum() / len(h)

def main():
    print 'start computing'
    reload (GradientBoosting)
    S = np.loadtxt('../spam.train.txt')
    X = S[:, 1:]
    y = S[:, 0]
    sample = np.array([random.randint(0, 4) for i in y])
    Xl = X[sample == 1]
    yl = y[sample == 1]
    del X, y
    del S
    S = np.loadtxt('../spam.test.txt')    
    Xu = S[:, 1:]
    yu = S[:, 0]
    sample = np.array([random.randint(0, 4) for i in yu])
    Xu = Xu[sample == 1]
    yu = yu[sample == 1]
    del S
    SSGB = SemiSupervisedGradientBoosting(n_estimators= 150)
    pr, Penaltys = SSGB.fit_predict(Xl, yl, Xu)
#    GB = GradientBoosting.GradientBoosting(n_estimators= 70)
    return SSGB, pr, [Xl, yl, Xu, yu], Penaltys

def graph(sc1, sc2) : 
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(len(sc1)), sc1, 'r-')
    plt.plot(range(len(sc2)), sc2, 'b-')
    plt.legend(['semisupervised', 'original'], loc ='lower right')
    plt.show()