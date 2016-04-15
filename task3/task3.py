# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:03:03 2016

@author: dshvetsov
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
import pandas as pd
import random
import commands
import scipy

X = []

class trees:
    def __init__(self, Xi, yi):
        X = np.array(Xi)
        y = np.array(yi)
       # print y
       # print X.shape
        self.trees = []
        self.idx = []
        for i in range(10):
            subidx = random.sample(range(0,10), random.randint(7, 10))
            subsample = np.unique([random.randint(0, X.shape[0] - 1) for j in y])
            tree = DecisionTreeRegressor()
         #   print subsample, subidx
            tree.fit(X[subsample][:,subidx], y[subsample])
 #           tree.fit(X[subsample], y[subsample])
            #Xt = np.array(random.sample(X, len(X) *  0.8))            
            #tree.fit(X, y)

            self.idx.append(subidx)
            self.trees.append(tree)
        
    def compute(self, X, it):
        ans = []
        for i, tree in enumerate(self.trees):
                ans.append(tree.predict(X[:,self.idx[i]])) 
 #               ans.append(tree.predict(X)) 

        ans = np.array(ans)
      #  print ans.shape
        return np.var(ans, axis=0)
     
    def predict(self, X):
        ans = np.zeros(X.shape[0])
        for k, tr in enumerate(self.trees):
            ans = ans + tr.predict(X[:,self.idx[k]])
            #ans = ans + tr.predict(X)

        ans = ans  / float(len(self.trees))
        return ans
    
    def printpredict(self, X):
        X = np.array(X)
        for i, tr in enumerate(self.trees):
            print tr.predict(X[:, self.idx[i]]), '------'
            #print tr.predict(X), '------'
    
def oraclemid(x):
    return float(oracle(x[0], x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))

def oracle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    query = "java -cp OracleRegression.jar Oracle " + str(x1) + " " + str(x2) + " " + str(x3) + " " + \
            str(x4) + " " + str(x5) + " " + " " + str(x6) + " " + str(x7) + " " + \
            str(x8) + " " + str(x9) + " " + str(x10)

    return commands.getoutput(query)

def oraclemid2(x):
    return float(oracle2(x[0], x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))

def oracle2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    query = "java -cp Regr.jar Oracle " + str(x1) + " " + str(x2) + " " + str(x3) + " " + \
            str(x4) + " " + str(x5) + " " + " " + str(x6) + " " + str(x7) + " " + \
            str(x8) + " " + str(x9) + " " + str(x10)

    return commands.getoutput(query)

def max_dist(X):
    KM = KMeans(n_clusters=8)
    y = KM.fit_predict(np.array(X))
    Xt = []
    for i in range(0, 8):
        Xt.extend(X[y == i][:20])
    return Xt
    
def collectX(X, K):
    C = [X[0]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C    
    
def main():
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 500)
    y = []
    for i, x in enumerate(X):
        tmp = oraclemid2(x)        
        print i, tmp
        y.append(tmp)
    
    print 'start'
    
    for i in range(0, 100):
        print i
        t = trees(X, y)
        ans = t.compute(Xall, i+1)
        addedidx = ans.argsort()[::-1][:10000]
       # print ans[addedidx]
        Xtmp = Xall[addedidx]
        Xq = collectX(Xtmp, 10)
        """
        tst = np.array(Xq[:2])
        t.printpredict(tst)
        print('++++++++++++++++++++++++++++++++++++++++')
        print oraclemid2(tst[0]),
        print oraclemid2(tst[1])
        print('========================================================')
        print('________________________________________________________')
        idx = ans.argsort()[:2]        
        tst = np.array(Xall[idx])
        t.printpredict(tst)
        print('++++++++++++++++++++++++++++++++++++++++')
        print oraclemid2(tst[0]),
        print oraclemid2(tst[1])
        print('##########################################')
        print X[:5]
        print t.printpredict(X[-5:-1])
        print('++++++++++++++++++++++++++++++++++++++++++')
        for x in X[-5:-1]:
            print (oraclemid2(x)),
        print ''
        """
        for i in Xq:
            #print ',adding ', i, 
            X.append(i)
            y.append(oraclemid2(i))
        
    np.save('answer.txt', t.predict(Xall))
    t2 = DecisionTreeRegressor()
    t2.fit(np.array(X), np.array(y))
    t3 = GradientBoostingRegressor()
    t3.fit(np.array(X), np.array(y))    
    return t, t2, t3
    
def f():
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 40) 
    for x in X:
        print oraclemid(x), oraclemid2(x)

def g():
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 40) 
    for x in X:
        print oraclemid2(x)
    
    
def cmpp3(t):
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 20)
    for x in X:
        print abs(t.predict(x.reshape(1, 10))[0] -  oraclemid2(x))

def cmpp2(t):
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 20)
    for x in X:
        print t.printpredict(x.reshape(1, 10)), oraclemid2(x)
        

def cmpp(t):
    Xallhlp = pd.read_csv('X_public.csv')
    Xall = np.array(Xallhlp)[:, 1:]
    X = random.sample(Xall, 20)
    for x in X:
        print t.predict(x.reshape(1, 10)), oraclemid2(x)
    