# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 00:26:21 2016

@author: dshvetsov
"""

import task, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

F_set = range(0, 102)
f_max = 102

n_est = 50
max_depth = 200
max_features = 40

class featureSelection:
    def __init__(self, features_set, criterion, d=40):
        self.F = set(features_set)
        self.J = criterion
        self.d = d
        
    def selections(self, n=10):
        bfeatures = []
        Q = 0
        jbest = 0 
        for j in range(0, self.d):
            fbest, curQ = self.find_best_feature(bfeatures)
            bfeatures.append(fbest)
            if curQ > Q :
                jbest = j
                Q = curQ
                print Q
            if j - jbest >= n:
                return bfeatures
            print '----------',j
        return bfeatures
    
    def find_best_feature(self, bfeatures):
        curQ = 0
        fbest = None
        for i in self.F.difference(bfeatures):
            #print i, 
            tmp = self.J.crit(bfeatures + [i])            
            if curQ < tmp :
                curQ = tmp
                fbest = i
        self.J.say_about_best_feature(fbest)
        return fbest, curQ
        
def embeddedfunc(algo, X, y):
    algo.fit(X, y)
    return algo.bestfeature()

        
class embedded:
    def __init__ (self, algo, Xtest, ytest, X=None, y = None):
        self.algo = algo
        if (X is not None and y is not None):        
            self.algo.fit(X, y)
        self.shuffled = []
        for i in Xtest.T:
            j = i.copy()
            random.shuffle(j)
            self.shuffled.append(j)
        self.shuffled = np.array(self.shuffled).T
        self.Xtest = Xtest
        self.ytest = ytest
        
    def crit(self, subfeatures):
        T = self.shuffled.copy()
        T[:,subfeatures] = self.Xtest[:,subfeatures]
        return task.conc(self.ytest, self.algo.predict(T))
        
    def say_about_best_feature(self,f):
        pass
    
class wrapper:
    def __init__ (self, algo, X, y, Xtest, ytest):
        self.algo = algo
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = np.array(ytest)
        
    def crit(self, subfeature):
        #print subfeature
        if (len(subfeature) == 0):
            return None
        Xtmp = self.X[:,subfeature]
        self.algo.fit(Xtmp, self.y)    
        ans = task.conc(self.algo.predict(self.Xtest[:,subfeature]), self.ytest)
        self.algo.erase()
        return ans
        
    def say_about_best_feature(self,f):
        pass

class wrapper2:
    def __init__ (self,  X, y, Xtest, ytest):
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = np.array(ytest)
        
    def crit(self, subfeature):
        #print subfeature
        if (len(subfeature) == 0):
            return None
        algo = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                      max_features='auto')  
        Xtmp = self.X[:,subfeature]
        algo.fit(Xtmp, self.y)    
        ans = algo.score(self.Xtest[:,subfeature], self.ytest)
        return ans
        
    def say_about_best_feature(self,f):
        pass



    
def wrapp():
    s = np.loadtxt('../spam.train.txt')
    X = s[:, 1:]
    y = s[:, 0]
    s = np.loadtxt('../spam.test.txt')
    Xtest = s[:, 1:]
    ytest = s[:, 0]
    wr = wrapper2(X, y, Xtest, ytest)
    fs = featureSelection(F_set, wr, d=20)
    return fs.selections(n=5)
    
class filtermethod:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        Mat = np.hstack((self.X,self.y.reshape(self.y.shape[0], 1)))
        self.cor = np.abs(np.corrcoef(Mat.T))
        print self.cor.shape        
        
    def crit(self, subfeature):
        if len(subfeature) == 0: 
            return 0
        
        a = self.cor[subfeature, -1].sum()
        b = self.cor[subfeature, subfeature].sum()
        return float(a) / (float(b) ** 0.5)
    
    def say_about_best_feature(self, f):
        pass
     
def score(feat, X, y, Xtest, ytest):
    scor = []
    for i in range(1, len(feat) + 1):
        #rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
         #                           max_features='auto')
        clf = tree.DecisionTreeClassifier()        
        clf.fit(X[:,feat[:i]], y)
        scor.append(clf.score(Xtest[:, feat[:i]], ytest))
    return scor

def embb():
    reload(task)
    s = np.loadtxt('../spam.train.txt')
    X = s[:, 1:]
    y = s[:, 0]
    rnd = np.array([random.randint(0,5) for i in y])
    XS = X[rnd == 1]
    ys = y[rnd == 1]
    s = np.loadtxt('../spam.test.txt')
    Xtest = X#s[:, 1:]
    ytest = y#s[:, 0]
    rf = task.RandomForest(n_estimators=6, max_depth=7, max_features= 40 )
    rf.fit(X, y)
    ft = rf.bestfeature()
    return score(ft[:20], XS, ys, Xtest, ytest)
    
     

def e():
    s = np.loadtxt('../spam.train.txt')
    X = s[:,1:]
    y = s[:, 0]
    del(s)
    s = np.loadtxt('../spam.test.txt')
    #rnd = np.array([random.randint(0,5) for i in y])
    Xtest = s[:, 1:]
    ytest = s[:, 0]
    del(s)
    XS = X
    ys= y
    wrap = wrapper2(X=XS, y=ys, Xtest=Xtest, ytest=ytest)
    fswrap = featureSelection(F_set, wrap, d=10)
    featwrap = fswrap.selections(n=5)
    return featwrap
    
def main():
    reload(task)

    s = np.loadtxt('../spam.train.txt')
    X = s[:,1:]
    y = s[:, 0]
    #del(s)
    #s = np.loadtxt('../spam.test.txt')
    rnd = np.array([random.randint(0,5) for i in y])
    Xtest = X[rnd == 1]
    ytest = X[rnd == 1]
    del(s)
    #sub
   # Xtest = X
   # ytest = y
    XS = X[rnd != 1]
    ys = y[rnd != 1]
    print 'start wrapper'
    wrap = wrapper2(X=XS, y=ys, Xtest=Xtest, ytest=ytest)
    fswrap = featureSelection(F_set, wrap, d=50)
    featwrap = fswrap.selections(n=5)
    del (wrap, fswrap)
    print 'start embedded'
    rf = task.RandomForest(n_estimators=n_est, max_depth=max_depth, max_features=max_features)  
    rf.fit(XS, ys)    
    featemb = rf.bestfeature()
    print 'start filter'
    fil = filtermethod(XS, ys)
    fsfil = featureSelection(F_set, fil, d=50)
    featfil = fsfil.selections(n=10)
    del(fil, fsfil)
    featrand = random.sample(F_set, 50)
    wrapscore = score(featwrap, XS, ys, Xtest, ytest)
    embscore = score(featemb[:50], XS, ys, Xtest, ytest)
    filscore = score(featfil, XS, ys, Xtest, ytest)
    randscore = score(featrand, XS, ys, Xtest, ytest)
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(1, len(featwrap) + 1), wrapscore, 'g-')
    plt.plot(range(1, len(featemb[:50]) +1), embscore, 'b-')
    plt.plot(range(1, len(filscore) + 1), filscore, 'r-')
    plt.plot(range(1, len(featrand) + 1), randscore, 'black')
    plt.legend(['wrapped', 'embedded', 'filter', 'random'], loc='lower right')
    plt.title('feature selection')
    plt.xlabel('Number of features')
    plt.ylim(0, 1)
    plt.show()
    
   # return featwrap, featemb, featfil