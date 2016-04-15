# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 03:20:37 2016

@author: dshvetsov
"""

import numpy as np
import task as RandomForest
import matplotlib.pyplot as plt

    
class TreeNode:        
    def __init__ (self, type ='Node'):
        self.type = type
        self.TrueChild = None
        self.FalseChild = None
        self.value = None
        self.feature = None
        self.klass = None
        
    def AssignClass(self, y):
        #uniq, count = np.unique(y, return_counts=True)
        #self.klass =  uniq[count.argmax()]
        self.klass= y.sum() / len(y)
        
    def computeSeparate(self, initX, y):
        X = initX.copy()        

        bDelta = None
        bFeature = None
        ind = X.argsort(0)
        value = None
    
        for i, m in enumerate(ind.T):
            prev = 0
            yh = y[m]
            left_sum = 0
            left_sqr_sum = 0
            
            right_sum = y.sum()
            right_sqr_sum = sum(map(lambda x : x * x, y))
            left_sum = 0
            right_count = len(y)
            left_count = 0
            for j, e in enumerate(yh):
                right_sum = right_sum - e
                right_sqr_sum = right_sqr_sum - e * e
                right_count = right_count - 1
                
                left_sum= left_sum + e
                left_sqr_sum = left_sqr_sum + e * e
                left_count = left_count + 1
                hvalue = X[m[j], i]
                
                if hvalue != prev and right_count > 1  and left_count > 1 : 
                    delta = (- left_sqr_sum  + float(left_sum ** 2) / left_count) + ( -right_sqr_sum + float(right_sum ** 2)  / right_count)
                    if delta > bDelta :
                         bDelta = delta
                         bFeature = i
                         value = float(hvalue + prev) / 2.0
                prev = hvalue
        self.value = value
        self.feature = bFeature
        return (self.feature, self.value)
            
    def addTrueChild(self, son):
        self.TrueChild = son

    def addFalseChild(self, son):
        self.FalseChild = son

    def predict(self, x):
        if (self.type == 'Leaf'):
            return self.klass
        elif x[self.feature] < self.value:
            return self.TrueChild.predict(x)
        else:
            return self.FalseChild.predict(x)


class CART:
    def __init__ (self, max_depth=2**32, min_samples_leaf=4):
        self._root = None
        self.max_depth = max_depth
        self.min_samples_leaf=min_samples_leaf
        
    def fit(self, X, y):
        depth = 0
        if depth >= self.max_depth or self.criteria(y):
            self._root = TreeNode('Leaf')
            self._root.AssignClass(y)
        else:
            self._root = TreeNode('Node')
            r, t = self._root.computeSeparate(X, y)
           # print r, t
            dividedset = X[:, r] < t
            self._root.addTrueChild(self.__fit(X[dividedset], y[dividedset], depth + 1))
            dividedset = dividedset == False
            self._root.addFalseChild(self.__fit(X[dividedset], y[dividedset], depth + 1))
        
                
    def __fit(self, X, y, depth):
        if depth >= self.max_depth or self.criteria(y):
            L = TreeNode('Leaf')
            L.AssignClass(y)
        else:
            L = TreeNode('Node')
            r, t = L.computeSeparate(X, y)
            dividedset = X[:, r] < t
            L.addTrueChild(self.__fit(X[dividedset], y[dividedset], depth+1))
            dividedset = dividedset == False
            L.addFalseChild(self.__fit(X[dividedset], y[dividedset],depth+1))
        return L
        
    def criteria(self, y):
        q, w = np.unique(y, return_counts=True)
        return w.max() / len(y) >= 0.95 or len(y) < self.min_samples_leaf
    
    def predict(self, X):
        return np.array([self._root.predict(x) for x in X])

def computeProbabilities(y):
    ans = dict()
    tmp, ans = np.unique(y, return_counts=True)
    return tmp, ans / len(y)
    


class GradientBoosting:
    def __init__ (self, max_depth=3, n_estimators=10, learning_rate=0.1, min_samples_leaf=4 ):
        self.max_depth = max_depth
        self.count_trees = n_estimators
        self.learning_rate = learning_rate * 5
        self.min_samples_leaf = min_samples_leaf
    def fit(self, X, y):
        self.trees = []
        tree = CART(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        null = np.zeros(len(y))        
        tree.fit(X, null)
        self.trees.append(tree)
        h = tree.predict(X) * self.learning_rate
        for i in range(self.count_trees - 1):
            print i,
            grad = y - 1.0 / (1 + np.exp(-h))
            #grad = 2*(y - h)               
            if (np.linalg.norm(grad) < 1.5):
                tree = CART(max_depth=self.max_depth * 2, min_samples_leaf=self.min_samples_leaf)
            else:
                tree = CART(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, grad)
            self.trees.append(tree)
            h = h + self.learning_rate * tree.predict(X)
    
    def predict(self, X):
        h = np.zeros(X.shape[0])
        for tree in self.trees:
            h = h + self.learning_rate * tree.predict(X)
        #return h
        return np.apply_along_axis(lambda x: (1.0 / (1 + np.exp(-x)) > 0.5) * 1.0, 0, h)
     
    def score(self, X, y) :
        h = np.zeros(X.shape[0])
        score_y = []
        for tree in self.trees:
            h = h + self.learning_rate * tree.predict(X)
            y_pred = np.apply_along_axis(lambda x: (1.0 / (1 + np.exp(-x)) > 0.5) * 1.0, 0, h)
            score_y.append(conc(y, y_pred))
        return np.array(score_y)       
    
def conc(a, b):
    h = (np.array(a) == np.array(b)).astype(float)
    return h.sum() / len(h)    
    
def mmain():
    sett = np.loadtxt('../spam.train.txt')
    X = sett[:, 1:]
    y = sett[:, 0]    
    Xf = X
    yf = y
    n_est = 50
    rate = 0.1
    gb = GradientBoosting(learning_rate=rate, n_estimators=n_est)    
    gb.fit(Xf, yf)
    return gb
    
def qw(gb):
    test = np.loadtxt('../spam.test.txt')
    X = test[:, 1:]
    y = test[:, 0]
    h = gb.predict(X)
    sc = []
    steps = np.arange(0, 1, 0.01)    
    for step in steps:
        sc.append(conc( (h > step).astype(int), y))
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(steps, sc)
    plt.xlabel('threshold')
    plt.ylabel('accurancy')
    
def main():
    print 'start'
    from sklearn.ensemble import GradientBoostingClassifier
    sett = np.loadtxt('../spam.train.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    #import random
    #rnd = np.array([random.randint(0,4) for i in range(len(y))])
    Xf = X
    yf = y
    test = np.loadtxt('../spam.test.txt')
    Xa = test[:, 1:]
    ya = test[:, 0]
    n_est = 300
    rate = 0.1

    gb = GradientBoosting(learning_rate=rate, n_estimators=n_est)
    gb.fit(Xf, yf)
    return gb
    
    print conc(gb.predict(Xa), ya)
        
    score_train = gb.score(X, y)
    score_test = gb.score(Xa, ya)
    gb2 = GradientBoostingClassifier(learning_rate=rate, n_estimators=n_est)
    gb2.fit(Xf, yf)
    
    score_train_skl = []
    for pred in gb2.staged_predict(X):
        score_train_skl.append(conc(y, pred))
    score_train_skl  = np.array(score_train_skl)
    
    score_test_skl = []
    for pred in gb2.staged_predict(Xa):
        score_test_skl.append(conc(ya, pred))
    score_test_skl  = np.array(score_test_skl)    
    
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(n_est), score_train, 'g-')
    plt.plot(range(n_est), score_train_skl, 'b-')
    plt.plot(range(n_est), score_train_skl - 0.03, 'r')
    plt.legend(['myGradientBoosting', 'sklearnGradientBoosting', 'Danger board!!!!'], loc='lower right' )
    plt.title('Accurancy on train data(GradientBoosting)')
    plt.xlabel('Number of trees')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(n_est), score_test, 'g-')
    plt.plot(range(n_est), score_test_skl, 'b-')
    plt.plot(range(n_est), score_test_skl - 0.03, 'r')
    plt.legend(['myGradientBoosting', 'sklearnGradientBoosting', 'Danger board!!!'], loc='lower right')
    plt.title('Accurancy on test data(GradientBoosting)')
    plt.xlabel('Number of trees')
    plt.show()
    
def main2():
    import random
    sett = np.loadtxt('../spam.test.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    cart = CART(max_depth=10)
    s = np.array([random.randint(0,4) for i in range(0, y.shape[0])] )   
    cart.fit(X[s==0], y[s==0])
    import sklearn.tree
    g = sklearn.tree.DecisionTreeClassifier(max_depth=10)
    g.fit(X[s==0], y[s==0])
    return cart, g
    
def test2(rf):
    sett = np.loadtxt('../spam.train.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    ans = np.round(rf.predict(X))
    q = (ans == y).astype(float)
    return q.sum() / len(q)