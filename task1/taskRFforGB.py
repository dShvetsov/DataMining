# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random
import matplotlib.pyplot as plt

class CART:
    def __init__ (self, max_depth=2**32, min_samples_leaf=10):
        self._root = None
        self.max_depth = max_depth
        self.min_samples_leaf=min_samples_leaf
        self.gains = None
        
    def fit(self, X, y):
        self.gains = np.zeros(X.shape[1])
        depth = 0
        if depth > self.max_depth or self.criteria(y):
            self._root = TreeNode('Leaf')
            self._root.AssignClass(y)
        else:
            self._root = TreeNode('Node')
            r, t, g = self._root.computeSeparate(X, y)
            if (r is None or t is None):
                self._root.AssignClass(y)
            # print r, t
            else:
                self.gains[r] = self.gains[r] + g
                dividedset = X[:, r] < t
                self._root.addTrueChild(self.__fit(X[dividedset], y[dividedset], depth + 1))
                dividedset = dividedset == False
                self._root.addFalseChild(self.__fit(X[dividedset], y[dividedset], depth + 1))
        
                
    def __fit(self, X, y, depth):
        if depth > self.max_depth or self.criteria(y):
            L = TreeNode('Leaf')
            L.AssignClass(y)
        else:
            L = TreeNode('Node')
            r, t, g = L.computeSeparate(X, y)
            if (r is None or t is None):
                L.AssignClass(y)
                return L
            # print depth,
            self.gains[r] = self.gains[r] + g
            dividedset = X[:, r] < t
            L.addTrueChild(self.__fit(X[dividedset,:], y[dividedset], depth+1))
            dividedset = dividedset == False
            L.addFalseChild(self.__fit(X[dividedset,:], y[dividedset],depth+1))
        return L
        
    def criteria(self, y):
        q, w = np.unique(y, return_counts=True)
        return w.max() / len(y) >= 0.95 or len(y) < self.min_samples_leaf
    
    def predict(self, X):
        return np.array([self._root.predict(x) for x in X])

    def getgain(self):
     #   print 'CART gains', self.gains
        return self.gains

def computeProbabilities(y):
    ans = dict()
    tmp, ans = np.unique(y, return_counts=True)
    return tmp, ans / len(y)
    

    
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
        self.klass = y.sum() / len(y)        
        self.type = 'Leaf'
        self.TrueChild = None
        self.FalseChild = None
        self.value = None
        self.feature = None
        
        
    def computeSeparate(self, initX, y):
        X = initX.copy()        
        uni, cnt = np.unique(y, return_counts=True)
        dicti = dict(zip(uni, cnt))
        null_dicti = dict(zip(uni, np.zeros(len(uni))))
        all_sum = sum(map(lambda x: x*x, dicti.values()))
        bDelta = None
        bFeature = None
        ind = X.argsort(0)
        value = None
        gdelta = None
        for i, m in enumerate(ind.T):
            prev = 0#None
            yh = y[m]
            #Xh = X[m, i]
            left_dict = null_dicti.copy()
            right_dict = dicti.copy()
            right_sum = all_sum
            left_sum = 0
            right_count = len(y)
            left_count = 0
            for j, e in enumerate(yh):
                re = right_dict[e]
                right_dict[e] = re - 1
                right_sum = right_sum - 2 * re + 1
                right_count = right_count - 1
                le = left_dict[e]
                left_dict[e] = left_dict[e] + 1
                left_sum = left_sum + 2 * le + 1
                left_count = left_count + 1
                hvalue = X[m[j], i]
                if hvalue != prev and right_count > 1  and left_count > 1 : 
                    delta = float(right_sum) / right_count + float(left_sum) / left_count
                    if delta > bDelta :
#                        if sum(X[:,i] < hvalue) != 0 and sum(X[:, i] >= hvalue) != 0:                        
                         bDelta = delta
                         bFeature = i
                         value = float(hvalue + prev) / 2.0
                       #  value = float(hvalue + X[m[j+1], i]) / 2.0
                prev = hvalue
        if (bDelta is not None):                 
            gdelta = (float(bDelta) / len(y)) - (float(all_sum) / (len(y) ** 2))
        self.value = value
        self.feature = bFeature
        return (self.feature, self.value, gdelta)
            
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
            


class helptree:
    def __init__(self, features, max_depth = 2**32, min_samples_leaf=10):
        self.features= features
        self.features.sort()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
    def fit(self, X, y):
        self.__ftrs = X.shape[1]
        self.cart = CART(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.cart.fit(X[:,self.features], y)
    def predict(self, X):
        return self.cart.predict(X[:, self.features])
    def getgain(self):
        r = np.zeros(self.__ftrs)
        r[self.features] = self.cart.getgain()
       # print self.features
     #   print 'helptree gains', r
        return r
def conc(a, b):
    h = (np.array(a) == np.array(b)).astype(float)
    return h.sum() / len(h)

class RandomForest:
    def __init__(self,n_estimators=10, max_depth = 2**32, max_features=2**31, bootstrap = True,
                 min_samples_leaf=10):
        self.trees = []
        self.treecount = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.min_samples_leaf=min_samples_leaf
    def fit(self, X, y):
        self.__ftrs = X.shape[1]
        features = None
        samp = None
        for i in range(0, self.treecount):
            if (self.max_features < X.shape[1] ): 
                features = np.array(random.sample(range(0, X.shape[1]),self.max_features))
            else:
                features = range(0, X.shape[1])
            tree = helptree(features, max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf)
            if (self.bootstrap):
                samp = np.unique([random.randint(0, X.shape[0] - 1) for j in xrange(X.shape[0])])
            print i, 
            tree.fit(X[samp], y[samp])
            self.trees.append(tree)
            
    def predict(self, X):
        '''q = []
        for i in self.trees:
            q.append(i.predict(X))
        q=np.array(q)
        ans = []        
        for i in q.T:
            a, b = np.unique(i, return_counts=True)
            ans.append(a[b.argmax()])
        return np.array(ans)
    '''
        q = np.zeros(X.shape[0])
        for i in self.trees:
            q = q + i.predict(X)
        return (q >= 0.5).astype(int)
        
    def score(self, X, y):
        ans = []
        hlp = []
        for tree in self.trees:
            hlp.append(tree.predict(X))
            tmp = np.array(hlp)
            predic = []
            for i in tmp.T:
                a, b = np.unique(i, return_counts=True)
                predic.append(a[b.argmax()])
            ans.append(conc(np.array(predic), y))
        return np.array(ans)
        
    def erase(self):
        self.trees = []
    
    def bestfeature(self):
        q = np.zeros(self.__ftrs)
        for i in self.trees:
            q = q + i.getgain()
        #print 'RF gains', q
        return q.argsort()[::-1]

def main():
    sett = np.loadtxt('../spam.test.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    cart = CART(max_depth=10)
    cart.fit(X, y)
    return cart
    
def main3():
    sett = np.loadtxt('../spam.train.txt')
    import random
    X = sett[:, 1:]
    y = sett[:, 0]
    k = np.array([random.randint(0,4) for i in range(0, len(y))])
    c = CART(max_depth = 10)
    Xtrain = X[k == 0]
    ytrain = y[k == 0]
    Xtest = X[k != 0]
    ytest = y[k != 0]
    c.fit(Xtrain, ytrain)
    print(conc(c.predict(Xtest), ytest))
    return c
    
def mmain(_n_est):
    from sklearn.ensemble import RandomForestClassifier
    sett = np.loadtxt('../spam.train.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    test = np.loadtxt('../spam.test.txt')
    Xa = test[:, 1:]
    ya = test[:, 0]
    n_est = _n_est
    rf = RandomForest(max_features=50, bootstrap = True, max_depth=100, n_estimators=n_est, min_samples_leaf=10)
    rf.fit(X, y)

    #print conc(rf.predict(Xa), ya)
        
    score_train = rf.score(X, y)
    score_test = rf.score(Xa, ya)
    
    score_train_skl = []
    score_test_skl = []
    for i in range (1, n_est + 1):
        rf2 = RandomForestClassifier(n_estimators=i, max_depth=100,max_features=50, bootstrap =True )
        rf2.fit(X, y)
        score_train_skl.append(conc(y, rf2.predict(X)))
        score_test_skl.append(conc(ya, rf2.predict(Xa)))
    score_train_skl  = np.array(score_train_skl)
    score_test_skl = np.array(score_test_skl)
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(n_est), score_train, 'g-')
    plt.plot(range(n_est), score_train_skl, 'b-')
    plt.plot(range(n_est), score_train_skl - 0.03, 'r-')
    plt.legend(['myRandomForest', 'sklearnRandomForest', 'DANGER BOARD!!!!'], loc='lower right' )
    plt.title('Accurancy on train data(RandomForest)')
    plt.xlabel('Number of trees')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(range(n_est), score_test, 'g-')
    plt.plot(range(n_est), score_test_skl, 'b-')
    plt.plot(range(n_est), score_test_skl - 0.03, 'r-')
    plt.legend(['myRandomForest', 'sklearnRandomForest', 'DANGER BOARD!!!!'], loc='lower right')
    plt.title('Accurancy on test data(RandomForest)')
    plt.xlabel('Number of trees')
    plt.show()
    
def main2():   
        sett = np.loadtxt('../spam.train.txt')
        X = sett[:, 1:]
        y = sett[:, 0]
        rf = RandomForest(max_depth=7, n_estimators=10, max_features=50, bootstrap=True)
        rf.fit(X, y)
        return rf
        
def test2(rf):
    sett = np.loadtxt('../spam.test.txt')
    X = sett[:, 1:]
    y = sett[:, 0]
    ans = rf.predict(X)
    q = (ans == y).astype(float)
    return q.sum() / len(q)
    
def test(C, X, y):
    import random
    for j in range(0, 30, 5):
        r = np.array([random.randint(0, j) for i in y])
        pr = []
        for i in range(0, j):
            Xh = X[r == i]
            yh = y[r == i]
            ans = C.predict(Xh)
            comp = (yh == ans).astype(float)
            pr.append(comp.sum() / len(comp))
        print pr
    
if __name__=='__main__':
    mmain()
