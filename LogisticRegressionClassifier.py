import numpy as np
import warnings
warnings.simplefilter("ignore")

class LogisticRegression:
    
    def __init__(self, learningRate = 0.001, epochs = 10000, fit_intercept = True):
        self.learningRate = learningRate
        self.epochs = epochs
        self.fit_intercept = fit_intercept 
    
    #Padding with 1's as first column for bias    
    def addIntercept(self, X):
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis = 1)
    
    def get_weights(self):
            return self.weights
    
    def sigmoid(self, z):   
        return (1/(1 + np.exp(-z)))
  
    #Weights initialised with random values
    def weightInitialisation(self, X):
        self.weights = np.random.rand(X.shape[1], 1)

    def fit(self, X, y):
                
        if self.fit_intercept:
            X = self.addIntercept(X)            
        
        self.nClasses = len(np.unique(y))
        if (self.nClasses == 2):
            self.fitBinaryClass(X, y)
        else:
            self.fitMultiClass(X, y)
           
    def fitBinaryClass(self, X, y):
        
        #Initialising Weights
        self.weightInitialisation(X)
        # Converting y into a column vector to match y_hat
        y = y.reshape((-1, 1))
        
        for IterNum in range(self.epochs): 
            z = np.dot(X, self.weights)
            y_hat = self.sigmoid(z)
            gradient = np.dot(X.T, (y_hat - y))
            #cost = (y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
            self.weights -= self.learningRate*gradient            
     
    #Treating as n Binary Classification Problems    
    def fitMultiClass(self, X, y):
        
        self.multiweights = np.array([])
        for i in range(self.nClasses):
            #Making i-th class 1 and rest 0
            temp_y = y==i 
            #Converting bool array to int array
            boolToInt = np.vectorize(lambda x: 1 if x else 0)
            yi = boolToInt(temp_y)
            
            self.fitBinaryClass(X, yi)
            self.multiweights = np.append(self.multiweights, self.get_weights())
            
        self.multiweights = self.multiweights.reshape(self.nClasses, -1)
      
    def predict(self, X):
        
        if self.fit_intercept:
            X = self.addIntercept(X)
            
        if self.nClasses == 2:
            y_pred = self.binaryPredict(X)
        else:
            y_pred = self.multiPredict(X)       
        return y_pred

    def binaryPredict(self, X):
        
        z = np.dot(X, self.weights)
        y_hat = self.sigmoid(z) 
        
        pred = lambda x: 0 if x<0.5 else 1
        y_pred = map(pred, y_hat)
        return y_pred
    
    #Choosing class with highest prob     
    def multiPredict(self, X):

        z = np.dot(X, self.multiweights.T)
        y_hat = self.sigmoid(z)
        y_pred = np.array([], dtype = int)
        
        for eachrow in y_hat:          
            eachlist = eachrow.tolist()
            index = eachlist.index(max(eachlist))
            y_pred = np.append(y_pred, index)
        
        return y_pred
    
    def accuracy_score(self, y_pred, yans, normalize = True):
            
        if (len(y_pred) != len(yans)):
            return -1
        count = 0
        for i,j in zip(y_pred, yans):
            count += int(i==j)    
    
        if not normalize:
            return count
        acc = count/len(y_pred)
        return acc
            
    def score(self, X, y):      

        y_pred = self.predict(X)
        accuracy = self.accuracy_score(y_pred, y)
        return accuracy