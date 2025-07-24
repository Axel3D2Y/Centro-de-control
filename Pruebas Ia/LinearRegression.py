
import numpy as np 

class LinearRegression():

    def __init__(self, learning_rate= 0.001, max_iter=1000):
        self.learning_rate= learning_rate
        self.max_iter = max_iter
        self.weights= None
        self.bias= None

    def fit(self,X,y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            y_pred = np.dot(X,self.weights) + self.bias

            dw = ( 1/self.n_samples )*( np.dot( X,(y_pred-y) )) 
            db= ( 1/self.n_samples )*(y_pred-y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * dw

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias



