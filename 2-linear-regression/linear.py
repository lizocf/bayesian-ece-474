import numpy as np   
import pandas as pd 
import matplotlib.pyplot as plt 
  
# Linear Regression 
class LinearRegression() : 
    def __init__(self, learning_rate, iterations) : 
        self.learning_rate = learning_rate
        self.iterations = iterations 
        self.params = []
              
    def fit(self, X, Y) : 
          
        # no_of_training_examples, no_of_features 
        self.m, self.n = X.shape 
          
        # weight initialization 
        self.W = np.zeros(self.n) 
    
        self.b = 0      
        self.X = X 
        self.Y = Y 
          
          
        # gradient descent learning 
        for i in range(self.iterations):
            self.update_weights() 
              
        return self
      
    def update_weights(self): 
        '''
        We know MSE = 1/m * sum((Y - Y_pred)^2), so calculating gradients:
        -> d(MSE)/dW = -2/m * sum(X * (Y - Y_pred))
        -> d(MSE)/db = -2/m * sum(Y - Y_pred)
        '''

        Y_pred = self.predict(self.X) 
          
        # weight gradient 
        dW = -(2 * (self.X.T).dot(self.Y - Y_pred)) / self.m 
       
        # bias gradient
        db = -2 * np.sum( self.Y - Y_pred ) / self.m  
          
        # update weights 
        self.W = self.W - self.learning_rate * dW 
        self.b = self.b - self.learning_rate * db 
          
        return self
      
      
    def predict(self, X): # h(X) = X.W + b 
      
        return X.dot( self.W ) + self.b 
    
    def sample_params(self):
        self.params.append([self.W, self.b])
        return self.params
     