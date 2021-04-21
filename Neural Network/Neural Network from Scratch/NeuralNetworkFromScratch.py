i# -*- coding: utf-8 -*-

"""
Author: Hemant Dhankar
CSB, IIIT DELHI
"""

"""
IMPORTING LIBRARIES
"""

from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
import numpy as np
import pandas as pd
import seaborn as sns
#from cuml.manifold import TSNE 
from random import random
import math
import matplotlib.pyplot as plt
np.random.seed(1)



"""
LAYER CLASS THAT REPRESENT A SINGLE LAYER OF NEURAL NETWORK
"""
class Layer:
    def __init__(self, n_inputs, n_neurons, weights):
        self.weights = weights
        self.biases = np.zeros(shape= (n_neurons,1))

    """
    FORWARD FEEDING: CALCULATING OUTPUTS OF A LAYER USING WEIGHTS AND BIASES
    """

    def forward(self, inputs, typ):
        output = np.dot(self.weights, inputs) + self.biases
        if(typ=="train"):      
          self.inputs = inputs
          self.output = output
        """
        #print(self.weights.shape,"Weights:\n",self.weights)
        #print(self.biases.shape, "Biases:\n",self.biases)
        #print("Inputs:\n", inputs)
        #print("Output:\n", self.output)
        """
        return output

    def set_derivatives(self, derivatives):
        self.derivatives = derivatives
        
        

class MyNeuralNetwork():
    """
    Implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        
        n_layers : int value specifying the number of layers
        
        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer
        
        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh
        
        learning_rate : float value specifying the learning rate to be used
        
        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal
        
        batch_size : int value specifying the batch size to be used
        
        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        pass

        self.n_layers = n_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_hidden_layers = layer_sizes[1:-1]
        self.num_inputs = layer_sizes[0]
        self.num_outputs = layer_sizes[-1]

        """
        CREATING A LAYER LIST THAT WILL STORE ALL THE LAYERS OF THE NEURAL NETWORK
        """

        self.Layers_list = []
        for i in range(n_layers-1):

            """
            WEIGHTS INITIALIZATION ACCORDING TO PARAMETER
            """
            if(weight_init == 'random'):
                weights = self.random_init((layer_sizes[i+1], layer_sizes[i]))
            elif(weight_init == 'zero'):
                weights = self.zero_init((layer_sizes[i+1], layer_sizes[i]))                
            elif(weight_init == 'normal'):
                weights = self.normal_init((layer_sizes[i+1], layer_sizes[i]))

            self.Layers_list.append(Layer(layer_sizes[i], layer_sizes[i+1], weights))


    """
    FORWARD FEEDING: FUNCTION TO CALCULATE THE OUTPUT OF NEURAL NETWORK. 
    """
    def forward_propogate(self, inputs, typ="train"):
        for i in range(len(self.Layers_list)-1):
            ##print("f=",inputs)
            outputs = self.Layers_list[i].forward(inputs, typ)
            """
            APPLYING ACTIVATION FUNCTION ON THE OUTPUT OF A LAYER
            """
            if(self.activation == 'relu'):
                inputs = self.relu(outputs)
            elif(self.activation == 'sigmoid'):
                inputs = self.sigmoid(outputs)
            elif(self.activation == 'linear'):
                inputs = self.linear(outputs)
            elif(self.activation == 'tanh'):
                inputs = self.tanh(outputs)
            elif(self.activation == 'softmax'):
                inputs = self.softmax(outputs)
        """
        SEPERATE ACTIVATION FUNCTION FOR LAST LAYER
        """
        outputs = self.Layers_list[i+1].forward(inputs, typ)
        inputs = self.sigmoid(outputs)

        return inputs

    """
    BACKWARD PROPOGATION : FUNCTION TO TRAIN THE MODEL AND APPLYING GRADIENT DESCENT.
    """
    def backward_propogate(self, error):
        i = len(self.Layers_list)-1
        activations = self.Layers_list[i].output
        delta = np.multiply(self.sigmoid_grad(activations), error)
        """
        CALCULATING DERIVATIVES
        """
        self.Layers_list[i].derivatives = (1/delta.shape[1]) * np.dot(delta, self.Layers_list[i].inputs.T )
        self.Layers_list[i].derivatives_bias = (1/delta.shape[1]) * np.sum(delta, axis = 1, keepdims=True)

        error = np.dot(self.Layers_list[i].weights.T, delta)

        self.Layers_list[i].weights = self.Layers_list[i].weights - (self.learning_rate * self.Layers_list[i].derivatives)
        self.Layers_list[i].biases = self.Layers_list[i].biases - (self.learning_rate * self.Layers_list[i].derivatives_bias)
        
        """
        ITERATING BACKWARD IN THE LAYERS LIST
        """
        for i in range(len(self.Layers_list)-2,-1,-1):
            activations = self.Layers_list[i].output
            #print("Activations: \n", activations)
            """
            CALCULATING GRADIENT OF ACTIVATION FUNCTION
            """
            if(self.activation == 'relu'):
               delta = error * self.relu_grad(activations)
            elif(self.activation == 'sigmoid'):
               delta = np.multiply(self.sigmoid_grad(activations), error)
            elif(self.activation == 'linear'):
               delta = error * self.linear_grad(activations)
            elif(self.activation == 'tanh'):
               delta = error * self.tanh_grad(activations)
            elif(self.activation == 'softmax'):
               delta = error * self.softmax_grad(activations)

            #print("Delta: ", delta)
            #print("Inputs:\n", self.Layers_list[i].inputs.T)
            self.Layers_list[i].derivatives = (1/delta.shape[1]) * np.dot(delta, self.Layers_list[i].inputs.T )
            self.Layers_list[i].derivatives_bias = (1/delta.shape[1]) * np.sum(delta, axis = 1, keepdims=True)

            ##print(self.Layers_list[i].weights.shape)
            ##print(self.Layers_list[i].derivatives.shape)


            error = np.dot(self.Layers_list[i].weights.T, delta)
            #print("derivatives:\n", self.Layers_list[i].derivatives)

            """
            UPDATING WEIGHTS AND BIASES
            """
            self.Layers_list[i].weights = self.Layers_list[i].weights - (self.learning_rate * self.Layers_list[i].derivatives)
            self.Layers_list[i].biases = self.Layers_list[i].biases - (self.learning_rate * self.Layers_list[i].derivatives_bias)
            
        ##print("derivatives for 0 \n", self.Layers_list[0].derivatives)

        return error

    """
    FUNCTION THAT RETURNS THE FEATURES OF LAST HIDDEN LAYER FOR TSNE ANALYSIS
    """
    def tsne_features(self, inputs, typ="test"):
      for i in range(len(self.Layers_list)-1):
          ##print("f=",inputs)           
          outputs = self.Layers_list[i].forward(inputs, typ)
          if(self.activation == 'relu'):
              inputs = self.relu(outputs)
          elif(self.activation == 'sigmoid'):
              inputs = self.sigmoid(outputs)
          elif(self.activation == 'linear'):
              inputs = self.linear(outputs)
          elif(self.activation == 'tanh'):
              inputs = self.tanh(outputs)
          elif(self.activation == 'softmax'):
              inputs = self.softmax(outputs)

      return inputs.T   

    def gradient_descent(self):
        for i in range(len(self.Layers_list)):
            derivatives = self.Layers_list[i].derivatives
            derivatives_bias = self.Layers_list[i].derivatives_bias
            #weights = self.Layers_list[i].weights
            #biases = self.Layers_list[i].biases

            ##print("\nWeights for ", i, " before update:\n", self.Layers_list[i].weights)
            self.Layers_list[i].weights += self.learning_rate * derivatives.T
            ##print("\nWeights for ", i, " after update")
            ##print(weights)

            ##print("\nBiases for ", i, " before update")
            ##print(biases)
            self.Layers_list[i].biases += self.learning_rate * derivatives_bias
            ##print("\nWeights for ", i, " after update:\n", self.Layers_list[i].weights)


    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer
        """
        cop = np.copy(X)
        cop[cop<0] = cop[cop<0]*0.01
        return cop

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer
        """
        dx = np.ones_like(X)
        dx[X<0] = 0.01
        return dx

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer
        """
        x_calc = 1/(1+np.exp(-X))
        return x_calc

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer
        """
        return self.sigmoid(X) * (1 - self.sigmoid(X)) 

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer
        """
        return np.zeros(X.shape)+1

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer
        """
        x_calc = np.tanh(X)
        return x_calc

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer
        """
        return 1-((self.tanh(X))**2)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer
        """
        x_calc = np.exp(X) / (np.sum(np.exp(X)))
        return x_calc

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer
        """
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer
        """
        return np.zeros(shape)+1

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer
        """

        return np.random.randn(shape[0], shape[1])*0.01

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer
        """
        return np.random.normal(0, 1, shape)*0.01

    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.
        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        """
        SPLITTING DATA FOR VALIDATION 
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

        error_history = []
        error_history_test = []
        num_batches = int(X_train.shape[0]/self.batch_size)
        print("rt=",len(X_train))

        """
        DIVIDING THE TRAINING DATA INTO BATCHES ACCORDING TO GIVEN BATCH SIZE
        """
        X = np.array_split(X_train, num_batches)
        print(len(X))
        y = np.array_split(y_train, num_batches)
        X_test = X_test.T
        y_test = y_test.T

        """
        RUNNING EPOCHS
        """
        for i in range(self.num_epochs):
          """
          ITERATING FOR BATCHES
          """
          for j in range(len(X)):  
            inputs = X[j].T
            label = y[j].T
            total_error = 0
            #for sample_number in range(len(X)):

            predicted = self.forward_propogate(inputs)

            #print("Predicted:\n", predicted)

            de = self.d_logloss(label, predicted)
            #print("De= ",de)
            self.backward_propogate(de)
            #self.gradient_descent()
          
          """
          CALCULATING TRAINING ERROR AND VALIDATION ERROR FOR EACH ECHO
          """
          error = (1/(inputs.shape[1])) * np.sum(self.logloss(label, predicted))
          predicted_test = self.forward_propogate(X_test, "test")
          error_test = (1/(X_test.shape[1])) * np.sum(self.logloss(y_test, predicted_test))
            
          error_history.append(error)
          error_history_test.append(error_test)
          print("Error EPOCH Training",i,": ", error, " ::  Error EPOCH Testing",i,": ", error_test)


        """
        PLOTTING LOSS VS EPOCHS
        """

        plt.plot(list(range(len(error_history))), error_history, label = "Training Error")
        plt.plot(list(range(len(error_history_test))), error_history_test, label = "Testing Error")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Training and Validation Error vs Epochs for "+ str(self.activation))
        plt.legend()
        plt.show()


        
        weights_to_save = []
        for layer in self.Layers_list:
          weights_to_save.append(layer.weights)

        """
        SAVING WEIGHTS USING PICKLE
        """
        pickle.dump(np.array(weights_to_save), open(self.activation+'.weights', 'wb'))
        
        return self

    def mse_cost_function(self, label, predicted):
        return np.average(math.pow((label-predicted), 2))

    def logloss(self, y, a):
        N = a.shape[0]
        return (-np.sum(y * np.log(a))) / N

    def d_logloss(self, y, a):
        return a-y

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
        
        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """
        X_test = X.T

        predicted = self.forward_propogate(X_test, "test")
        #print(predicted.shape)

        predicted = predicted.T
        return predicted

    def predict(self, X):
        """
        Predicting values using the trained linear model.
        
        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
        
        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

    
        X_test = X.T

        predicted = self.forward_propogate(X_test, "test")
        #print(predicted.shape)

        predicted = predicted.T
        final_predictions = np.argmax(predicted, axis = 1)

        return final_predictions

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.
        
        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        predictions = self.predict(X)
        correct = 0
        for i in range(len(predictions)):
          if(predictions[i]==y[i]):
            correct+=1

        return correct/len(predictions)


if __name__ == '__main__':
    
    mnn = MyNeuralNetwork(3, np.array([2,5,1]), 'sigmoid', 0.1, 'random', 32, 50)

    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    mnn.fit([[0,0],[0,1],[1,0],[1,1]], [0,1,1,0])
