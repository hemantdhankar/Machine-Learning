import numpy as np
from random import random
import math
import matplotlib.pyplot as plt
np.random.seed(1)

class Layer:
    def __init__(self, n_inputs, n_neurons, weights):
        self.weights = weights
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def set_activations(self, activations):
        self.activations = activations

    def set_derivatives(self, derivatives):
        self.derivatives = derivatives
        
        

class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
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

        self.Layers_list = []
        for i in range(n_layers-1):
            if(weight_init == 'random'):
                weights = self.random_init((layer_sizes[i], layer_sizes[i+1]))
            elif(weight_init == 'zero'):
                weights = self.zero_init((layer_sizes[i], layer_sizes[i+1]))                
            elif(weight_init == 'normal'):
                weights = self.normal_init((layer_sizes[i], layer_sizes[i+1]))

            self.Layers_list.append(Layer(layer_sizes[i], layer_sizes[i+1], weights))

    def forward_propogate(self, inputs):
        self.inputs = np.array(inputs)
        for layer in self.Layers_list:
            #print("f=",inputs)
            outputs = layer.forward(inputs)
            if(self.activation == 'relu'):
                inputs = self.relu(outputs)
                layer.set_activations(inputs)
            elif(self.activation == 'sigmoid'):
                inputs = self.sigmoid(outputs)
                layer.set_activations(inputs)
            elif(self.activation == 'linear'):
                inputs = self.linear(outputs)
                layer.set_activations(inputs)
            elif(self.activation == 'tanh'):
                inputs = self.tanh(outputs)
                layer.set_activations(inputs)
            elif(self.activation == 'softmax'):
                inputs = self.softmax(outputs)
                layer.set_activations(inputs)
        return inputs

    def backward_propogate(self, error):
        for i in range(len(self.Layers_list)-1,0,-1):
            activations = self.Layers_list[i].activations
            if(self.activation == 'relu'):
               delta = error * self.relu_grad(activations)
            elif(self.activation == 'sigmoid'):
               delta = error * self.sigmoid_grad(activations)
            elif(self.activation == 'linear'):
               delta = error * self.linear_grad(activations)
            elif(self.activation == 'tanh'):
               delta = error * self.tanh_grad(activations)
            elif(self.activation == 'softmax'):
               delta = error * self.softmax_grad(activations)

            delta_reshaped = delta.reshape(delta.shape[0],-1).T

            previous_activation = self.Layers_list[i-1].activations
            previous_activation_reshaped = previous_activation.reshape(previous_activation.shape[0],-1)
            self.Layers_list[i].derivatives = np.dot(previous_activation_reshaped, delta_reshaped)
            self.Layers_list[i].derivatives_bias = np.sum(delta_reshaped)

            error = np.dot(delta, self.Layers_list[i].weights.T)
            #print("derivatives for ", i,"\n", self.Layers_list[i].derivatives)

        activations = self.Layers_list[0].activations
        if(self.activation == 'relu'):
           delta = error * self.relu_grad(activations)
        elif(self.activation == 'sigmoid'):
           delta = error * self.sigmoid_grad(activations)
        elif(self.activation == 'linear'):
           delta = error * self.linear_grad(activations)
        elif(self.activation == 'tanh'):
           delta = error * self.tanh_grad(activations)
        elif(self.activation == 'softmax'):
           delta = error * self.softmax_grad(activations)

        delta_reshaped = delta.reshape(delta.shape[0],-1).T

        previous_activation = self.inputs
        previous_activation_reshaped = previous_activation.reshape(previous_activation.shape[0],-1)
        self.Layers_list[0].derivatives = np.dot(previous_activation_reshaped, delta_reshaped)
        self.Layers_list[0].derivatives_bias = np.sum(delta_reshaped)

        error = np.dot(delta, self.Layers_list[0].weights.T)

        #print("derivatives for 0 \n", self.Layers_list[0].derivatives)

        return error


    def gradient_descent(self):
        for i in range(len(self.Layers_list)):
            derivatives = self.Layers_list[i].derivatives
            derivatives_bias = self.Layers_list[i].derivatives_bias
            #weights = self.Layers_list[i].weights
            #biases = self.Layers_list[i].biases

            #print("\nWeights for ", i, " before update:\n", self.Layers_list[i].weights)
            self.Layers_list[i].weights += self.learning_rate * derivatives
            #print("\nWeights for ", i, " after update")
            #print(weights)

            #print("\nBiases for ", i, " before update")
            #print(biases)
            self.Layers_list[i].biases += self.learning_rate * derivatives_bias
            #print("\nWeights for ", i, " after update:\n", self.Layers_list[i].weights)


    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = np.maximum(0,X)
        return x_calc

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = 1/(1+np.exp(-X))
        return x_calc

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X * (1.0 - X)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = np.tanh(X)
        return x_calc

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = np.exp(X) / (np.sum(np.exp(X)))
        return x_calc

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """

        return np.random.rand(shape[0], shape[1])

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return numpy.random.normal(0, 1, shape)*0.01

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

        # fit function has to return an instance of itself or else it won't work with test.py
        error_history = []
        for i in range(self.num_epochs):
            total_error = 0
            for sample_number in range(len(X)):
                inputs = X[sample_number]
                label = y[sample_number]

                predicted = self.forward_propogate(inputs)
                error = label-predicted
                self.backward_propogate(error)
                self.gradient_descent()
                
                total_error += self.mse_cost_function(label, predicted)

            print("Error EPOCH ",i,": ", total_error/len(X))
            error_history.append(total_error/len(X))

        plt.plot(list(range(self.num_epochs)), error_history)
        plt.show()

        return self

    def mse_cost_function(self, label, predicted):
        return np.average(math.pow((label-predicted), 2))

    def logloss(y, a):
        return -(y*np.log(a) + (1-y)*np.log(1-a))

    def d_logloss(y, a):
        return (a - y)/(a*(1 - a))

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

        # return the numpy array y which contains the predicted values
        return None

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

        # return the numpy array y which contains the predicted values
        return None

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

        # return the numpy array y which contains the predicted values
        return None


if __name__ == '__main__':
    
    mnn = MyNeuralNetwork(3, np.array([2,5,1]), 'sigmoid', 0.1, 'random', 32, 50)

    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    mnn.fit([[0,0],[0,1],[1,0],[1,1]], [0,1,1,0])
