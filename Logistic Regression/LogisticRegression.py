import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    # Class Attributes
    SGD_theta_list = []
    SGD_bias = 0
    BGD_theta_list = []
    BGD_bias = 0
    training_rate = 0
    FINAL_SGD_TRAIN_LOSS = []
    FINAL_SGD_VALIDATE_LOSS = []
    FINAL_BGD_TRAIN_LOSS = []
    FINAL_BGD_VALIDATE_LOSS = []

    def __init__(self):
        pass

    def set_training_rate(self, alpha):
        self.training_rate = alpha

    def update_thetas(self, X, y, theta_list, bias, learning_rate):
        """
        A function to update the values of thetas while training.
        """
        total_samples = len(X)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0

        for i in range(total_samples):                      #update the thetas and bias by gradient descent
            hypothesis = 0

            hypothesis = hypothesis+bias

            hypothesis += np.matmul(X[i], np.array(theta_list).T)

            sigmoidhypothesis = 1./(1.+np.exp(-hypothesis))

            sigmoidhypothesis = sigmoidhypothesis - y[i]

            bias_derivative += sigmoidhypothesis


            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += sigmoidhypothesis*X[i][feature_index]

        
        bias -= (bias_derivative/total_samples) * learning_rate

        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/total_samples) * learning_rate

        return bias, theta_list



    def cost_function(self, X, y, theta_list, bias):
        """
        Calculate the cost based on predictions.
        """
        total_samples = len(y)
        loss = 0

        for i in range(total_samples):
            hypothesis = bias
            hypothesis += np.matmul(X[i], np.array(theta_list).T)
            
            de = 1.0 + np.exp(-hypothesis)
            sigmoidhypothesis = 1.0/de

            loss += (y[i]*np.log(sigmoidhypothesis)) + ((1-y[i])*(np.log(1 - sigmoidhypothesis)))

        return -1 * (loss/total_samples)                #loss calculation


    def fit(self, X, y, X_validate, y_validate):
        """
        A Function to train the model. The test data is used for validation analysis (can be skipped).
        """
        
        iterate = 800
        
        self.SGD_theta_list = [0]*len(X[0])
        self.SGD_bias = 0

        SGD_cost_history = []
        SGD_validate_cost_history = []

        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")
            selection = random.randint(0, len(X)-1)         #selecting one random row for SGD
            temp_X = []
            temp_X.append(X[selection])
            temp_y = []
            temp_y.append(y[selection])
            self.SGD_bias, self.SGD_theta_list = self.update_thetas(np.array(temp_X), np.array(temp_y), self.SGD_theta_list, self.SGD_bias,self.training_rate)
            SGD_cost = self.cost_function(X, y, self.SGD_theta_list, self.SGD_bias)
            SGD_cost_history.append(SGD_cost)
            SGD_validate_cost = self.cost_function(X_validate, y_validate,self.SGD_theta_list, self.SGD_bias)
            SGD_validate_cost_history.append(SGD_validate_cost)

        self.FINAL_SGD_TRAIN_LOSS.append(SGD_cost_history[-1])
        self.FINAL_SGD_VALIDATE_LOSS.append(SGD_validate_cost_history[-1])

        plt.plot(list(range(iterate)), SGD_cost_history)
        plt.plot(list(range(iterate)), SGD_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss SGD")
        plt.show()
        
        
        self.BGD_theta_list = [0]*len(X[0])
        self.BGD_bias = 0

        BGD_cost_history = []
        BGD_validate_cost_history = []

        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")
            selection = random.randint(0, len(X)-1)
            
            self.BGD_bias, self.BGD_theta_list = self.update_thetas(X, y, self.BGD_theta_list, self.BGD_bias,self.training_rate)

            BGD_cost = self.cost_function(X, y, self.BGD_theta_list, self.BGD_bias)
            BGD_cost_history.append(BGD_cost)
            BGD_validate_cost = self.cost_function(X_validate, y_validate,self.BGD_theta_list, self.BGD_bias)
            BGD_validate_cost_history.append(BGD_validate_cost)

        self.FINAL_BGD_TRAIN_LOSS.append(BGD_cost_history[-1])
        self.FINAL_BGD_VALIDATE_LOSS.append(BGD_validate_cost_history[-1])

        plt.plot(list(range(iterate)), BGD_cost_history)
        plt.plot(list(range(iterate)), BGD_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss BGD")
        plt.show()

        print("FINAL_SGD_TRAIN_LOSS\n",self.FINAL_SGD_TRAIN_LOSS)
        print("FINAL_SGD_VALIDATE_LOSS\n",self.FINAL_SGD_VALIDATE_LOSS)
        print("FINAL_BGD_TRAIN_LOSS\n",self.FINAL_BGD_TRAIN_LOSS)
        print("FINAL_BGD_VALIDATE_LOSS\n",self.FINAL_BGD_VALIDATE_LOSS)

        
        return self

    def predict(self, X):
        """
        A function to predict the values on the basis of theta and bias calculated while training.
        Use two methods of Gradient Descents i.e. SGD and BGD
        """

        SGD_predicted = []
        for i in range(len(X)):
            hypothesis = 0
            hypothesis = self.SGD_bias
            hypothesis += np.matmul(X[i], np.array(self.SGD_theta_list).T)
            
            de = 1.0 + np.exp(-hypothesis)
            sigmoidhypothesis = 1.0/de

            if(sigmoidhypothesis>=0.5):
                SGD_predicted.append(1)                     #classification
            else:
                SGD_predicted.append(0) 
        
        BGD_predicted = []
        
        for i in range(len(X)):
            hypothesis = 0
            hypothesis = self.BGD_bias
            hypothesis += np.matmul(X[i], np.array(self.BGD_theta_list).T)
            
            de = 1.0 + np.exp(-hypothesis)
            sigmoidhypothesis = 1.0/de

            if(sigmoidhypothesis>=0.5):
                BGD_predicted.append(1)
            else:
                BGD_predicted.append(0) 
        

        return SGD_predicted, BGD_predicted

    def accuracy(self,predicted, original):
        """
        A Function to return the accuracy based on predicitons.
        """
        TP=0
        TN=0
        FP=0
        FN=0
        for i in range(len(predicted)):
            if(predicted[i]==1 and original[i]==1):
                TP+=1
            elif(predicted[i]==0 and original[i]==1):
                FN+=1
            elif(predicted[i]==1 and original[i]==0):
                FP+=1
            elif(predicted[i]==0 and original[i]==0):
                TN+=1

        acc = (TP+TN)/(TP+TN+FP+FN)
        return acc
