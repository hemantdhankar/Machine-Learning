import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MyLinearRegression():
    """
    Implementation of Linear Regression.
    """

    """
    Class Attributes
    """
    mae_theta_list = []
    mae_bias = 0
    rmse_theta_list = []
    rmse_bias = 0
    training_rate = 0
    FINAL_MAE_TRAIN_LOSS = []
    FINAL_MAE_VALIDATE_LOSS = []
    FINAL_RMSE_TRAIN_LOSS = []
    FINAL_RMSE_VALIDATE_LOSS = []

    def __init__(self):
        pass

    def set_training_rate(self, alpha):
        self.training_rate = alpha

    def rmse_update_thetas(self, features, global_sales, theta_list, bias, learning_rate):
        """
        A function to update the values of thetas according to RMSE
        """
        total_samples = len(global_sales)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0
        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis += theta * features[sample_number][feature_index]        #hypothesis calculation
                feature_index+=1
            hypothesis = hypothesis - global_sales[sample_number]           #bias derivative calculation
            bias_derivative += hypothesis           #bias derivative calculation

            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += hypothesis*features[sample_number][feature_index]        
        
        current_cost = self.rmse_cost_function(features, global_sales,theta_list, bias)
        bias -= (bias_derivative/(total_samples * 2 * current_cost)) * learning_rate            #calculating gradient descent

        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/(total_samples * 2 * current_cost)) * learning_rate   #calculating gradient descent

        return bias, theta_list

    def rmse_cost_function(Self,X,y,theta_list, bias):
        """
        A function to calcuate the cost based on RMSE
        """
        total_samples = len(y)
        rmse = 0
        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis = hypothesis + theta * X[sample_number][feature_index]       #hypothesis calculation
                feature_index+=1
            
            rmse += (hypothesis-y[sample_number])**2

        return math.pow(rmse/(2*total_samples), 0.5)            #cost calculation

    def mae_update_thetas(self, features, global_sales, theta_list, bias, learning_rate):
        """
        A function to update the values of thetas according to MAE
        """
        total_samples = len(global_sales)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0

        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis += theta * features[sample_number][feature_index]            #hypothesis calculation
                feature_index+=1
            if(hypothesis>global_sales[sample_number]):
                hypothesis = 1
            elif(hypothesis<global_sales[sample_number]):
                hypothesis = -1
            else:
                hypothesis = 0

            bias_derivative += hypothesis

            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += hypothesis*features[sample_number][feature_index]

        bias -= (bias_derivative/total_samples) * learning_rate

        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/total_samples) * learning_rate            #gradient descent

        return bias, theta_list

    def mae_cost_function(Self,X,y,theta_list, bias):
        """
        A function to calcuate the cost based on MAE
        """
        total_samples = len(y)
        mae = 0
        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis = hypothesis + theta * X[sample_number][feature_index]           #hypothesis calculation
                feature_index+=1
            
            mae += abs(hypothesis-y[sample_number])

        return mae/total_samples            #loss

    def fit(self, X, y, test_X, test_y):
        """
        A function that takes X to train the model. The test data is passed in order to 
        calcuate the validation error, it can be skipped and removed.
        """

        iterate = 500
        self.mae_theta_list = [0]*len(X[0])
        self.mae_bias = 0
        mae_cost_history = []
        mae_validate_cost_history = []
        mae_min = 2147483647
        mae_good_iter = 0
        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")

            self.mae_bias, self.mae_theta_list = self.mae_update_thetas(X, y, self.mae_theta_list, self.mae_bias,self.training_rate)

            mae_cost = self.mae_cost_function(X, y, self.mae_theta_list, self.mae_bias)
            mae_cost_history.append(mae_cost)
            mae_validate_cost = self.mae_cost_function(test_X, test_y,self.mae_theta_list, self.mae_bias)
            mae_validate_cost_history.append(mae_validate_cost)
            if(mae_cost<mae_min):
                mae_good_iter = i+1
                mae_min=mae_cost

        self.FINAL_MAE_TRAIN_LOSS.append(mae_cost_history[-1])
        self.FINAL_MAE_VALIDATE_LOSS.append(mae_validate_cost_history[-1])

        plt.plot(list(range(iterate)), mae_cost_history)
        plt.plot(list(range(iterate)), mae_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss MAE")
        plt.show()


        self.rmse_theta_list = [0]*len(X[0])
        self.rmse_bias = 0

        rmse_cost_history = []
        rmse_validate_cost_history = []
        rmse_min = 2147483647
        rmse_good_iter = 0
        greater = 5
        for i in range(iterate):            #Iterations to be performed on the dataset
            if(i%100==0):
                print(i," iterations")
            self.rmse_bias, self.rmse_theta_list = self.rmse_update_thetas(X, y, self.rmse_theta_list, self.rmse_bias,self.training_rate)

            rmse_cost = self.rmse_cost_function(X, y, self.rmse_theta_list, self.rmse_bias)
            rmse_cost_history.append(rmse_cost)
            rmse_validate_cost = self.rmse_cost_function(test_X, test_y,self.rmse_theta_list, self.rmse_bias)
            rmse_validate_cost_history.append(rmse_validate_cost)
            if(rmse_cost<rmse_min):
                rmse_good_iter=i+1
                rmse_min=rmse_cost
        

        self.FINAL_RMSE_TRAIN_LOSS.append(rmse_cost_history[-1])
        self.FINAL_RMSE_VALIDATE_LOSS.append(rmse_validate_cost_history[-1])

        plt.plot(list(range(iterate)), rmse_cost_history)
        plt.plot(list(range(iterate)), rmse_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss RMSE")
        plt.show()


        print("FINAL_MAE_TRAIN_LOSS\n",self.FINAL_MAE_TRAIN_LOSS)
        print("FINAL_MAE_VALIDATE_LOSS\n",self.FINAL_MAE_VALIDATE_LOSS)
        print("FINAL_RMSE_TRAIN_LOSS\n",self.FINAL_RMSE_TRAIN_LOSS)
        print("FINAL_RMSE_VALIDATE_LOSS\n",self.FINAL_RMSE_VALIDATE_LOSS)
        return self




    def predict(self, X):
        """
        A function that takes test_X as input and returns prediction as the output for both RMSE and MAE.
        """       
        mae_predicted = []
        for i in range(len(X)):
            hypothesis = 0
            hypothesis = hypothesis+self.mae_bias
            for j in range(len(self.mae_theta_list)):
                hypothesis += X[i][j]*self.mae_theta_list[j]

            mae_predicted.append(hypothesis) 

        rmse_predicted = []
        for i in range(len(X)):
            hypothesis = 0
            hypothesis = hypothesis+self.rmse_bias
            for j in range(len(self.rmse_theta_list)):
                hypothesis += X[i][j]*self.rmse_theta_list[j]

            rmse_predicted.append(hypothesis) 

        return mae_predicted, rmse_predicted