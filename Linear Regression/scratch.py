import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass


    def pre_process(self, dataset):

        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            """
            with open('abalone/Dataset.data') as input_file:
                lines = input_file.readlines()
                newLines = []
                for line in lines:
                    newLine = line.strip().split()
                    newLines.append( newLine )

            with open('AbalonDataset.csv', 'wb') as test_file:
                file_writer = csv.writer(test_file)
                file_writer.writerows(newLines)
            """
            dataframe = pd.read_csv('abalone/Dataset.csv')
            dataframe.insert(1, "M", 0) 
            dataframe.insert(2, "F", 0) 
            dataframe.insert(3, "I", 0) 
            dataframe['M'] = dataframe.apply(lambda row: 1 if(row.Sex == 'M') else 0, axis = 1)     #Binary Hot coding
            dataframe['F'] = dataframe.apply(lambda row: 1 if(row.Sex == 'F') else 0, axis = 1)
            dataframe['I'] = dataframe.apply(lambda row: 1 if(row.Sex == 'I') else 0, axis = 1)
            del dataframe['Sex']
            X = dataframe.iloc[:,[0,1,2,3,4,5,6,7,8,9]]             #spliting x and y
            y = dataframe['Rings']
            print(dataframe)
            dataframe = dataframe.sample(frac = 1) 
            s = np.array_split(dataframe, 3)
            #print(s[0])
            return s
            pass
        elif dataset == 1:
            dataframe = pd.read_csv("VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv")           #Reading csv
            dataframeuse = dataframe[['User_Score', 'Critic_Score', 'Global_Sales']]
            finaldataframe = dataframeuse.dropna()              #Removing nan and tbd values
            finaldataframe = finaldataframe[finaldataframe['User_Score']!='tbd']
            finaldataframe['User_Score'] = finaldataframe['User_Score'].astype(float)
            finaldataframe = finaldataframe[finaldataframe['Global_Sales'] < 20]            #removing outliers


            user_column = finaldataframe['User_Score']  
            max_val = user_column.max()
            #print(max_val)
            min_val = user_column.min()
            #print(min_val)

            finaldataframe['User_Score'] = finaldataframe['User_Score'].apply(lambda x: (((x - min_val)/(max_val - min_val) )*(10-1))+1)        #scacling the scores to 0 - 10
            #finaldataframe=finaldataframe.sort_values(by=['User_Score'])
            #plt.scatter(finaldataframe['User_Score'], finaldataframe['Global_Sales'])
            #plt.show()

            critic_column = finaldataframe['Critic_Score']
            max_val = critic_column.max()
            #print(max_val)
            min_val = critic_column.min()
            #print(min_val)

            finaldataframe['Critic_Score'] = finaldataframe['Critic_Score'].apply(lambda x: (((x - min_val)/(max_val - min_val) )*(10-1))+1)
            #finaldataframe=finaldataframe.sort_values(by=['Critic_Score'])
            #plt.scatter(finaldataframe['Critic_Score'], finaldataframe['Global_Sales'])
            #plt.show()

            finaldataframe=finaldataframe.sort_values(by=['Global_Sales'])
            finaldataframe = finaldataframe.sample(frac = 1) 
            s = np.array_split(finaldataframe, 3)               #splitting dataset into 3 splits
            #print(s[0])
            return s
            X = finaldataframe[['User_Score', 'Critic_Score']]
            y = finaldataframe['Global_Sales']

            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            dataframe = pd.read_csv("data_banknote_authentication.txt")     #reading csv
            dataframe.columns = ["variance","skewness","curtosis","entropy","class"]
            print(dataframe)
            print(dataframe.describe())
            class_1 = dataframe['class'].sum()
            class_2 = len(dataframe)-class_1
            v = [class_1, class_2]
            plt.pie(v, labels = ['Class -> 1 ('+str(class_1)+')', 'Class -> 0 ('+str(class_2)+')'],autopct='%1.1f%%')       #pie chart of class distribution
            plt.show()
            plt.show()
            plt.subplot(221)
            plt.scatter('variance', 'class', data = dataframe)
            plt.xlabel('Variance')
            plt.ylabel('Class')
            plt.subplot(222)
            plt.scatter('skewness', 'class', data = dataframe)
            plt.xlabel('skewness')
            plt.ylabel('class')                                                       #PLOTS
            plt.subplot(223)
            plt.scatter('curtosis', 'class', data = dataframe)
            plt.xlabel('curtosis')
            plt.ylabel('class')
            plt.subplot(224)
            plt.scatter('entropy', 'class', data = dataframe)
            plt.xlabel('entropy')
            plt.ylabel('class')
            plt.show()
            dataframe = dataframe[dataframe['curtosis'] <= 10]              #removing outliers
            dataframe = dataframe[dataframe['entropy'] >= -7]
            #print(len(dataframe))
            dataframe.boxplot()
            plt.show()
            dataframe = dataframe.sample(frac = 1) 
            s = np.array_split(dataframe, 10)           #spliting data into train-test-val
            train_split = pd.concat(s[:7])
            validate_split = s[7]
            test_split = pd.concat(s[8:])

            print(train_split)
            print(validate_split)
            print(test_split)

            return train_split, validate_split,test_split
            pass

        X = X.to_numpy()
        y = y.to_numpy()
        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
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
        total_samples = len(global_sales)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0
        #print("max=",max(global_sales))
        #print("min=",min(global_sales))

        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis += theta * features[sample_number][feature_index]        #hypothesis calculation
                feature_index+=1
            #print("(",hypothesis,",",global_sales[sample_number], end=") ")
            hypothesis = hypothesis - global_sales[sample_number]           #bias derivative calculation
            #print(hypothesis, end=' ')
            bias_derivative += hypothesis           #bias derivative calculation

            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += hypothesis*features[sample_number][feature_index]        

        #print("\n",bias,"bias_derivative =", bias_derivative, total_samples, learning_rate)
        
        current_cost = self.rmse_cost_function(features, global_sales,theta_list, bias)
        bias -= (bias_derivative/(total_samples * 2 * current_cost)) * learning_rate            #calculating gradient descent
        #print(bias)
        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/(total_samples * 2 * current_cost)) * learning_rate   #calculating gradient descent

        return bias, theta_list

    def rmse_cost_function(Self,X,y,theta_list, bias):
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
        total_samples = len(global_sales)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0
        #print("max=",max(global_sales))
        #print("min=",min(global_sales))

        for sample_number in range(total_samples):
            hypothesis = 0
            hypothesis = hypothesis+bias
            feature_index = 0
            for theta in theta_list:
                hypothesis += theta * features[sample_number][feature_index]            #hypothesis calculation
                feature_index+=1
            #print("(",hypothesis,",",global_sales[sample_number], end=") ")
            if(hypothesis>global_sales[sample_number]):
                hypothesis = 1
            elif(hypothesis<global_sales[sample_number]):
                hypothesis = -1
            else:
                hypothesis = 0

            #print(hypothesis, end=' ')
            bias_derivative += hypothesis

            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += hypothesis*features[sample_number][feature_index]

        #print("\n",bias,"bias_derivative =", bias_derivative, total_samples, learning_rate)
        bias -= (bias_derivative/total_samples) * learning_rate
        #print(bias)
        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/total_samples) * learning_rate            #gradient descent

        return bias, theta_list

    def mae_cost_function(Self,X,y,theta_list, bias):

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

        iterate = 500
        self.mae_theta_list = [0]*len(X[0])
        self.mae_bias = 0
        #print(self.mae_theta_list, self.mae_bias, self.training_rate)
        mae_cost_history = []
        mae_validate_cost_history = []
        mae_min = 2147483647
        mae_good_iter = 0
        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")

            self.mae_bias, self.mae_theta_list = self.mae_update_thetas(X, y, self.mae_theta_list, self.mae_bias,self.training_rate)
            #print("ok",self.mae_theta_list, self.mae_bias)
            mae_cost = self.mae_cost_function(X, y, self.mae_theta_list, self.mae_bias)
            mae_cost_history.append(mae_cost)
            mae_validate_cost = self.mae_cost_function(test_X, test_y,self.mae_theta_list, self.mae_bias)
            mae_validate_cost_history.append(mae_validate_cost)
            if(mae_cost<mae_min):
                mae_good_iter = i+1
                mae_min=mae_cost
        self.FINAL_MAE_TRAIN_LOSS.append(mae_cost_history[-1])
        self.FINAL_MAE_VALIDATE_LOSS.append(mae_validate_cost_history[-1])
        #print("MAE COSTS\nTRAINING\n",mae_cost_history)
        #print("VALIDATE\n",mae_validate_cost_history)
        plt.plot(list(range(iterate)), mae_cost_history)
        plt.plot(list(range(iterate)), mae_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss MAE")
        plt.show()

        #print("iter = ",mae_good_iter)


        self.rmse_theta_list = [0]*len(X[0])
        self.rmse_bias = 0
        #print(self.rmse_theta_list, self.rmse_bias)
        rmse_cost_history = []
        rmse_validate_cost_history = []
        rmse_min = 2147483647
        rmse_good_iter = 0
        greater = 5
        for i in range(iterate):            #Iterations to be performed on the dataset
            if(i%100==0):
                print(i," iterations")
            self.rmse_bias, self.rmse_theta_list = self.rmse_update_thetas(X, y, self.rmse_theta_list, self.rmse_bias,self.training_rate)
            #print("ok",self.rmse_theta_list, self.rmse_bias)
            rmse_cost = self.rmse_cost_function(X, y, self.rmse_theta_list, self.rmse_bias)
            rmse_cost_history.append(rmse_cost)
            rmse_validate_cost = self.rmse_cost_function(test_X, test_y,self.rmse_theta_list, self.rmse_bias)
            rmse_validate_cost_history.append(rmse_validate_cost)
            if(rmse_cost<rmse_min):
                rmse_good_iter=i+1
                rmse_min=rmse_cost
        

        self.FINAL_RMSE_TRAIN_LOSS.append(rmse_cost_history[-1])
        self.FINAL_RMSE_VALIDATE_LOSS.append(rmse_validate_cost_history[-1])
        #print("RMSE COSTS\nTRAINING\n",rmse_cost_history)
        #print("VALIDATE\n",rmse_validate_cost_history)
        #print("iter = ",rmse_good_iter)

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




    def predict(self, X):       #predict y using theta and bias calculated while training

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


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """
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
        total_samples = len(X)
        theta_derivative = [0]*len(theta_list)
        bias_derivative = 0
        #print("max=",max(global_sales))
        #print("min=",min(global_sales))
        for i in range(total_samples):                      #update the thetas and bias by gradient descent
            hypothesis = 0
            hypothesis = hypothesis+bias
            #print(X[i])
            #print(theta_list)
            hypothesis += np.matmul(X[i], np.array(theta_list).T)
            #print(hypothesis)
            sigmoidhypothesis = 1./(1.+np.exp(-hypothesis))
            #print(sigmoidhypothesis, y[i])
            #print("(",hypothesis,",",global_sales[sample_number], end=") ")
            sigmoidhypothesis = sigmoidhypothesis - y[i]
            #print(hypothesis, end=' ')
            bias_derivative += sigmoidhypothesis
            #print("bi=", bias_derivative)

            feature_index=0
            for feature_index in range(len(theta_list)):
                theta_derivative[feature_index] += sigmoidhypothesis*X[i][feature_index]

        #print("\n",bias,"bias_derivative =", bias_derivative, total_samples, learning_rate)
        
        bias -= (bias_derivative/total_samples) * learning_rate
        #print(bias)
        for j in range(len(theta_list)):
            theta_list[j] -= (theta_derivative[j]/total_samples) * learning_rate

        #print("re=", bias, theta_list)

        return bias, theta_list



    def cost_function(self, X, y, theta_list, bias):
        total_samples = len(y)
        loss = 0
        #print("t=", theta_list)
        #print("bias=", bias)
        for i in range(total_samples):
            hypothesis = bias
            hypothesis += np.matmul(X[i], np.array(theta_list).T)
            
            de = 1.0 + np.exp(-hypothesis)
            sigmoidhypothesis = 1.0/de
            #print(sigmoidhypothesis)
            loss += (y[i]*np.log(sigmoidhypothesis)) + ((1-y[i])*(np.log(1 - sigmoidhypothesis)))

        return -1 * (loss/total_samples)                #loss calculation


    def fit(self, X, y, X_validate, y_validate):

        
        iterate = 800
        
        self.SGD_theta_list = [0]*len(X[0])
        self.SGD_bias = 0
        #print(self.SGD_theta_list, self.SGD_bias, self.training_rate)
        SGD_cost_history = []
        SGD_validate_cost_history = []

        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")
            selection = random.randint(0, len(X)-1)         #selecting one random row for SGD
            #print(selection)
            temp_X = []
            temp_X.append(X[selection])
            temp_y = []
            temp_y.append(y[selection])
            self.SGD_bias, self.SGD_theta_list = self.update_thetas(np.array(temp_X), np.array(temp_y), self.SGD_theta_list, self.SGD_bias,self.training_rate)
            #print("ok",self.SGD_theta_list, self.SGD_bias)
            SGD_cost = self.cost_function(X, y, self.SGD_theta_list, self.SGD_bias)
            SGD_cost_history.append(SGD_cost)
            SGD_validate_cost = self.cost_function(X_validate, y_validate,self.SGD_theta_list, self.SGD_bias)
            SGD_validate_cost_history.append(SGD_validate_cost)

        self.FINAL_SGD_TRAIN_LOSS.append(SGD_cost_history[-1])
        self.FINAL_SGD_VALIDATE_LOSS.append(SGD_validate_cost_history[-1])
        #print("SGD COSTS\nTRAINING\n",SGD_cost_history)
        #print("VALIDATE\n",SGD_validate_cost_history)
        plt.plot(list(range(iterate)), SGD_cost_history)
        plt.plot(list(range(iterate)), SGD_validate_cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss SGD")
        plt.show()
        
        
        self.BGD_theta_list = [0]*len(X[0])
        self.BGD_bias = 0
        #print(self.BGD_theta_list, self.BGD_bias, self.training_rate)
        BGD_cost_history = []
        BGD_validate_cost_history = []

        for i in range(iterate):
            if(i%100==0):
                print(i," iterations")
            selection = random.randint(0, len(X)-1)
            #print(selection)
            
            self.BGD_bias, self.BGD_theta_list = self.update_thetas(X, y, self.BGD_theta_list, self.BGD_bias,self.training_rate)
            #print("ok",self.BGD_theta_list, self.BGD_bias)
            BGD_cost = self.cost_function(X, y, self.BGD_theta_list, self.BGD_bias)
            BGD_cost_history.append(BGD_cost)
            BGD_validate_cost = self.cost_function(X_validate, y_validate,self.BGD_theta_list, self.BGD_bias)
            BGD_validate_cost_history.append(BGD_validate_cost)

        self.FINAL_BGD_TRAIN_LOSS.append(BGD_cost_history[-1])
        self.FINAL_BGD_VALIDATE_LOSS.append(BGD_validate_cost_history[-1])
        #print("BGD COSTS\nTRAINING\n",BGD_cost_history)
        #print("VALIDATE\n",BGD_validate_cost_history)
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

    def predict(self, X):               #predict the values on the basis of theta and bias calculated while training

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

    def accuracy(self,predicted, original):             #returns the accuracy on the basis of predicted and original values
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
