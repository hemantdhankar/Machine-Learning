import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

class gaussianNaiveBias:
	"""
	Gaussion Naive Bias implementation from scratch.
	"""

	"""
	Class attributes
	"""
	mean_A = []
	variance_A = []
	prior_probability_A = []
	y_dimension = 0

	def fit(self,X_train, Y_train):
		"""
		Function to train the model using train dataset		
		"""

		final_dataframe_A = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis=1)
		print(final_dataframe_A.head())


		"""
		This next line of code is hardcoded for a dataset having labels in range 0-9.
		What you need to do is to make a dictionary with keys equal to the labels and
		value equal to all the samples corresponding to that label. 
		Change the code accordingly and you are good to go.
		"""
		final_dataframe_A_dict = {0:final_dataframe_A[final_dataframe_A.iloc[:,-1]==0],1:final_dataframe_A[final_dataframe_A.iloc[:,-1]==1],2:final_dataframe_A[final_dataframe_A.iloc[:,-1]==2],3:final_dataframe_A[final_dataframe_A.iloc[:,-1]==3],4:final_dataframe_A[final_dataframe_A.iloc[:,-1]==4],5:final_dataframe_A[final_dataframe_A.iloc[:,-1]==5],6:final_dataframe_A[final_dataframe_A.iloc[:,-1]==6],7:final_dataframe_A[final_dataframe_A.iloc[:,-1]==7],8:final_dataframe_A[final_dataframe_A.iloc[:,-1]==8],9:final_dataframe_A[final_dataframe_A.iloc[:,-1]==9]}	


		mean_A = []
		variance_A = []
		prior_probability = []

		for i in range(self.y_dimension):
			mean_A.append(final_dataframe_A_dict[i].mean())
			variance_A.append(final_dataframe_A_dict[i].var())
			prior_probability.append(len(final_dataframe_A_dict[i])/len(X_train))

		self.mean_A = np.array(mean_A)
		self.variance_A = np.array(variance_A)
		self.prior_probability_A = np.array(prior_probability)

		return

	def predict(self,X_test):
		"""
		Function to use trained model for predictions on the test set.
		"""
		predictions = []
		X_test = np.array(X_test)

		for i in range(len(X_test)):
			maxval = [0]*self.y_dimension
			for k in range(self.y_dimension):
				score = math.log(self.prior_probability_A[k])
				for j in range(X_test[i].size):
					rootterm = math.sqrt(2*3.14*self.variance_A[k][j])
					squareterm =  math.pow((X_test[i][j]-self.mean_A[k][j]),2)
					conditionalProbability = ( 1 / rootterm) * ( math.exp( (-1*(squareterm)) / (2*self.variance_A[k][j]) ) )
					score+= math.log(conditionalProbability)
				maxval[k]=score
			maxscore = max(maxval)
			co = 0
			for value in maxval:
				if(value == maxscore):
					predictions.append(co)
				co+=1

		return predictions


	def accuracy(self,predicted, original):
		"""
		Returns the accuracy on the basis of predicted and original values.
		"""
		counter = 0
		for i in range(len(predicted)):
			if(predicted[i]==original[i]):
				counter+=1
		return counter/len(predicted)