from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np


################# question 2 my code ################

# ##loading data
# example_dataset = datasets.load_breast_cancer()
# example_df = pd.DataFrame(example_dataset.data)
# example_df.columns = example_dataset.feature_names
# #print(example_df.head())

# example_npy_target_column = np.asarray(example_dataset.target)
# example_df["Breast_cancer"] = pd.Series(example_npy_target_column)

# #seperating predictors and response
# predictors = example_df.iloc[:,:-1]
# response = example_df.iloc[:,-1]
# #print(response.head())

# X_train, X_test, Y_train, Y_test = train_test_split(predictors,response,test_size=0.20)
# # print(X_train.shape)
# # print(X_test.shape)

# ##apply a normal linear regression
# linearreg = LinearRegression()
# linearreg.fit(X_train,Y_train)

# #predicting on test
# linearreg_prediction = linearreg.predict(X_test)

# #calculating mean squared error
# ridgeRegressor = Ridge(alpha = 0.5) # setting alpha 1
# ridgeRegressor.fit(X_train,Y_train)

# y_predicted_rifge = ridgeRegressor.predict(X_test)


# #calculatin mean squared error (mse)
# R_Squared = r2_score(y_predicted_rifge,Y_test)


# #putting together the coeffcient and their corresponding variable names
# coefficient_df = pd.DataFrame()
# coefficient_df["Column_Name"] = X_train.columns
# coefficient_df["Coefficiant_Value"] = pd.Series(linearreg.coef_)

# plt.rcParams["figure.figsize"] = (15,6)
# plt.bar(coefficient_df["Column_Name"],coefficient_df["Coefficiant_Value"])

# plt.show()

#https://youtu.be/21TgKhy1GY4?si=VIRUCPKSdJHJwbzy



#######################code given by the assignment######################

diabetes = datasets.load_diabetes()
X = diabetes.data
t = diabetes.target

NumData, NumFeatures = X.shape

#linear regression code
lin = LinearRegression()
lin.fit(X, t)

th1 = lin . predict ( X )

# Pseudo - incerse solution to linear regression
w = np.linalg.inv(X.T@X)@X.T @t
th2 = X@w

# Plot predictions to check if they look the same !
#
# fig , ax = plt . subplots ( nrows =1 , ncols =2 , figsize =(10 ,5) )
# ax[0].scatter (t , th1 , c='c', s =3)
# ax[1].scatter(t,th2,c='r',s=3)
# plt.show()


#Tikhanov (quadratic) Regularizer
gamma = 0.5
wR = np.linalg.inv ( X . T @ X + gamma * np . identity ( NumFeatures ) ) @ X .T @ t


fig , ax = plt . subplots ( nrows =1 , ncols =2 , figsize =(8 ,4) )
ax [0].bar(np.arange(len(w)),w)
ax [1].bar(np.arange(len(wR)),wR)
plt . savefig (" LeastSquaresAndRegularizedWeights .jpg")
plt.show()