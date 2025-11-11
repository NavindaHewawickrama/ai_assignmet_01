import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso



#explore the plot commands
diabetes = datasets.load_diabetes()
X = diabetes.data
t = diabetes.target

#inspect sizes

NumData, NumFeatures = X.shape
print ( NumData , NumFeatures ) # 442 X 10
print ( t . shape ) # 442

#plot and save

fig , ax = plt.subplots( nrows =1 , ncols =2 , figsize =(12 , 4) )
ax [0].hist (t , bins =40)
ax [1].scatter ( X [: ,6] , X [: ,7] , c ='m', s =3)
ax [1].grid ( True )
plt.tight_layout ()
plt.savefig (" DiabetesTargetAndTwoInputs .jpg")
plt.show()

#linear regression code
lin = LinearRegression()
lin.fit(X, t)

th1 = lin . predict ( X )

# Pseudo - incerse solution to linear regression
w = np.linalg.inv(X.T@X)@X.T @t
th2 = X@w

# Plot predictions to check if they look the same !
#
fig , ax = plt . subplots ( nrows =1 , ncols =2 , figsize =(10 ,5) )
ax[0].scatter (t , th1 , c='c', s =3)
ax[1].scatter(t,th2,c='r',s=3)
plt.show()


#Tikhanov (quadratic) Regularizer
gamma = 0.5
wR = np.linalg.inv ( X . T @ X + gamma * np . identity ( NumFeatures ) ) @ X .T @ t


fig , ax = plt . subplots ( nrows =1 , ncols =2 , figsize =(8 ,4) )
ax [0].bar(np.arange(len(w)),w)
ax [1].bar(np.arange(len(wR)),wR)
plt . savefig (" LeastSquaresAndRegularizedWeights .jpg")
plt.show()

#numerical comparison between methods - prompt : give me a comparision between the two mthods
print("R² (sklearn):", r2_score(t, th1))
print("R² (pseudo-inverse):", r2_score(t, th2))
print("R² (regularized):", r2_score(t, X @ wR))


############## question 03 ####################
ll = Lasso ( alpha =0.2)
ll.fit (X , t )
###getting the weights of the lasso
lasW = ll.coef_
th_lasso = ll.predict ( X )

fig , ax = plt.subplots ( nrows =1 , ncols =3 , figsize =(15 ,4) )
ax[0].bar(np.arange (len ( w ) ) , w )
ax[1].bar(np.arange(len(wR)),wR)
ax[2].bar(np.arange(len(lasW)),lasW)
print("R² (Lasso):", r2_score(t, th_lasso))

plt . savefig (" solutions .png")
plt.show()