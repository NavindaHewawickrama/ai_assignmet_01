# from sklearn.datasets import load_diabetes
# import matplotlib.pyplot as plt


######this is the code that i did in the beginning#####
#diabetes = load_diabetes()
#print(diabetes.target[:])
#print(diabetes)
#print(diabetes.data[0])
#print(diabetes.feature_names)
#print(diabetes.DESCR)



#plt.hist(diabetes.data, bins=10)

# plt.xlabel("Value")
# plt.ylabel("frequency")
# plt.title("Histogram of diabetes data")

#display the histogram
# plt.show()


#######assesment provided code##############

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#load the data
#inspec the data
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
