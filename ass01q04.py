import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

sol = pd.read_excel("Husskonen_Solubility_Features.xlsx",verbose=False)
# print(sol.shape)
# print(sol.columns)

t = sol ['LogS.M.']. values

fig , ax = plt . subplots ( figsize =(4 ,4) )
ax . hist (t , bins =40 , facecolor ='m')
ax . set_title (" Histogram of Log Solubility ", fontsize =14)
X = sol.iloc[:, 5:].select_dtypes(include=[np.number])

# print ( X . shape )
# print ( t . shape )

# Drop rows with NaN
X = X.dropna()
t = t[:len(X)]

# Split data into training and test sets
#

X_train , X_test , t_train , t_test = train_test_split (X , t ,test_size =0.3, random_state=42)

# Regularized regression
#
gamma = 2.3
N , p = X . shape
w = np . linalg . inv ( X_train.to_numpy() . T @ X_train.to_numpy() + gamma * np . identity ( p ) ) @ X_train.to_numpy() . T @ t_train
th_train = X_train.to_numpy() @ w
th_test = X_test.to_numpy() @ w 

# Plot training and test predictions
#
fig , ax = plt . subplots ( nrows =1 , ncols =2 , figsize =(10 ,4) )
ax [0]. scatter ( t_train , th_train , c ='m', s =3)
ax[1].scatter(t_test, th_test,c='r',s=3)
plt.show()


alphas = np.logspace(-4, 1, 20)   
mse_test = []
nonzeros = []

for a in alphas:
    model = Lasso(alpha=a, max_iter=10000)
    model.fit(X_train, t_train)
    preds = model.predict(X_test)
    mse_test.append(mean_squared_error(t_test, preds))
    nonzeros.append(np.sum(model.coef_ != 0))


# Plot test error and number of non-zero coefficients vs alpha
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogx(alphas, mse_test, marker='o')
ax[0].set_xlabel("alpha (L1 regularization strength)")
ax[0].set_ylabel("Test MSE")
ax[0].set_title("Lasso Test Error vs Regularization")

ax[1].semilogx(alphas, nonzeros, marker='o', color='r')
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("Non-zero Coefficients")
ax[1].set_title("Model Sparsity vs Regularization")
plt.tight_layout()
plt.show()


best_alpha = alphas[np.argmin(mse_test)]
best_model = Lasso(alpha=best_alpha, max_iter=10000)
best_model.fit(X_train, t_train)

# Find top-10 features by absolute weight
coef_series = pd.Series(best_model.coef_, index=X_train.columns)
top10 = coef_series.abs().sort_values(ascending=False).head(10)
print("Top 10 features predicting solubility:")
print(top10)


X_train_top = X_train[top10.index]
X_test_top  = X_test[top10.index]

best_model.fit(X_train_top, t_train)
pred_top = best_model.predict(X_test_top)

from sklearn.metrics import r2_score
print("R² with all features (ridge):", r2_score(t_test, th_test))
print("R² with top-10 Lasso features:", r2_score(t_test, pred_top))
