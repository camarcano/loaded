# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:07:28 2022

@author: admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge, Lars, BayesianRidge
#import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as mae
import pickle
import warnings
warnings.filterwarnings("ignore")



cwd = 'C:/Users/admin/PycharmProjects/loaded/'
cwd = cwd.replace("\\", "/")


dataset = pd.read_csv(cwd + 'data.csv')
dataset = dataset.fillna(0)
dataset = dataset.drop(columns=['KN%', 'KNv'])
dataset = dataset.sample(frac=1).reset_index(drop=True)

X, y = dataset.iloc[:, 3:-1], dataset.iloc[:, -1]

nX = len(X.columns)

switch=1
feat = []

dropped_columns = []
shape = X.shape
for l in range(0,shape[1]):
    feat.append(l)
    
X = X.drop(X.columns[dropped_columns], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                test_size=0.10, random_state=1)


X_train_poly_scaled = (X_train - X_train.mean())/X_train.std()
X_train_poly_scaled = pd.DataFrame( \
        data = X_train_poly_scaled, columns = X_train.columns)

    
bbb = X_test.mean()
ccc = X_test.std()
X_test_poly_scaled = (X_test - bbb)/ ccc
X_test_poly_scaled = pd.DataFrame( \
        data = X_test_poly_scaled, columns = X_test.columns)

X_train = X_train_poly_scaled.copy()
X_test = X_test_poly_scaled.copy()

regressors = {
    "Lasso": Lasso(),
    "LinearRegression": LinearRegression(),
    "XGBRegressor": XGBRegressor(),
    "BayesianRidge": BayesianRidge(),
    "Lars": Lars(),
    "HuberRegressor": HuberRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    "LassoCV": LassoCV(),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "GaussianProcessRegressor": GaussianProcessRegressor(),
    "SVR": SVR(),
    "NuSVR": NuSVR(),
    "LinearSVR": LinearSVR(),
    "KernelRidge": KernelRidge(),
    "Ridge":Ridge(),
    "TheilSenRegressor": TheilSenRegressor(),
    "ARDRegression": ARDRegression(),
    "ElasticNet": ElasticNet(),
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
}


df_models = pd.DataFrame(columns=['model', 'run_time', 'rmse', 'rmse_cv'])

for key in regressors:

    print('*',key)

    start_time = time.time()

    regressor = regressors[key]
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = cross_val_score(model, 
                             X_train, 
                             y_train,
                             scoring="neg_mean_absolute_error", 
                             cv=10)

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60,2)),
           'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred))),
           'rmse_cv': round(np.mean(np.sqrt(-scores))),
           'mae': round(mae(y_test, y_pred),2)
    }

    df_models = df_models.append(row, ignore_index=True)

y_test = y_test.to_numpy()


y_test = y_test.ravel()

test = pd.DataFrame()
test['Predicted value'] = 0.0
test['Actual value'] = 0.0



for key in regressors:    
    regressor = regressors[key]
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fig= plt.figure(figsize=(16,8))
    if key!="LinearRegression" and key!="Ridge" and key!="GaussianProcessRegressor" and key!="KernelRidge":
        test['Actual value'] = y_test.tolist()
        test['Predicted value'] = y_pred.tolist()
    plt.subplot(1,2,1)
    plt.plot(test)
    plt.legend(['Predicted value', 'Actual value'])
    plt.title(key + "  - MAE: " + str(round(mae(test['Actual value'], test['Predicted value']),2)))
    plt.subplot(1,2,2)
    plt.scatter(test['Actual value'], test['Predicted value'])
    plt.title(key + "  - Corr: " + str(round(np.corrcoef(test['Actual value'], test['Predicted value'])[0][1],4)))
    plt.show()

stack = StackingCVRegressor(regressors=(HuberRegressor(), 
                            LassoCV(), LinearSVR()),
                            meta_regressor=ARDRegression(), cv=10,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=1)

stack.fit(X_train, y_train)

pred = stack.predict(X_test)
score = mae(y_test, pred)
print('Model: {0}, MAE: {1}'.format(type(stack).__name__, score))

vvv=np.corrcoef(y_test, pred)[0][1]
plt.scatter(y_test, pred)
plt.title('Model: {0}, MAE: {1}, CORR: {2}'.format(type(stack).__name__, round(score,2), round(vvv,4)))

z = np.polyfit(y_test, pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--", color='r')
plt.show()

hhh = pd.DataFrame()
hhh['a'] = y_test
hhh['b'] = pred
plt.plot(hhh)
plt.title('Model: {0}, MAE: {1}, CORR: {2}'.format(type(stack).__name__, round(score,2), round(vvv,4)))
plt.show()
print(vvv)


filename = 'model2.sav'
pickle.dump(model, open(filename, 'wb'))



cwd = 'C:/Users/admin/PycharmProjects/loaded/'
dataset = pd.read_csv(cwd + "2018-2022.csv")
dataset = dataset.fillna(0)
dataset = dataset.drop(columns=['KN%', 'KNv'])




# Feature, target arrays
X, y = dataset.iloc[:, 3:-1], dataset.iloc[:, -1]



X_train = X




X_train_poly_scaled = (X_train - X_train.mean())/X_train.std()
X_train_poly_scaled = pd.DataFrame( \
        data = X_train_poly_scaled, columns = X_train.columns)


X_train = X_train_poly_scaled.copy()

model_filename = "model2.sav"
stack = pickle.load(open(model_filename, 'rb'))


pred = stack.predict(X_train)

plt.show()

iii = pd.DataFrame()


iii['b'] = pred
iii.reset_index(drop=True, inplace=True)
plt.plot(iii[:285])
plt.show()


final = pd.DataFrame()

final['Name'] = dataset['Name']
final['IP_2023'] = pred[:285]
