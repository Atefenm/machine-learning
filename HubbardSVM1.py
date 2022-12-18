
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVR


df = pd.read_csv('data-hubbard.csv')
df.shape
df.head()

print(df)



properties = list(df.columns.values)
properties.remove('Name')
properties.remove('Uh')

print (properties)
X =df[properties]
X = pd.DataFrame(X)

y = df['Uh']
y = pd.DataFrame(y)
print(X)
print(y)
k_folds = KFold(n_splits = 7, shuffle = True, random_state = 42)
for train_index, test_index in k_folds.split(X):
 X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

svr = SVR(kernel='linear')


svr.fit(X_train, y_train)


test_preds = svr.predict(X_test)
train_preds = svr.predict(X_train) 


plt.figure(1)
plt.suptitle('red are original data, black are predcited data', fontsize=14, fontweight='bold')

plt.xlabel('Data number')
plt.ylabel('Data')
plt.plot(test_preds, 'o', color='black')

y_test_plot=y_test.to_numpy()
plt.plot(y_test_plot, '^', color='red')
plt.show()


plt.figure(2)
plt.suptitle('red are original data, black are predcited data', fontsize=14, fontweight='bold')

plt.xlabel('Data number')
plt.ylabel('Data')
plt.plot(train_preds, 'o', color='black')

y_train_plot=y_train.to_numpy()
plt.plot(y_train_plot, '^', color='red')
plt.show()

plt.figure(3)
plt.plot(y_train,train_preds, 'o', color='gray')
plt.plot(y_test,test_preds, '^', color='red')
plt.plot([0,8],[0,8])
x=np.arange(0,8)
y=x
y1 = 0.90*x 
y2 = 1.10*x
y3 = 0.80*x 
y4 = 1.20*x
plt.plot(x,y,  color='black')
plt.plot(x,y1,linestyle='dashed', color='red')
plt.plot(x,y2,linestyle='dashed', color='red')
plt.plot(x,y3,linestyle='dashed', color='blue')
plt.plot(x,y4,linestyle='dashed', color='blue')
plt.title('Hubbard')
plt.xlabel('data of DFT')
plt.ylabel('data of machine')
plt.show()


df = pd.read_csv('data-phubbard.csv')
print(df)

preproperties = list(df.columns.values)
preproperties.remove('Name')
 
print (preproperties)

z=df[preproperties]
print(z)

prediction=svr.predict(z)
print(prediction)


prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('Hubbardpred_SVR.csv')
