
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold



df = pd.read_csv('data-cell.csv')


properties = list(df.columns.values)
properties.remove('Name')
properties.remove('Cellparameter')


X =df[properties]
X = pd.DataFrame(X)

y = df['Cellparameter']
y = pd.DataFrame(y)
print(X)
print(y)





k_folds = KFold(n_splits = 7, shuffle = True, random_state = 42)
for train_index, test_index in k_folds.split(X):
 X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

N=60

model =keras.Sequential([keras.layers.Flatten(input_shape=(15,)),
                         keras.layers.Dense(N, activation=tf.nn.selu), 
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(1)])


model.compile(optimizer='adamax', loss=tf.keras.losses.mse, metrics=['mae'])
data=tf.keras.utils.normalize(y_train)
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

hist = model.fit(X_train, y_train, epochs=60, batch_size=10)
test_preds = model.predict(X_test)
train_preds = model.predict(X_train)  


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
plt.plot([2.5,4.5],[2.5,4.5])
x=np.arange(2.6,4.5)
y=np.arange(2.6,4.5)
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
plt.title('Cell parameter')
plt.xlabel('data of DFT')
plt.ylabel('data of machine')
plt.show()


df = pd.read_csv('data-pcell.csv')


preproperties = list(df.columns.values)
preproperties.remove('Name')
 


z = df[preproperties]


prediction=model.predict(z)
print(prediction)


prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('cellpred_NN.csv')









