
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold



df = pd.read_csv('data-magnetic phase.csv')

print(df)



properties = list(df.columns.values)
properties.remove('Name')
properties.remove('Magnetic phase')


print (properties)
X =df[properties]
X = pd.DataFrame(X)

y = df['Magnetic phase']
y = pd.DataFrame(y)
print(X)
print(y)

k_folds = KFold(n_splits = 7, shuffle = True, random_state = 32)
for train_index, test_index in k_folds.split(X):
 X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

N=120

model =keras.Sequential([keras.layers.Flatten(input_shape=(22,)),
                         keras.layers.Dense(N, activation=tf.nn.selu), 
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu), 
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.selu),
                         keras.layers.Dense(N, activation=tf.nn.sigmoid),
                         keras.layers.Dense(N, activation=tf.nn.sigmoid),
                         keras.layers.Dense(1)])


model.compile(optimizer='adamax', loss=tf.keras.losses.mse, metrics=['mae'])
data=tf.keras.utils.normalize(y_train)
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

hist = model.fit(X_train, y_train, epochs=120, batch_size=1)
test_preds = model.predict(X_test)
train_preds = model.predict(X_train) 


df = pd.read_csv('data-p-magneticphase.csv')
print(df)

preproperties = list(df.columns.values)
preproperties.remove('Name')
 
print (preproperties)

z=df[preproperties]
print(z)

prediction=model.predict(z)
print(prediction)

prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('magnetic phasepred_NN.csv')
