# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# data_creation


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
df = pd.read_csv('data-cell.csv')

print(df)



properties = list(df.columns.values)
properties.remove('Name')
properties.remove('Cellparameter')

print (properties)





X =df[properties]
y =df['Cellparameter']

print(X)
print(y)




from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_test, y_test = train_test_split(X_test, y_test, test_size=0.2)



print (X_train)
print (y_train)



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


cv = KFold(n_splits=10, random_state=1, shuffle=True)



N=60

model =keras.Sequential([keras.layers.Flatten(input_shape=(14,)),
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
keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

hist = model.fit(X_train, y_train, epochs=60, batch_size=10)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_acc)

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

import matplotlib.pyplot as plt

plt.figure(1)
plt.suptitle('Blue is lose function for training, orange validation', fontsize=14, fontweight='bold')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.figure(2)
plt.suptitle('red are orginal data, black are predcited data', fontsize=14, fontweight='bold')

plt.xlabel('Data number')
plt.ylabel('Data')
testing=model.predict(X_test)
plt.plot(testing, 'o', color='black')

y_test_plot=y_test.to_numpy()
plt.plot(y_test_plot, '^', color='red')
plt.show()

plt.figure(3)
plt.plot(y_test_plot,testing, '^', color='red')
plt.plot([2.5,4],[2.5,4])
plt.show()
plt.figure(4)
plt.suptitle('red are orginal data, black are predcited data', fontsize=14, fontweight='bold')

plt.xlabel('Data number')
plt.ylabel('Data')
train=model.predict(X_train)
plt.plot(train, 'o', color='black')

y_train_plot=y_train.to_numpy()
plt.plot(y_train_plot, '^', color='red')
plt.show()


plt.figure(5)
plt.plot(y_train_plot,train, 'o', color='gray')
plt.plot(y_test_plot,testing, '^', color='red')
plt.plot([2.5,4],[2.5,4])
x=np.arange(2.6,5)
y=np.arange(2.6,5)
y=x
y1 = 0.95*x 
y2 = 1.05*x
y3 = 0.90*x 
y4 = 1.10*x
plt.plot(x,y,  color='black')
plt.plot(x,y1,linestyle='dashed', color='red')
plt.plot(x,y2,linestyle='dashed', color='red')
plt.plot(x,y3,linestyle='dashed', color='blue')
plt.plot(x,y4,linestyle='dashed', color='blue')
plt.title('Cell parameter')
plt.xlabel('data of DFT')
plt.ylabel('data of machine')
plt.show()



from keras.models import load_model
model.save('my_modelkh10.h5')
model=load_model('my_modelselukh10.h5')


df = pd.read_csv('data-pcell.csv')
print(df)

preproperties = list(df.columns.values)
preproperties.remove('Name')
 
print (preproperties)

z=df[preproperties]
print(z)

prediction=model.predict(z)
print(prediction)

#numpy.savetxt('C:/Desktop/hubbard u prediction/prediction.csv', delimiter=',')
prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('predictionkh10.csv')
