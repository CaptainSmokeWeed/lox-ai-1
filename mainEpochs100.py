import keras
from keras import layers

import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#data upload
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#to categorial vctr
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
dummy_y = to_categorical(encoded_Y)

#dense model
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

#learn settings
model.compile(optimizer='adam',loss='categorical_crossentropy',
metrics=['accuracy'])

#it's learning my man
model.fit(X, dummy_y, epochs=100, batch_size=10,
validation_split=0.1)