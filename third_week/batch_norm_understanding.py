import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
import numpy as np


class my_callbacks(Callback):
    def on_train_begin(self,logs=None):
        print(f'Training started..........')



dataset_diabetes = load_diabetes()
data = dataset_diabetes['data']
targets = dataset_diabetes['target']

targets = (targets-targets.mean(axis=0))/targets.std()


train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)


def model_return():
    model = Sequential([
        Dense(64,activation='relu',input_shape=(train_data.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256,activation='relu')
        
    ])
    return model

model = model_return()

model.add(tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.95,
    epsilon=0.001,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05),
    gamma_initializer=tf.keras.initializers.constant(value=0.9)
))
model.add(Dense(1))

model.compile(optimizer='adam',loss="mse",metrics=['mae'])
history = model.fit(train_data,train_targets,validation_split=0.15,epochs=100,verbose=False,callbacks=[my_callbacks()])


result_df = pd.DataFrame(history.history)
epochs = np.arange(len(result_df))

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax.plot(epochs,result_df['loss'],label="training_loss")
ax.plot(epochs,result_df["val_loss"],label="validation_loss")
ax.set_title("epoch vs loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()


ax = fig.add_subplot(122)
ax.plot(epochs,result_df["mae"],label="training_error")
ax.plot(epochs,result_df["val_mae"],label="validation_error")
ax.set_title("epoch vs error")
ax.set_xlabel("epoch")
ax.set_ylabel("error")
ax.legend()




plt.show()