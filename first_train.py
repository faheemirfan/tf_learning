import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D, Dense
import numpy as np


import matplotlib.pyplot as plt
import pandas as pd


model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10,activation='softmax')
])
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt,
              loss = 'sparse_categorical_crossentropy',
              metrics = [acc,mae])


dataset =  tf.keras.datasets.fashion_mnist
(train_images,train_labes),(test_images,test_labels) =  dataset.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

history = model.fit(train_images[...,np.newaxis],train_labes,epochs=8,batch_size=256,verbose=2)


df = pd.DataFrame(history.history)

loss_plt = df.plot(y="loss",title="loss  vs epoch",legend=False)
loss_plt.set(xlabel="Epoch",ylabel="loss")
plt.savefig("loss.png")

loss_plt = df.plot(y="sparse_categorical_accuracy",title="sparse_categorical_accuracy  vs epoch",legend=False)
loss_plt.set(xlabel="Epoch",ylabel="sparse_categorical_accuracy")
plt.savefig("sparse_categorical_accuracy.png")

loss_plt = df.plot(y="mean_absolute_error",title="mean_absolute_error  vs epoch",legend=False)
loss_plt.set(xlabel="Epoch",ylabel="mean_absolute_error")
plt.savefig("mean_absolute_error.png")