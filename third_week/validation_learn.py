import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


diabetes_dataset = load_diabetes()
data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

#normalization
targets = (targets - targets.mean())/targets.std()

train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)


def get_model(wd,rate):
    model = Sequential([
        Dense(128,activation='relu',input_shape=(train_data.shape[1],),kernel_regularizer=tf.keras.regularizers.l2(wd)),
        Dropout(rate),
        Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(wd)),
        Dropout(rate),
        Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(wd)),
        Dropout(rate),
        Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(wd)),
        Dropout(rate),
        Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(wd)),
        Dropout(rate),
        Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(wd)),
        Dropout(rate),
        Dense(1)
    ])
    return model

model = get_model(0.001,0.0)
model.summary()

model.compile(optimizer="adam",loss="mse",metrics=["mae"])
history = model.fit(train_data,train_targets,epochs=100,validation_split=0.15,batch_size=64,verbose=0)
model.evaluate(test_data,test_targets,verbose=2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss vs Epoch")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["Train","validation"],loc="upper right")
plt.show()