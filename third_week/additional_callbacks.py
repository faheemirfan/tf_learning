import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



dataset_diabetes = load_diabetes()
data = dataset_diabetes['data']
targets = dataset_diabetes['target']
targets = (targets-targets.mean())/targets.std()

train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)

def return_model():
    model  = Sequential([
        Dense(128,activation='relu',input_shape=(train_data.shape[1],)),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(1)
    ])
    return model

model = return_model()

model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])

def lr_function(epoch,lr):
    if epoch%2==0:
        return lr
    else:
        return lr+epoch/1000
    
callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.2,verbose=1)

history = model.fit(train_data,train_targets,epochs=1000,callbacks=[callback],verbose=False)
    

    
