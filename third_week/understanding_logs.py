import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



dataset_diabetes = load_diabetes()
data = dataset_diabetes['data']
targets = dataset_diabetes['target']
targets = (targets-targets.mean())/targets.std()

train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)


lr_tuple = [(4,0.03),(7,0.02),(11,0.005),(15,0.007)]

def get_lrn(epoch,learn_rate):
    epoch_index = [i for i in range(len(lr_tuple)) if (lr_tuple[i][0]==epoch)]
    if(len(epoch_index)>0):
        return lr_tuple[epoch_index[0]][1]
    else:
        return learn_rate
    
class my_callback(tf.keras.callbacks.Callback):
    def __init__(self,learn_rate):
        super(my_callback,self).__init__()
        self.learn_rate = learn_rate

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer,'lr'):
            raise ValueError("Erorr no attribute of learning rate...")
        curr_lr = tf.keras.backend.get_value(self.model.optimizer.lr)

        scheduled_lr = self.learn_rate(epoch,curr_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr,scheduled_lr)
        # print('Learning rate for epoch : {} is {:7.3f}'.format(epoch,scheduled_lr))

def return_model():
    model =Sequential([
        Dense(128,activation='relu',input_shape=(train_data.shape[1],)),
        Dense(64,activation='relu'),
        BatchNormalization(),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(1)
    ])
    return model

model = return_model()
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse'])
model.fit(train_data,train_targets,epochs=100,batch_size=100,validation_split=0.15,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],verbose=2)