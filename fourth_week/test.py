import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from read_images import create_image_dataset



MAIN_FOLDER_PATH="/home/faheem/Learning/Tensorflow/repo/tf_learning/fourth_week/EuroSAT_RGB"
img_dataset_object = create_image_dataset(MAIN_FOLDER_PATH)
train_data, test_data, train_labels,test_labels = img_dataset_object.load_dataset()

def get_new_model(input_shape):
    model = Sequential([
        Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='SAME',name='conv_1',input_shape=(64,64,3)),
        Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='SAME',name='conv_2'),
        MaxPool2D(pool_size=(8,8),name='pool_1'),
        Flatten(),
        Dense(units=32,activation='relu',name='dense_1'),
        Dense(units=10,activation='softmax',name='dense_2')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

model = get_new_model((64,64,3))

model.summary()

def get_test_accuracy(model,x_test,y_test):
    test_loss,test_acc = model.evaluate(x=x_test,y=y_test,verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))

# get_test_accuracy(model,test_data,test_labels)



def get_checkpoint_every_epoch():
    return tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch:03d}',save_weights_only=True,save_freq='epoch')

def get_checkpoint_best_only():
    return tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_best_only/checkpoint',save_freq='epoch',monitor='val_accuracy',save_weights_only=True,save_best_only=True)

def get_early_stopping():
    return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)


checkpoint_every_epoch = get_checkpoint_every_epoch()
checkpoint_best_only = get_checkpoint_best_only()
early_stopping = get_early_stopping()


# callbacks = [checkpoint_every_epoch, checkpoint_best_only, early_stopping]
# model.fit(train_data,train_labels,epochs=50,validation_data=(test_data,test_labels),callbacks=callbacks,batch_size=12)

#
# return last instance of model
#
def get_model_last_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('checkpoints_every_epoch'))
    return model


#
#
#
def get_model_best_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('checkpoints_best_only'))
    return model

    
model_last_epoch = get_model_last_epoch(get_new_model((64,64,3)))
model_best_epoch = get_model_best_epoch(get_new_model((64,64,3)))

    
get_test_accuracy(model_last_epoch,test_data,test_labels)
get_test_accuracy(model_best_epoch,test_data,test_labels)