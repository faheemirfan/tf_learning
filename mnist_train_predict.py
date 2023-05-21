import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from IPython import embed

mnist_dataset = tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)  = mnist_dataset.load_data()

train_images_scale = train_images/255.0
train_images_scale = train_images_scale[...,np.newaxis]
test_images_scale = test_images/255.0
test_images_scale =  test_images_scale[...,np.newaxis]


def model_return(input_shape):
    model = Sequential([
        Conv2D(8,(3,3),activation='relu',padding='SAME',input_shape=input_shape),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')
    ])
    return model

def model_compile(model):
    opt = tf.keras.optimizers.Adam()
    loss = 'sparse_categorical_crossentropy'
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt,loss=loss,metrics=[acc])


def train_model(model,train_images,train_label):
    history = model.fit(train_images,train_label,epochs=5)
    return history


model = model_return(train_images_scale[0].shape)
model_compile(model)
history = train_model(model,train_images_scale,train_labels)

# df = pd.DataFrame(history.history)
# plot_acc = df.plot(y="sparse_categorical_accuracy",title="acccuracy vs epoch",legend=False)
# plot_acc.set(xlabel="epoch",ylabel="accuracy")
# plt.savefig("accuracy.png")

# plot_loss = df.plot(y="loss", title="Loss vs epoch",legend=False)
# plot_loss.set(xlabel="epoch",ylabel="loss")
# plt.savefig("loss.png")


num_test_images = test_images_scale.shape[0]
random_im_num = np.random.choice(num_test_images,4)
random_test_images = test_images_scale[random_im_num,...]
random_test_labels = test_labels[random_im_num,...]

predictions  = model.predict(random_test_images)

fig,axes = plt.subplots(4,2,figsize=(16,12))
fig.subplots_adjust(hspace=0.4,wspace=0.2)

for i,(prediction,image,label) in enumerate(zip(predictions,random_test_images,random_test_labels)):
    axes[i,0].imshow(np.squeeze(image))
    axes[i,0].get_xaxis().set_visible(False)
    axes[i,0].get_yaxis().set_visible(False)
    axes[i,0].text(10.,-1.5,f'Digit {label}')

    axes[i,1].bar(np.arange(len(prediction)),prediction)
    axes[i,1].set_xticks(np.arange(len(prediction)))
    axes[i,1].set_title(f'Categorical distribution. Model prediction : {np.argmax(prediction)}')
plt.show()



