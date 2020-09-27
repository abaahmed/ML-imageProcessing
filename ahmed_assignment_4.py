
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd


fashion_mnist = keras.datasets.fashion_mnist
# splitting training and test data and corresponding labels 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_dict = {i:class_name for i,class_name in enumerate(class_names)}

def show_image(index):
    plt.figure()
    # cmap=plt.cm.binary allows us to show the picture in grayscale
    plt.imshow(train_images[index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[index]])
    plt.colorbar() # adds a bar to the side with values
    plt.show()

show_image(0)


#Normalize Data
train_images = train_images / 255
test_images =  test_images / 255


show_image(0)


# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#Reshape the data into a 28 x 28 = 784 Dimen. 
print(f'Before reshape, train_images shape: {train_images.shape} test_images shape: {test_images.shape}')
train_images = np.reshape(train_images, (train_images.shape[0], 784))
test_images = np.reshape(test_images, (test_images.shape[0], 784))
print(f'Before reshape, train_images shape: {train_images.shape} test_images shape: {test_images.shape}')


# Add training data into a dataframe
img_data = {f"z{i}":train_images[:,i] for i in range(784)}
img_data["label"] = train_labels
df_img_train = pd.DataFrame(img_data)
df_img_train["class"] = df_img_train["label"].map(class_dict)
df_img_train.head()

# Add test data into a dataframe
img_data = {f"z{i}":test_images[:,i] for i in range(784)}
img_data["label"] = test_labels
df_img_test = pd.DataFrame(img_data)
df_img_test["class"] = df_img_test["label"].map(class_dict)
df_img_test.head()



# A function for getting a subset of the data
def get_data_subset(df, classes=[], shuffle=True, shuffle_seed=42):
    """
    Used to retrieve columns from df
    """
    if classes == []:
        print("Pleas")
    else:
        df_filtered = df[(df["class"] == classes[0]) | (df["class"] == classes[1])].copy()
        df_filtered["binary_label"] = 0
        df_filtered.loc[df["class"] == classes[1], "binary_label"] = 1
        data = df_filtered.filter(regex=("z[0-9]+")).values
        labels = df_filtered["binary_label"].values
        if shuffle:
            np.random.seed(shuffle_seed)
            np.random.shuffle(data)
            np.random.seed(shuffle_seed)
            np.random.shuffle(labels)

    return data, labels.reshape(-1,1)

# Preparing data for training, use get_data_subset along with df_img_train

X, y = get_data_subset(df_img_train, classes=["Pullover", "Coat"])


model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])



from tensorflow.keras.layers import Input, Dense # only use these layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import * 
from tensorflow.keras import regularizers


input_layer = Input(shape=(784))
x = Dense(45, activation='tanh')(input_layer)
x = Dense(25, activation='tanh')(x)
out = Dense(10, activation='softmax', activity_regularizer=regularizers.l2(1e-5))(x)
   

model = Model(input_layer, out)

# Show a summary of  model
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# Call fit on model passing in the train_images, train_labels data above with validation split of 0.2 and train for 100 epochs

hist =  model.fit(X, y, validation_split = 0.2,  epochs = 100)

def plot_losses(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()
def plot_accuracies(hist):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

# Plot losses and accuracies
plot_losses(hist)
plot_accuracies(hist)

