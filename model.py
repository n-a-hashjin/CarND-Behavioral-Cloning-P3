import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### load data
# define datapath -where our collected samples are stored-
data_path = "./data/"

# use panda to read csv file that contains images pathes and their corresponding steering values
database = pd.read_csv(data_path + 'driving_log.csv', skipinitialspace=True)


# load csv file in list, images_list for images pathes
# labels_list for steering values
images_list = []
labels_list = []

# define correction factor for calculation of proper
# steering values from left and right sidesveiew points
correction = 0.2

# load csv file to lists
for i in range(len(database)):
    images_list.append(database.center[i])
    labels_list.append(database.steering[i])

    images_list.append(database.left[i])
    labels_list.append(database.steering[i] + correction)

    images_list.append(database.right[i])
    labels_list.append(database.steering[i] - correction)


### create a generator to load data batches
from sklearn.utils import shuffle

def generator(images_list, labels_list, data_path, batch_size=32, shuffle_on=True, flip_on=True):
    ### generator function, generates data batches of defined baches_size (default 32)
    # inputs:
    #            images_list: list of strings, images pathes in dataset directory.
    #            labels_list: list of steering values corresponded to images in images_list
    #            data_path:   string, of dataset path, where data (csv and images) are stored
    #            batch_size:  integer, size of ouput batches (first dimension of output)
    #            shuffle_on:  bool, turning on shuffle when True
    #            flip_on:     bool, flip images as data augmentation process
    # outputs:
    #            images:      images batch, four dimensional numpy array
    #            steerings:   steerings angels of images in images batch

    if flip_on:
        n_img = 2
    else:
        n_img = 1
    # in case of flip_on batches size will be devided on 2
    # however as each image produces a flip/miror image the output has batches size as defined
    # in case of odd number of batches_size output batches have 1 image fewer
    batch_size = int(batch_size / n_img)

    while 1:
        # shuffle data if shuffle is on
        if shuffle_on:
                images_list, labels_list = shuffle(images_list, labels_list)
        # create batches
        for offset in range(0, len(images_list)-batch_size, batch_size):
            images = []
            steerings = []
            # load images and streen values in batch
            for i in range(offset,offset + batch_size):

                image = plt.imread(data_path + images_list[i])
                steering = labels_list[i]

                images.append(image)
                steerings.append(steering)

            # convert list to numpy array as network works only on numpy arrays
            images = np.array(images)
            steerings = np.array(steerings)

            # flip loaded images and add it to batch, (douplicate the size of loaded batch)
            if flip_on:
                images = np.append(images, np.copy(images)[:,:,::-1,:], axis=0)
                steerings = np.append(steerings, np.copy(steerings)*-1, axis=0)

            # shuffle batch
            if shuffle_on:
                images, steerings = shuffle(images, steerings)

            # using yield to make a generator
            yield images, steerings

### Training and validation data
# separate train and validation samples of dataset, 80% training data and 20% validation data
from sklearn.model_selection import train_test_split
train_images, validation_images, train_labels, validation_labels = train_test_split(images_list, labels_list, train_size=0.8)

### Create nerual network model using Keras
from keras.models import Model
from keras.layers import Dense, Flatten, Lambda, Conv2D, Cropping2D, Dropout, Input

# Nvidia self driving car model with dropout layer to generalization model more
# input
inputs = Input(shape=(160,320,3))
# preprocessing, cropping and normalization of images
cropping_1 = Cropping2D(cropping=((55,25),(0,0)))(inputs)
lambda_1 = Lambda(lambda x: x / 256.0 - 0.5)(cropping_1)
# ConvNet
conv2d_1 = Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu')(lambda_1)
conv2d_2 = Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu')(conv2d_1)
conv2d_3 = Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu')(conv2d_2)
conv2d_4 = Conv2D(64, kernel_size=(3,3), activation='relu')(conv2d_3)
conv2d_5 = Conv2D(64, kernel_size=(3,3), activation='relu')(conv2d_4)
# Dropout
dropout_1 = Dropout(0.5)(conv2d_5)
# flatten output of CNN after dropout
flatten_1 = Flatten()(dropout_1)
# MLN
dense_1 = Dense(100)(flatten_1)
dense_2 = Dense(50)(dense_1)
dense_3 = Dense(10)(dense_2)
output  = Dense(1)(dense_3)

# create model, define inputs and outputs of network
model = Model(inputs=inputs, outputs=output)

# print network architecture
print(model.summary())

# define train and validation data generators
batch_size = 120
train_generator = generator(train_images, train_labels, data_path, batch_size=batch_size, flip_on=True)
validation_generator = generator(validation_images, validation_labels, data_path, batch_size=batch_size, flip_on=True)

# compile model using 'adam' algorithem -which controls learning process
model.compile(optimizer='adam', loss='mse')

# calculate number of steps per epochs, training data
n_train_batches = int(2 * len(train_images) / batch_size) - 1
# calculate number of steps per epochs, validation data
n_valid_batches = int(2 * len(train_images) / batch_size) - 1

# run model and start training process
model.fit_generator(train_generator, steps_per_epoch=n_train_batches, epochs=7,\
validation_data=validation_generator, validation_steps=n_valid_batches, verbose=1)

# save model and exit
model.save('model.h5')
exit()
