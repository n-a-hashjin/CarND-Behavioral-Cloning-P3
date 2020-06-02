# **Behavioral Cloning** 

## Self Driving Car Nanodegree

### Nasser Azarbakhsh HAshjin

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_imgs/csv_file.PNG "Log csv file"
[image2]: ./report_imgs/flip_image.PNG "Augmentation images"
[image3]: ./report_imgs/center_view.jpg "Center view"
[image4]: ./report_imgs/cropping.PNG "Cropping"
[image5]: ./report_imgs/NN_Model.PNG "Neural Network architecture"
[image6]: ./report_imgs/NN-Architecture.PNG "Model summary"
[image7]: ./report_imgs/recover-1.jpg "recover 1st image"
[image8]: ./report_imgs/recover-2.jpg "recover 2nd image"
[image9]: ./report_imgs/recover-3.jpg "recover 3rd image"


### Submitted files

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb represent the model.py screept and illustrates the dataset
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### drive.py script
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### model.py script

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. Model architecture

My model is mostly the same as Nvidia self drivig neural network [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). It consists of 5 [convolution](https://keras.io/api/layers/convolution_layers/convolution2d/) neural network following with 4 fully connected layers. First 3 convolutional networks have size of 5x5 with strides of 2-by-2 then. Their depths are 24, 36 and 48 respectively. After them there is another two convolutional layers that have 3x3 size and strides of 1-by-1 with depths are 64.

Then network is followed by 4 fully connected layers of sizes 100, 50, 10 and 1. I have used Keras [Dense](https://keras.io/api/layers/core_layers/dense/) layer for this purpose.

The model includes RELU layers to introduce nonlinearity (code lines from 108 to 112), and the data is croppted and normalized in the model using keras [Cropping2D](https://keras.io/api/layers/reshaping_layers/cropping2d/) and a Keras [Lambda](https://keras.io/api/layers/core_layers/lambda/) layer (code lines 105 and 106 respectively). 

#### 2. Attempts to reduce overfitting in the model

After convolutional layers I have used a [Dropout](https://keras.io/api/layers/regularization_layers/dropout/) layer to reduce overfitting (model.py lines 114).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94 and 95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 135).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding and subtracting a correction factor of 0.2 to left and right sides view respectively.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to a combination of convolutional neural network (CNN) and fully connected layers to build a deep NN so it becomes an end to end process. The input of network is images of cameras installed on a car (here from simulation application) and the output is steering angel to control the steering of vehicle.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it has multy convolutional layers and ables to find different lines and shapes, which can be useful to learn to detect lane lines. And Multy Layer perceptrons gives the network the ability to learn how produce steering angles from detected features (lines and shapes from ConvNet).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I have used a dropout layer to generalize the model and reduce overfitting issue.

Then I saw improving result but still it wasn't good enough to drive car flawless. I realize my model might need more deep network specially in convolutional layers where features extracts from images. I have used more convolutional layers and it was helpful so my model could learn better and drive more reliable.

I have finally imployed Nvidia self driving car model by adding a fully connected layer with one node at the end of model to reduce outputs and get just one steering angel as output of network. I have also used a dropout as mentioned before to reduce overfitting issue.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I cropped the input image to feed the model with more useful part of input image, which is road and lane lines part of images. And also I used a lambda layer to implement normalization that helps to stablizing the learning process and reduces the number of training epochs through reduce "*internal covariate shift*".

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road ([video](video.mp4)).

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-124) consisted of a convolution neural network with the following layers and layer sizes:
![alt text][image6]

Here is a visualization of the architecture.

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and one counter-clock lap. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image7]
![alt text][image8]
![alt text][image9]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image2]

Etc ....

After the collection process, I had 8036 number of data points. Using left and right side view images and flipping images as augmentation method provided 48216 data points in total. Here is a part of csv log file (contains image paths and steering angels using pandas library).

![alt text][image1]

Then I preprocessed this data by cropping the input image to feed the model with more useful part of input image, which is road and lane lines part of images. And also I used a lambda layer to implement normalization that helps to stablizing the learning process and reduces the number of training epochs through reduce "*internal covariate shift*".

Here is an example of what input looks like after cropping it (25 from bottom and 55 from top of image).
![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
