---
layout: post
title:  "Using Tensorflow for Image Classification"
categories: blog assignment
permalink: posts/image-classification
author: Pei Xi Kwok
---
In this blog post we will learn how to use Tensorflow to build an image classification model. For this, we are interested in teaching a machine learning algorithm to distinguish between pictures of dogs and pictures of cats.

## 1. Load packages and obtain data

Let's start by importing some packages that we'll need.
```python
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

Now, let's access the data. We’ll use a sample data set provided by the TensorFlow team that contains labeled images of cats and dogs.

```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```
By running this code, we have created TensorFlow Datasets for training, validation, and testing. You can think of a Dataset as a pipeline that feeds data to a machine learning model. We use data sets in cases in which it’s not necessarily practical to load all the data into memory.

In our case, we’ve used a special-purpose keras utility called image_dataset_from_directory to construct a Dataset. The most important argument is the first one, which says where the images are located. The shuffle argument says that, when retrieving data from this directory, the order should be randomized. The batch_size determines how many data points are gathered from the directory at once. Here, for example, each time we request some data we will get 32 images from each of the data sets. Finally, the image_size specifies the size of the input images, just like you’d expect.

Let's explore our dataset. The function below produces a two-row visualization that shows three random pictures of cats in the first row and three random pictures of dogs in the second row. 
```python
def visualize(dataset):
  fig, axes = plt.subplots(2,3)
  for images, labels in dataset.take(1):
    for i in range(6):
      if labels[i]== 0:
        axes[0,i%3].imshow(images[i].numpy().astype("uint8"))
        axes[0,i%3].set_title("cats")
        axes[0,i%3].set_axis_off()
      elif labels[i]== 1:
        axes[1,i%3].imshow(images[i].numpy().astype("uint8"))
        axes[1,i%3].set_title("dogs")
        axes[1,i%3].set_axis_off()

visualize(train_dataset)
```
![visualize.png](/images/visualize.png)

We want to find out the frequencies of each of the labels in the dataset to understand what the baseline model should be. In our case, we want to find out the number of images in the training data that are labelled 0 (corresponding to "cat") and 1 (corresponding to "dog"). We'll do this by creating an iterator called labels.

```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
```

Next, we'll create a for-loop to count the number of images with the two labels.
```python
cat_count = 0
dog_count = 0

for label in labels_iterator:
    if label == 0:
      cat_count += 1
    elif label == 1:
      dog_count += 1

cat_count,dog_count
```
The output of the code is
```
    (1000, 1000)
```
This indicates to us that the baseline model will achieve 50% accuracy, given that there is an equal number of dog photos and cat photos. Going forward, we will want our models to achieve accuracy of more than 50%.

## 2. Building our first model

In Tensorflow, we build a model using different layers. In our case, we are interested in creating a convolution neural network, where convolution is performed to extract features from images before being put through multilayer perceptrons.

To do this, we will make use of the Keras API to construct a model, including:
- Conv2D, which convolves an image with a kernel
- MaxPooling2D, which downsamples the input by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input
- Flatten, which flattens the input
- Dense, which creates a densely-connected neural network layer

```python
model1 = models.Sequential([
          layers.Conv2D(32,(3,3),activation='relu',input_shape = (160,160,3)),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(32, (3, 3), activation='relu'),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(64, (3, 3), activation='relu'),
          layers.Flatten(),

          layers.Dense(64, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dropout(rate=0.5),
          layers.Dense(2)
])
```
After constructing our model, we need to compile it.

```python
model1.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```

Now that our model has been created, we need to train it using our train_dataset.

```python
history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 64s 1s/step - loss: 17.5892 - accuracy: 0.4990 - val_loss: 0.6929 - val_accuracy: 0.4913
    Epoch 2/20
    63/63 [==============================] - 63s 997ms/step - loss: 0.6955 - accuracy: 0.4955 - val_loss: 0.6944 - val_accuracy: 0.4777
    Epoch 3/20
    63/63 [==============================] - 63s 1s/step - loss: 0.7076 - accuracy: 0.4940 - val_loss: 0.6897 - val_accuracy: 0.5507
    Epoch 4/20
    63/63 [==============================] - 63s 997ms/step - loss: 0.6883 - accuracy: 0.5550 - val_loss: 0.6740 - val_accuracy: 0.6002
    Epoch 5/20
    63/63 [==============================] - 63s 1s/step - loss: 0.6312 - accuracy: 0.6475 - val_loss: 0.6717 - val_accuracy: 0.6089
    Epoch 6/20
    63/63 [==============================] - 63s 1000ms/step - loss: 0.4813 - accuracy: 0.7735 - val_loss: 0.8303 - val_accuracy: 0.6077
    Epoch 7/20
    63/63 [==============================] - 63s 1s/step - loss: 0.3258 - accuracy: 0.8630 - val_loss: 1.1339 - val_accuracy: 0.5866
    Epoch 8/20
    63/63 [==============================] - 63s 998ms/step - loss: 0.2195 - accuracy: 0.9210 - val_loss: 1.6777 - val_accuracy: 0.5743
    Epoch 9/20
    63/63 [==============================] - 63s 994ms/step - loss: 0.1341 - accuracy: 0.9555 - val_loss: 1.7734 - val_accuracy: 0.5854
    Epoch 10/20
    63/63 [==============================] - 63s 998ms/step - loss: 0.0792 - accuracy: 0.9735 - val_loss: 2.4515 - val_accuracy: 0.6052
    Epoch 11/20
    63/63 [==============================] - 63s 999ms/step - loss: 0.0723 - accuracy: 0.9810 - val_loss: 2.1215 - val_accuracy: 0.5953
    Epoch 12/20
    63/63 [==============================] - 63s 994ms/step - loss: 0.0501 - accuracy: 0.9885 - val_loss: 2.5952 - val_accuracy: 0.5817
    Epoch 13/20
    63/63 [==============================] - 63s 998ms/step - loss: 0.0305 - accuracy: 0.9915 - val_loss: 2.2058 - val_accuracy: 0.5371
    Epoch 14/20
    63/63 [==============================] - 63s 998ms/step - loss: 0.0618 - accuracy: 0.9855 - val_loss: 2.7465 - val_accuracy: 0.5668
    Epoch 15/20
    63/63 [==============================] - 63s 1s/step - loss: 0.0236 - accuracy: 0.9935 - val_loss: 3.2483 - val_accuracy: 0.5953
    Epoch 16/20
    63/63 [==============================] - 63s 996ms/step - loss: 0.0089 - accuracy: 0.9980 - val_loss: 2.7318 - val_accuracy: 0.5928
    Epoch 17/20
    63/63 [==============================] - 63s 994ms/step - loss: 0.0693 - accuracy: 0.9900 - val_loss: 2.7174 - val_accuracy: 0.5483
    Epoch 18/20
    63/63 [==============================] - 63s 994ms/step - loss: 0.0901 - accuracy: 0.9755 - val_loss: 2.6094 - val_accuracy: 0.5767
    Epoch 19/20
    63/63 [==============================] - 63s 1000ms/step - loss: 0.0167 - accuracy: 0.9945 - val_loss: 3.5129 - val_accuracy: 0.5903
    Epoch 20/20
    63/63 [==============================] - 63s 994ms/step - loss: 0.0205 - accuracy: 0.9950 - val_loss: 2.7705 - val_accuracy: 0.5990

The accuracy of model1 stabilized between 56% to 60% during training, which makes the accuracy of model1 6-9% better than the baseline model. As the training accuracy is often much higher than the validation accuracy in any given epoch, there is an issue with overfitting in model1.

## 3. 


