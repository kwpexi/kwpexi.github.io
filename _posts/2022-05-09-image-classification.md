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

The code below can help us visualize the performance of the model over the course of the 20 epochs.
```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![model1.png](/images/model1.png)

The accuracy of model1 stabilized between **56% to 60%** during training, which makes the accuracy of model1 6-9% better than the baseline model. As the training accuracy is often much higher than the validation accuracy in any given epoch, there is an issue with overfitting in model1. 

## 3. Data augmentation
Data augmentation refers to the practice of including modified copies of the same image in the training set. For example, a picture of a cat is still a picture of a cat even if we flip it upside down or rotate it 90 degrees. Including such transformed versions of the image in our training process can help our model learn so-called invariant features of our input images.

We can do this using tf.keras.layers.RandomFlip(), which randomly flips the image.
```python
from tensorflow.python.data.ops.dataset_ops import RandomDataset

random_flip = tf.keras.Sequential([
  tf.keras.layers.RandomFlip()
])
```
This code will help us visualize the flipped images.

```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = random_flip(tf.expand_dims(first_image, 0),training=True)
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```
![randomflip.png](/images/randomflip.png)

Another way we can do this is by using tf.keras.layers.RandomRotation(), which randomly rotates the image.
```python
random_rotation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.2)
])
```
```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = random_rotation(tf.expand_dims(first_image, 0), training = True)
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```
![randomrotate.png](/images/randomrotate.png)

Now that we have an idea of how randomflip and randomrotate work, we'll use it in our model to augment our data. Similar to model1, we begin by creating and compiling the model.

```python
model2 = models.Sequential([
          layers.RandomFlip('horizontal_and_vertical'),
          layers.RandomRotation(0.1, fill_mode='reflect'),
          layers.Conv2D(32,(3,3),activation='relu',input_shape = (160,160,3)),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(32, (3, 3), activation='relu'),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(64, (3, 3), activation='relu'),
          layers.Flatten(),
          layers.Dropout(0.2),

          layers.Dense(64, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dense(2)  
])
```
```python
model2.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```
Let's train our new model using our training data.
```python
history = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 16s 63ms/step - loss: 6.7116 - accuracy: 0.5205 - val_loss: 0.7006 - val_accuracy: 0.5285
    Epoch 2/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6925 - accuracy: 0.5280 - val_loss: 0.7231 - val_accuracy: 0.5260
    Epoch 3/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.6911 - accuracy: 0.5370 - val_loss: 0.6888 - val_accuracy: 0.5619
    Epoch 4/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6851 - accuracy: 0.5590 - val_loss: 0.6869 - val_accuracy: 0.5767
    Epoch 5/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.6878 - accuracy: 0.5595 - val_loss: 0.6875 - val_accuracy: 0.5631
    Epoch 6/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.6674 - accuracy: 0.6050 - val_loss: 0.7140 - val_accuracy: 0.5545
    Epoch 7/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.6698 - accuracy: 0.6155 - val_loss: 0.7005 - val_accuracy: 0.5656
    Epoch 8/20
    63/63 [==============================] - 4s 66ms/step - loss: 0.6828 - accuracy: 0.5560 - val_loss: 0.6756 - val_accuracy: 0.6126
    Epoch 9/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.6674 - accuracy: 0.6005 - val_loss: 1.1808 - val_accuracy: 0.4975
    Epoch 10/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6845 - accuracy: 0.5855 - val_loss: 0.7278 - val_accuracy: 0.5111
    Epoch 11/20
    63/63 [==============================] - 4s 61ms/step - loss: 0.6835 - accuracy: 0.5715 - val_loss: 0.6767 - val_accuracy: 0.5755
    Epoch 12/20
    63/63 [==============================] - 4s 60ms/step - loss: 0.6679 - accuracy: 0.5975 - val_loss: 0.6747 - val_accuracy: 0.6077
    Epoch 13/20
    63/63 [==============================] - 4s 61ms/step - loss: 0.6591 - accuracy: 0.6070 - val_loss: 0.6736 - val_accuracy: 0.5854
    Epoch 14/20
    63/63 [==============================] - 4s 61ms/step - loss: 0.6616 - accuracy: 0.5995 - val_loss: 0.6773 - val_accuracy: 0.5668
    Epoch 15/20
    63/63 [==============================] - 4s 62ms/step - loss: 0.6601 - accuracy: 0.6155 - val_loss: 0.6737 - val_accuracy: 0.5780
    Epoch 16/20
    63/63 [==============================] - 4s 59ms/step - loss: 0.6584 - accuracy: 0.6195 - val_loss: 0.6693 - val_accuracy: 0.5767
    Epoch 17/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6488 - accuracy: 0.6270 - val_loss: 0.6544 - val_accuracy: 0.6498
    Epoch 18/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6531 - accuracy: 0.6175 - val_loss: 0.6578 - val_accuracy: 0.6275
    Epoch 19/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6623 - accuracy: 0.6010 - val_loss: 0.6802 - val_accuracy: 0.5792
    Epoch 20/20
    63/63 [==============================] - 4s 58ms/step - loss: 0.6488 - accuracy: 0.6290 - val_loss: 0.6825 - val_accuracy: 0.5780
    
![model2.png](/images/model2.png)
The accuracy of model2 stabilized between **57% to 64%** during training. In addition to the data augmentation layers, I also moved the dropout layer to be after the flatten layer, and adjusted the value to be lower. Model2's performance is between 6-10% better than the baseline model, and although there is some overfitting , the maximum difference between training accuracy and validation accuracy is much lower than in model1, being less than 6%.

## 4. Data preprocessing
Sometimes, it can be helpful to make simple transformations to the input data. For example, in this case, the original data has pixels with RGB values between 0 and 255, but many models will train faster with RGB values normalized between 0 and 1, or possibly between -1 and 1. These are mathematically identical situations, since we can always just scale the weights. But if we handle the scaling prior to the training process, we can spend more of our training energy handling actual signal in the data and less energy having the weights adjust to the data scale.

We will do this by creating a preprocessing layer which can be added to the model pipeline.
```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```
We will then construct and compile the model with the added preprocessor layer.
```python
model3 = models.Sequential([
          preprocessor,
          layers.RandomFlip(),
          layers.RandomRotation(0.1, fill_mode='reflect'),
          layers.Conv2D(32,(3,3),activation='relu',input_shape = (160,160,3)),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(32, (3, 3), activation='relu'),
          layers.MaxPooling2D((2, 2)),
          layers.Conv2D(64, (3, 3), activation='relu'),
          layers.Flatten(),
          layers.Dropout(rate=0.2),

          layers.Dense(64, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dense(2)  
])
```
```python
model3.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```
Let's train our new model using our training data.
```python
history = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 5s 59ms/step - loss: 0.7093 - accuracy: 0.5195 - val_loss: 0.6662 - val_accuracy: 0.6200
    Epoch 2/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.6770 - accuracy: 0.5675 - val_loss: 0.6699 - val_accuracy: 0.5842
    Epoch 3/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.6643 - accuracy: 0.5790 - val_loss: 0.6307 - val_accuracy: 0.6411
    Epoch 4/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.6353 - accuracy: 0.6325 - val_loss: 0.6097 - val_accuracy: 0.6770
    Epoch 5/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.6217 - accuracy: 0.6465 - val_loss: 0.6427 - val_accuracy: 0.5941
    Epoch 6/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.6132 - accuracy: 0.6590 - val_loss: 0.5996 - val_accuracy: 0.6671
    Epoch 7/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5818 - accuracy: 0.6895 - val_loss: 0.6315 - val_accuracy: 0.6745
    Epoch 8/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5854 - accuracy: 0.6940 - val_loss: 0.5766 - val_accuracy: 0.6807
    Epoch 9/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5656 - accuracy: 0.6920 - val_loss: 0.5733 - val_accuracy: 0.7129
    Epoch 10/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5568 - accuracy: 0.7030 - val_loss: 0.5683 - val_accuracy: 0.7092
    Epoch 11/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5468 - accuracy: 0.7180 - val_loss: 0.5668 - val_accuracy: 0.7116
    Epoch 12/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5417 - accuracy: 0.7115 - val_loss: 0.5746 - val_accuracy: 0.7030
    Epoch 13/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5348 - accuracy: 0.7225 - val_loss: 0.5519 - val_accuracy: 0.7030
    Epoch 14/20
    63/63 [==============================] - 4s 57ms/step - loss: 0.5408 - accuracy: 0.7185 - val_loss: 0.6004 - val_accuracy: 0.6918
    Epoch 15/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5240 - accuracy: 0.7260 - val_loss: 0.5620 - val_accuracy: 0.7277
    Epoch 16/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5079 - accuracy: 0.7400 - val_loss: 0.5648 - val_accuracy: 0.7166
    Epoch 17/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5130 - accuracy: 0.7385 - val_loss: 0.6006 - val_accuracy: 0.7030
    Epoch 18/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.4933 - accuracy: 0.7620 - val_loss: 0.6130 - val_accuracy: 0.7141
    Epoch 19/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.5027 - accuracy: 0.7480 - val_loss: 0.5336 - val_accuracy: 0.7314
    Epoch 20/20
    63/63 [==============================] - 4s 56ms/step - loss: 0.4805 - accuracy: 0.7680 - val_loss: 0.5795 - val_accuracy: 0.7116
    
```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![model3.png](/images/model3.png)

The accuracy of model3 stabilized between **70% to 72%** during training. This was achieved with the addition of the preprocessing layer. The model is now 20-22% better than the baseline accuracy. Again, there is a small amount of overfitting observed.

## 5. Transfer learning
So far, we’ve been training models for distinguishing between cats and dogs from scratch. In some cases, however, someone might already have trained a model that does a related task, and might have learned some relevant patterns. In this section, we'll be trying out a pre-existing model for image recognition. 

We'll start by downloading ```MobileNetV2``` and configuring it as a layer that can be included in our model.

```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```
Let's create a model using base_model_layer. Here, we'll be adding a ```GlobalMaxPooling2D``` layer, which performs something similar to ```MaxPooling2D```, with pool size being equal to input size.
```python
model4 = models.Sequential([
          preprocessor,
          layers.RandomFlip(),
          layers.RandomRotation(0.1, fill_mode='reflect'),
          base_model_layer,
          layers.GlobalMaxPooling2D(),
          layers.Dense(64, activation='relu'),
          layers.Dense(2)             
])
```
```python
model4.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```
Let's take a look at what's in our model.

```python
model4.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model (Functional)          (None, 160, 160, 3)       0         
                                                                     
     random_flip_8 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     random_rotation_7 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     model_1 (Functional)        (None, 5, 5, 1280)        2257984   
                                                                     
     global_max_pooling2d_7 (Glo  (None, 1280)             0         
     balMaxPooling2D)                                                
                                                                     
     dense_14 (Dense)            (None, 64)                81984     
                                                                     
     dense_15 (Dense)            (None, 2)                 130       
                                                                     
    =================================================================
    Total params: 2,340,098
    Trainable params: 82,114
    Non-trainable params: 2,257,984
    _________________________________________________________________
    
Based on this, we can see that we need to train 82,114 parameters.

Let's train our model!

```python
history4 = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 8s 81ms/step - loss: 0.6628 - accuracy: 0.8265 - val_loss: 0.1502 - val_accuracy: 0.9443
    Epoch 2/20
    63/63 [==============================] - 4s 63ms/step - loss: 0.2384 - accuracy: 0.9090 - val_loss: 0.0820 - val_accuracy: 0.9691
    Epoch 3/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.2026 - accuracy: 0.9240 - val_loss: 0.0781 - val_accuracy: 0.9666
    Epoch 4/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1929 - accuracy: 0.9165 - val_loss: 0.0736 - val_accuracy: 0.9703
    Epoch 5/20
    63/63 [==============================] - 4s 63ms/step - loss: 0.1709 - accuracy: 0.9265 - val_loss: 0.0682 - val_accuracy: 0.9802
    Epoch 6/20
    63/63 [==============================] - 4s 65ms/step - loss: 0.1449 - accuracy: 0.9415 - val_loss: 0.0665 - val_accuracy: 0.9765
    Epoch 7/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1701 - accuracy: 0.9335 - val_loss: 0.0644 - val_accuracy: 0.9802
    Epoch 8/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1782 - accuracy: 0.9230 - val_loss: 0.0755 - val_accuracy: 0.9728
    Epoch 9/20
    63/63 [==============================] - 4s 65ms/step - loss: 0.1392 - accuracy: 0.9435 - val_loss: 0.0663 - val_accuracy: 0.9691
    Epoch 10/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1247 - accuracy: 0.9495 - val_loss: 0.0652 - val_accuracy: 0.9728
    Epoch 11/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1427 - accuracy: 0.9415 - val_loss: 0.0534 - val_accuracy: 0.9839
    Epoch 12/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1159 - accuracy: 0.9520 - val_loss: 0.0549 - val_accuracy: 0.9740
    Epoch 13/20
    63/63 [==============================] - 4s 65ms/step - loss: 0.1291 - accuracy: 0.9445 - val_loss: 0.0903 - val_accuracy: 0.9666
    Epoch 14/20
    63/63 [==============================] - 4s 65ms/step - loss: 0.1154 - accuracy: 0.9480 - val_loss: 0.0688 - val_accuracy: 0.9691
    Epoch 15/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1165 - accuracy: 0.9525 - val_loss: 0.0654 - val_accuracy: 0.9691
    Epoch 16/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1193 - accuracy: 0.9530 - val_loss: 0.0487 - val_accuracy: 0.9790
    Epoch 17/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1133 - accuracy: 0.9510 - val_loss: 0.0603 - val_accuracy: 0.9752
    Epoch 18/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.1130 - accuracy: 0.9555 - val_loss: 0.0527 - val_accuracy: 0.9777
    Epoch 19/20
    63/63 [==============================] - 4s 64ms/step - loss: 0.0902 - accuracy: 0.9655 - val_loss: 0.0612 - val_accuracy: 0.9715
    Epoch 20/20
    63/63 [==============================] - 4s 66ms/step - loss: 0.1118 - accuracy: 0.9535 - val_loss: 0.0637 - val_accuracy: 0.9740
    
```python
plt.plot(history4.history["accuracy"], label = "training")
plt.plot(history4.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![model4.png](/images/model4.png)

The accuracy of model4 stabilized between 96% to 97% during training. This was with the use of the base model layer, as well as an additional dense64 layer. The model is now 46-47% better than the baseline accuracy. In this model, the validation accuracy is often higher than the training accuracy, indicating no overfitting.

## 6. Score on test data
Let's now evaluate our model using the unseen test_dataset!

```python
loss, accuracy = model4.evaluate(test_dataset)
print('Test accuracy :', accuracy)
```

    6/6 [==============================] - 1s 78ms/step - loss: 0.0477 - accuracy: 0.9844
    Test accuracy : 0.984375

The accuracy of our model on the test data is approximately 98%, which is quite high. Good job!

And that's how you use Tensorflow to build an image classification model.






