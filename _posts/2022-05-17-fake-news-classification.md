---
layout: post
title:  "Classifying fake news"
categories: blog assignment
permalink: posts/fake-news-classification
author: Pei Xi Kwok
---

In this blog post, we'll be training and developing a fake news classifier using Tensorflow. We'll be making use of data from the article
- Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).

## 1. Import packages and read in data

We'll start by importing the necessary packages and reading in the data needed.
```python
# for building models
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string

from tensorflow import keras
from tensorflow.keras import layers, losses, Input, Model, utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for embedding visualization
import matplotlib.pyplot as plt

import plotly.express as px 
import plotly.io as pio
pio.templates.default = "plotly_white"

# reading in data
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
df = pd.read_csv(train_url, index_col = 0)
df.head()
```
Here's a look at our data:

 <div id="df-d5b4a725-955d-4032-b298-361afb38e2d7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17366</th>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5634</th>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17487</th>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12217</th>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5535</th>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d5b4a725-955d-4032-b298-361afb38e2d7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d5b4a725-955d-4032-b298-361afb38e2d7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d5b4a725-955d-4032-b298-361afb38e2d7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

Each row of the data corresponds to an article. The title column gives the title of the article, while the text column gives the full article text. The final column, called fake, is 0 if the article is true and 1 if the article contains fake news, as determined by the authors of the paper above.

## 2. Making a dataset

Before we start splitting the data and training our model, we have to preprcoess the data by removing stopwords and construct and return a `tf.data.Dataset`. 

A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but”, and having them in our training data can lead to lower accuracy levels for our model.
```python
# import necessary packages
from gensim.utils import simple_preprocess # deals with lowercases, tokenizes, de-accents
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

def remove_stopwords(texts):
  return [' '.join([word for word in simple_preprocess(str(doc)) if word not in stop_words]) for doc in texts]
```
Let's write a function that will help to create a `tf.data.Dataset` with two inputs and one output. The `tf.data.Dataset` API helps to create more descriptive and efficient input pipelines so that the full dataset does not need to fit into memory.
```python
def make_dataset(data):
  # applies the function to remove stopwords
  data['title'] = remove_stopwords(data['title']) #remove stopwords from titles
  data['text'] = remove_stopwords(data['text']) #remove stopwords from text
  data = tf.data.Dataset.from_tensor_slices( #process it into a tensorflow dataset
      (
        {
            "title" : data[["title"]], 
            "text" : data[["text"]]
        }, 
        {
            "fake" : data["fake"]
        }
    )
  )
  return data.batch(100) # return batches of 100 rows to allow model to train on chunks of data
  # rather than individual rows

df = make_dataset(df)
```
We can now shuffle and split our data into the training and validation data.
```python
df = df.shuffle(buffer_size = len(df)) # shuffles data

# determine size for train and val set
train_size = int(0.8*len(df))
val_size   = int(0.2*len(df))

# splits data into train and val
train = df.take(train_size)
val = df.skip(train_size).take(val_size)
```
Before we dive into building our models, let's get a sense of our base rate by looking at the proportion of articles labelled as "fake" in our data.
```python
# create a labels iterator
labels_iterator= train.unbatch().map(lambda text,fake:fake).as_numpy_iterator()

# count number of fake articles and not-fake articles
fake_count = 0
not_fake_count = 0

for label in labels_iterator:
    if label['fake'] == 0:
      not_fake_count += 1
    elif label['fake'] == 1:
      fake_count += 1

sum = fake_count + not_fake_count
fake_count/sum, not_fake_count/sum
```
    (0.5219232269207198, 0.4780767730792802)
As base rate refers to the accuracy of a model that always makes the same guess, we can assume that the base rate will be approximately 52%.

Additionally, when working with NLP models, we need to vectorize the text and transform words into integers so that our model can make sense of the data. We will do this using TextVectorization.
```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

# preparing a text vectorization layer for titles
title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary,
    output_mode='int',
    output_sequence_length=25) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"])) # the layer learns how to map words to integers

# preparing a text vectorization layer for texts
# change o/p sequence length because text should be longer
text_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary,
    output_mode='int',
    output_sequence_length=500) 

text_vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```
Now we can begin to build our models to answer the question
> When detecting fake news, is it most effective to focus on only the title of the article, the full text of the article, or both?

## 3. Building a model using titles as input
We'll start with building a model which only uses the article titles data as input. As we will be using the Keras Functional API, we'll start by creating the input.
```python
titles_input = Input(
    shape = (1,), 
    name = "title",
    dtype = "string"
)
```
The next step is to create the several layers that will go into the model, including layers such as
- Embedding
- Dropout
- GlobalAveragePooling1D
- Dense

```python
# create shared embedding layer
shared_embedding = layers.Embedding(size_vocabulary, output_dim = 10, name = "embedding")

titles_features = title_vectorize_layer(titles_input)
titles_features = shared_embedding(titles_features)
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.GlobalAveragePooling1D()(titles_features)
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.Dense(32, activation='relu')(titles_features)

# create output layer
titles_output = layers.Dense(2, name = "fake")(titles_features)
```
We then create a `Model` by specifying its inputs and outputs in the graph of layers.
```python
model1 = keras.Model(
    inputs = titles_input,
    outputs = titles_output
)
```
We will then compile the model.
```python
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```
Let's take a look at our model.
```python
model1.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 25)               0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 25, 10)            20000     
                                                                     
     dropout (Dropout)           (None, 25, 10)            0         
                                                                     
     global_average_pooling1d (G  (None, 10)               0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 10)                0         
                                                                     
     dense (Dense)               (None, 32)                352       
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 20,418
    Trainable params: 20,418
    Non-trainable params: 0
    _________________________________________________________________
    

We will now train our model using the training data.
```python
history = model1.fit(train, 
                    validation_data=val,
                    epochs = 50
                    )

```
Epoch 1/50
    

    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    

    180/180 [==============================] - 2s 5ms/step - loss: 0.6083 - accuracy: 0.7204 - val_loss: 0.4041 - val_accuracy: 0.8885
    Epoch 2/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.2879 - accuracy: 0.8994 - val_loss: 0.2093 - val_accuracy: 0.9224
    Epoch 3/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1838 - accuracy: 0.9305 - val_loss: 0.1597 - val_accuracy: 0.9389
    Epoch 4/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1529 - accuracy: 0.9404 - val_loss: 0.1393 - val_accuracy: 0.9444
    Epoch 5/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1399 - accuracy: 0.9456 - val_loss: 0.1193 - val_accuracy: 0.9511
    Epoch 6/50
    180/180 [==============================] - 1s 7ms/step - loss: 0.1305 - accuracy: 0.9497 - val_loss: 0.1169 - val_accuracy: 0.9547
    Epoch 7/50
    180/180 [==============================] - 1s 7ms/step - loss: 0.1222 - accuracy: 0.9508 - val_loss: 0.1179 - val_accuracy: 0.9536
    Epoch 8/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1168 - accuracy: 0.9546 - val_loss: 0.1035 - val_accuracy: 0.9589
    Epoch 9/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1131 - accuracy: 0.9564 - val_loss: 0.0874 - val_accuracy: 0.9664
    Epoch 10/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1081 - accuracy: 0.9582 - val_loss: 0.1121 - val_accuracy: 0.9569
    Epoch 11/50
    180/180 [==============================] - 2s 9ms/step - loss: 0.1125 - accuracy: 0.9573 - val_loss: 0.0911 - val_accuracy: 0.9633
    Epoch 12/50
    180/180 [==============================] - 1s 7ms/step - loss: 0.1095 - accuracy: 0.9591 - val_loss: 0.1012 - val_accuracy: 0.9604
    Epoch 13/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1067 - accuracy: 0.9603 - val_loss: 0.0826 - val_accuracy: 0.9707
    Epoch 14/50
    180/180 [==============================] - 1s 7ms/step - loss: 0.1035 - accuracy: 0.9591 - val_loss: 0.0860 - val_accuracy: 0.9687
    Epoch 15/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.1015 - accuracy: 0.9611 - val_loss: 0.0925 - val_accuracy: 0.9656
    Epoch 16/50
    180/180 [==============================] - 1s 7ms/step - loss: 0.1019 - accuracy: 0.9604 - val_loss: 0.0855 - val_accuracy: 0.9689
    Epoch 17/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.0989 - accuracy: 0.9619 - val_loss: 0.0824 - val_accuracy: 0.9651
    Epoch 18/50
    180/180 [==============================] - 1s 6ms/step - loss: 0.0989 - accuracy: 0.9609 - val_loss: 0.0805 - val_accuracy: 0.9711
    Epoch 19/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.1006 - accuracy: 0.9615 - val_loss: 0.0814 - val_accuracy: 0.9711
    Epoch 20/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.1008 - accuracy: 0.9613 - val_loss: 0.0789 - val_accuracy: 0.9682
    Epoch 21/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0974 - accuracy: 0.9616 - val_loss: 0.0877 - val_accuracy: 0.9682
    Epoch 22/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0977 - accuracy: 0.9626 - val_loss: 0.0800 - val_accuracy: 0.9687
    Epoch 23/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0962 - accuracy: 0.9642 - val_loss: 0.0808 - val_accuracy: 0.9653
    Epoch 24/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0943 - accuracy: 0.9641 - val_loss: 0.0753 - val_accuracy: 0.9724
    Epoch 25/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0990 - accuracy: 0.9622 - val_loss: 0.0838 - val_accuracy: 0.9699
    Epoch 26/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0956 - accuracy: 0.9634 - val_loss: 0.0881 - val_accuracy: 0.9661
    Epoch 27/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0961 - accuracy: 0.9622 - val_loss: 0.0806 - val_accuracy: 0.9702
    Epoch 28/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0956 - accuracy: 0.9645 - val_loss: 0.0756 - val_accuracy: 0.9744
    Epoch 29/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0939 - accuracy: 0.9639 - val_loss: 0.0736 - val_accuracy: 0.9724
    Epoch 30/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0966 - accuracy: 0.9638 - val_loss: 0.0874 - val_accuracy: 0.9687
    Epoch 31/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0960 - accuracy: 0.9636 - val_loss: 0.0872 - val_accuracy: 0.9678
    Epoch 32/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0952 - accuracy: 0.9635 - val_loss: 0.0771 - val_accuracy: 0.9701
    Epoch 33/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0942 - accuracy: 0.9643 - val_loss: 0.0732 - val_accuracy: 0.9724
    Epoch 34/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0953 - accuracy: 0.9632 - val_loss: 0.0824 - val_accuracy: 0.9713
    Epoch 35/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0928 - accuracy: 0.9650 - val_loss: 0.0732 - val_accuracy: 0.9744
    Epoch 36/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0919 - accuracy: 0.9653 - val_loss: 0.0785 - val_accuracy: 0.9724
    Epoch 37/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0942 - accuracy: 0.9630 - val_loss: 0.0828 - val_accuracy: 0.9691
    Epoch 38/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0965 - accuracy: 0.9624 - val_loss: 0.0750 - val_accuracy: 0.9740
    Epoch 39/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0886 - accuracy: 0.9663 - val_loss: 0.0818 - val_accuracy: 0.9711
    Epoch 40/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0935 - accuracy: 0.9636 - val_loss: 0.0832 - val_accuracy: 0.9664
    Epoch 41/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0920 - accuracy: 0.9658 - val_loss: 0.0821 - val_accuracy: 0.9698
    Epoch 42/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0947 - accuracy: 0.9641 - val_loss: 0.0767 - val_accuracy: 0.9724
    Epoch 43/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0925 - accuracy: 0.9650 - val_loss: 0.0761 - val_accuracy: 0.9735
    Epoch 44/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0929 - accuracy: 0.9638 - val_loss: 0.0756 - val_accuracy: 0.9751
    Epoch 45/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0952 - accuracy: 0.9641 - val_loss: 0.0827 - val_accuracy: 0.9704
    Epoch 46/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0918 - accuracy: 0.9657 - val_loss: 0.0804 - val_accuracy: 0.9731
    Epoch 47/50
    180/180 [==============================] - 1s 4ms/step - loss: 0.0914 - accuracy: 0.9654 - val_loss: 0.0772 - val_accuracy: 0.9733
    Epoch 48/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0910 - accuracy: 0.9663 - val_loss: 0.0870 - val_accuracy: 0.9685
    Epoch 49/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0922 - accuracy: 0.9643 - val_loss: 0.0782 - val_accuracy: 0.9721
    Epoch 50/50
    180/180 [==============================] - 1s 3ms/step - loss: 0.0947 - accuracy: 0.9630 - val_loss: 0.0798 - val_accuracy: 0.9689

Based on this, it appears that our model1 using only titles input performs well above base rate at approximately 96%, with very little overfitting. We can gain further insight into this by trying to visualize the training history.

```python
from matplotlib import pyplot as plt

plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![title_hist.png](/images/title_hist.png)

We can also plot the model as a graph to further understand our model.
```python
from tensorflow.keras import utils
utils.plot_model(model1)
```
![title_model.png](/images/title_model.png)

## 4. Building a model using only article text as input
We'll now move on to building a model using only article text as input. Similar to model1, we'll start by creating the input.
```python
text_input = Input(
    shape = (1,),
    name = "text",
    dtype = "string"
)
```
We then construct several layers for our model before constructing and compiling our model, much like with model1.
```python
text_input = Input(
    shape = (1,),
    name = "text",
    dtype = "string"
)
```
```python
model2 = keras.Model(
    inputs = text_input,
    outputs = output
)
```
```python
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```
Let's now train our model on the training data.
```python
history = model2.fit(train, 
                    validation_data=val,
                    epochs = 50
                    )
```

    Epoch 1/50
    

    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    

    180/180 [==============================] - 4s 17ms/step - loss: 0.6343 - accuracy: 0.6946 - val_loss: 0.4751 - val_accuracy: 0.9104
    Epoch 2/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.3270 - accuracy: 0.9150 - val_loss: 0.2162 - val_accuracy: 0.9504
    Epoch 3/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.1978 - accuracy: 0.9440 - val_loss: 0.1550 - val_accuracy: 0.9469
    Epoch 4/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.1497 - accuracy: 0.9604 - val_loss: 0.1277 - val_accuracy: 0.9689
    Epoch 5/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.1293 - accuracy: 0.9654 - val_loss: 0.1095 - val_accuracy: 0.9722
    Epoch 6/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.1116 - accuracy: 0.9687 - val_loss: 0.1017 - val_accuracy: 0.9728
    Epoch 7/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.1010 - accuracy: 0.9725 - val_loss: 0.0832 - val_accuracy: 0.9793
    Epoch 8/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0929 - accuracy: 0.9753 - val_loss: 0.0725 - val_accuracy: 0.9811
    Epoch 9/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0850 - accuracy: 0.9774 - val_loss: 0.0710 - val_accuracy: 0.9829
    Epoch 10/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0801 - accuracy: 0.9793 - val_loss: 0.0716 - val_accuracy: 0.9838
    Epoch 11/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0724 - accuracy: 0.9809 - val_loss: 0.0600 - val_accuracy: 0.9860
    Epoch 12/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0678 - accuracy: 0.9819 - val_loss: 0.0553 - val_accuracy: 0.9865
    Epoch 13/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0675 - accuracy: 0.9816 - val_loss: 0.0488 - val_accuracy: 0.9907
    Epoch 14/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0619 - accuracy: 0.9838 - val_loss: 0.0509 - val_accuracy: 0.9867
    Epoch 15/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0566 - accuracy: 0.9850 - val_loss: 0.0465 - val_accuracy: 0.9889
    Epoch 16/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0524 - accuracy: 0.9861 - val_loss: 0.0486 - val_accuracy: 0.9893
    Epoch 17/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0502 - accuracy: 0.9872 - val_loss: 0.0459 - val_accuracy: 0.9893
    Epoch 18/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0516 - accuracy: 0.9858 - val_loss: 0.0390 - val_accuracy: 0.9913
    Epoch 19/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0463 - accuracy: 0.9879 - val_loss: 0.0351 - val_accuracy: 0.9924
    Epoch 20/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0453 - accuracy: 0.9879 - val_loss: 0.0348 - val_accuracy: 0.9922
    Epoch 21/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0439 - accuracy: 0.9894 - val_loss: 0.0325 - val_accuracy: 0.9931
    Epoch 22/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0406 - accuracy: 0.9893 - val_loss: 0.0265 - val_accuracy: 0.9942
    Epoch 23/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0392 - accuracy: 0.9899 - val_loss: 0.0352 - val_accuracy: 0.9918
    Epoch 24/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0368 - accuracy: 0.9904 - val_loss: 0.0270 - val_accuracy: 0.9936
    Epoch 25/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0383 - accuracy: 0.9905 - val_loss: 0.0260 - val_accuracy: 0.9953
    Epoch 26/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0363 - accuracy: 0.9903 - val_loss: 0.0254 - val_accuracy: 0.9944
    Epoch 27/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0330 - accuracy: 0.9915 - val_loss: 0.0248 - val_accuracy: 0.9949
    Epoch 28/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0320 - accuracy: 0.9913 - val_loss: 0.0222 - val_accuracy: 0.9949
    Epoch 29/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0306 - accuracy: 0.9920 - val_loss: 0.0245 - val_accuracy: 0.9949
    Epoch 30/50
    180/180 [==============================] - 3s 15ms/step - loss: 0.0318 - accuracy: 0.9918 - val_loss: 0.0226 - val_accuracy: 0.9966
    Epoch 31/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0323 - accuracy: 0.9918 - val_loss: 0.0232 - val_accuracy: 0.9953
    Epoch 32/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0314 - accuracy: 0.9906 - val_loss: 0.0242 - val_accuracy: 0.9960
    Epoch 33/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0304 - accuracy: 0.9913 - val_loss: 0.0190 - val_accuracy: 0.9966
    Epoch 34/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0289 - accuracy: 0.9917 - val_loss: 0.0181 - val_accuracy: 0.9962
    Epoch 35/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0280 - accuracy: 0.9922 - val_loss: 0.0213 - val_accuracy: 0.9951
    Epoch 36/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0283 - accuracy: 0.9925 - val_loss: 0.0163 - val_accuracy: 0.9969
    Epoch 37/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0272 - accuracy: 0.9918 - val_loss: 0.0165 - val_accuracy: 0.9960
    Epoch 38/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0270 - accuracy: 0.9930 - val_loss: 0.0188 - val_accuracy: 0.9969
    Epoch 39/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0243 - accuracy: 0.9936 - val_loss: 0.0167 - val_accuracy: 0.9967
    Epoch 40/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0267 - accuracy: 0.9927 - val_loss: 0.0317 - val_accuracy: 0.9896
    Epoch 41/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0241 - accuracy: 0.9934 - val_loss: 0.0156 - val_accuracy: 0.9973
    Epoch 42/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0237 - accuracy: 0.9934 - val_loss: 0.0125 - val_accuracy: 0.9982
    Epoch 43/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0225 - accuracy: 0.9939 - val_loss: 0.0169 - val_accuracy: 0.9971
    Epoch 44/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0227 - accuracy: 0.9936 - val_loss: 0.0146 - val_accuracy: 0.9978
    Epoch 45/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0222 - accuracy: 0.9939 - val_loss: 0.0135 - val_accuracy: 0.9973
    Epoch 46/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0226 - accuracy: 0.9938 - val_loss: 0.0142 - val_accuracy: 0.9984
    Epoch 47/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0221 - accuracy: 0.9936 - val_loss: 0.0123 - val_accuracy: 0.9982
    Epoch 48/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0196 - accuracy: 0.9940 - val_loss: 0.0124 - val_accuracy: 0.9978
    Epoch 49/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0182 - accuracy: 0.9944 - val_loss: 0.0073 - val_accuracy: 0.9991
    Epoch 50/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0201 - accuracy: 0.9934 - val_loss: 0.0139 - val_accuracy: 0.9973
    
Based on this, it appears that model2 is performing at close to 100% accuracy, with minimal overfitting as well. We can take a look at our training history and a graph of our model.

```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![text_hist.png](/images/text_hist.png)
```python
from tensorflow.keras import utils
utils.plot_model(model2)
```
![text_model.png](/images/text_model.png)

## 5. Building a model using both article text and article titles
Now, let's try to build a model which uses both article text and title as input. This time, we'll start by concatenating the layers used in model1 and model2, and have the layers in these pre-existing models become part of our new model.

```python
main = layers.concatenate([titles_features, text_features], axis = 1)
main = layers.Dense(32, activation='relu')(main)
main = layers.Dense(32)(main)
output = layers.Dense(2, name="fake")(main) 
```
Similar to model1 and model2, we'll construct the model, compile it and train it on our training data.
```python
model3 = keras.Model(
    inputs = [titles_input, text_input],
    outputs = output
)
```
```python
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```
```python
history = model3.fit(train, 
                    validation_data=val,
                    epochs = 50
                    )
```

    Epoch 1/50
    180/180 [==============================] - 4s 18ms/step - loss: 0.0825 - accuracy: 0.9818 - val_loss: 0.0085 - val_accuracy: 0.9984
    Epoch 2/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0139 - accuracy: 0.9951 - val_loss: 0.0065 - val_accuracy: 0.9982
    Epoch 3/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0082 - accuracy: 0.9969 - val_loss: 0.0023 - val_accuracy: 0.9993
    Epoch 4/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0105 - accuracy: 0.9968 - val_loss: 0.0017 - val_accuracy: 1.0000
    Epoch 5/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0087 - accuracy: 0.9971 - val_loss: 0.0019 - val_accuracy: 0.9996
    Epoch 6/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0085 - accuracy: 0.9973 - val_loss: 0.0022 - val_accuracy: 0.9993
    Epoch 7/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0081 - accuracy: 0.9969 - val_loss: 7.4302e-04 - val_accuracy: 0.9998
    Epoch 8/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0072 - accuracy: 0.9972 - val_loss: 0.0013 - val_accuracy: 1.0000
    Epoch 9/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0080 - accuracy: 0.9976 - val_loss: 0.0014 - val_accuracy: 0.9998
    Epoch 10/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0055 - accuracy: 0.9982 - val_loss: 0.0021 - val_accuracy: 0.9996
    Epoch 11/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0058 - accuracy: 0.9980 - val_loss: 3.7496e-04 - val_accuracy: 1.0000
    Epoch 12/50
    180/180 [==============================] - 4s 21ms/step - loss: 0.0043 - accuracy: 0.9986 - val_loss: 3.3611e-04 - val_accuracy: 1.0000
    Epoch 13/50
    180/180 [==============================] - 4s 19ms/step - loss: 0.0068 - accuracy: 0.9974 - val_loss: 4.5032e-04 - val_accuracy: 1.0000
    Epoch 14/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0069 - accuracy: 0.9977 - val_loss: 0.0019 - val_accuracy: 0.9996
    Epoch 15/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0052 - accuracy: 0.9985 - val_loss: 9.0162e-04 - val_accuracy: 0.9998
    Epoch 16/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0052 - accuracy: 0.9986 - val_loss: 0.0020 - val_accuracy: 0.9996
    Epoch 17/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0059 - accuracy: 0.9981 - val_loss: 0.0010 - val_accuracy: 0.9998
    Epoch 18/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0043 - accuracy: 0.9985 - val_loss: 3.6893e-04 - val_accuracy: 1.0000
    Epoch 19/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0032 - accuracy: 0.9989 - val_loss: 7.6268e-04 - val_accuracy: 0.9998
    Epoch 20/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0059 - accuracy: 0.9983 - val_loss: 4.4604e-04 - val_accuracy: 1.0000
    Epoch 21/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0052 - accuracy: 0.9985 - val_loss: 2.8723e-04 - val_accuracy: 1.0000
    Epoch 22/50
    180/180 [==============================] - 3s 19ms/step - loss: 0.0047 - accuracy: 0.9983 - val_loss: 0.0018 - val_accuracy: 0.9996
    Epoch 23/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0049 - accuracy: 0.9984 - val_loss: 0.0026 - val_accuracy: 0.9993
    Epoch 24/50
    180/180 [==============================] - 5s 29ms/step - loss: 0.0071 - accuracy: 0.9978 - val_loss: 9.0306e-04 - val_accuracy: 0.9998
    Epoch 25/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0032 - accuracy: 0.9990 - val_loss: 9.2213e-05 - val_accuracy: 1.0000
    Epoch 26/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0051 - accuracy: 0.9981 - val_loss: 8.7897e-04 - val_accuracy: 0.9998
    Epoch 27/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 6.6564e-04 - val_accuracy: 1.0000
    Epoch 28/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0040 - accuracy: 0.9991 - val_loss: 2.7562e-04 - val_accuracy: 1.0000
    Epoch 29/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0046 - accuracy: 0.9984 - val_loss: 4.2067e-04 - val_accuracy: 1.0000
    Epoch 30/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0035 - accuracy: 0.9987 - val_loss: 2.5220e-04 - val_accuracy: 1.0000
    Epoch 31/50
    180/180 [==============================] - 3s 16ms/step - loss: 0.0047 - accuracy: 0.9984 - val_loss: 2.2577e-04 - val_accuracy: 1.0000
    Epoch 32/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0041 - accuracy: 0.9987 - val_loss: 4.5422e-04 - val_accuracy: 0.9998
    Epoch 33/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0043 - accuracy: 0.9986 - val_loss: 1.2040e-04 - val_accuracy: 1.0000
    Epoch 34/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0044 - accuracy: 0.9983 - val_loss: 1.6391e-04 - val_accuracy: 1.0000
    Epoch 35/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0043 - accuracy: 0.9984 - val_loss: 3.0221e-04 - val_accuracy: 1.0000
    Epoch 36/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0056 - accuracy: 0.9981 - val_loss: 1.4100e-04 - val_accuracy: 1.0000
    Epoch 37/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0029 - accuracy: 0.9989 - val_loss: 6.0845e-05 - val_accuracy: 1.0000
    Epoch 38/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0047 - accuracy: 0.9981 - val_loss: 1.0723e-04 - val_accuracy: 1.0000
    Epoch 39/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0045 - accuracy: 0.9979 - val_loss: 2.5744e-04 - val_accuracy: 1.0000
    Epoch 40/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0032 - accuracy: 0.9990 - val_loss: 4.4981e-04 - val_accuracy: 1.0000
    Epoch 41/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0024 - accuracy: 0.9991 - val_loss: 1.3138e-04 - val_accuracy: 1.0000
    Epoch 42/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0042 - accuracy: 0.9986 - val_loss: 1.1142e-04 - val_accuracy: 1.0000
    Epoch 43/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0024 - accuracy: 0.9991 - val_loss: 1.6318e-04 - val_accuracy: 1.0000
    Epoch 44/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0030 - accuracy: 0.9987 - val_loss: 1.5637e-04 - val_accuracy: 1.0000
    Epoch 45/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0023 - accuracy: 0.9990 - val_loss: 4.1086e-05 - val_accuracy: 1.0000
    Epoch 46/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0056 - accuracy: 0.9981 - val_loss: 2.6530e-04 - val_accuracy: 1.0000
    Epoch 47/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0039 - accuracy: 0.9987 - val_loss: 1.0181e-04 - val_accuracy: 1.0000
    Epoch 48/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0035 - accuracy: 0.9989 - val_loss: 3.3413e-04 - val_accuracy: 1.0000
    Epoch 49/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0019 - accuracy: 0.9993 - val_loss: 4.5285e-05 - val_accuracy: 1.0000
    Epoch 50/50
    180/180 [==============================] - 3s 17ms/step - loss: 0.0029 - accuracy: 0.9989 - val_loss: 7.2835e-05 - val_accuracy: 1.0000
    
Model3 performs at 100% accuracy, with minimal overfitting. Let's take a look at our training history and the graph of our model.
```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![both_hist.png](/images/both_hist.png)

```python
utils.plot_model(model3)
```
![both_model.png](/images/both_model.png)

## 6. Testing our model on unseen data
Now we are on to the final evaluation of our model, to test it against unseen data. We'll start by loading in the test data.
```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
df2 = pd.read_csv(test_url, index_col = 0)
df2 = make_dataset(df2)
```
Let's evaluate our model's performance on this new data.
```python
model3.evaluate(df2)
```

    225/225 [==============================] - 2s 7ms/step - loss: 0.1171 - accuracy: 0.9811
    




    [0.11706098914146423, 0.9810681939125061]

Our model is able to reach a 98% accuracy rate on the data!

## 7. Visualizing the embedding
Before we wrap things up, let's take a look at the learned word embedding used in our model. 
```python
weights = model3.get_layer('embedding').get_weights()[0] # gets the weights from the embedding layer
vocab = title_vectorize_layer.get_vocabulary()           # gets the vocabulary from article titles

# reduce the number of dimensions in weights
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

# construct a df with words and its corresponding weights
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```





  <div id="df-23c19791-70ce-468e-be86-c6ee4adf37f5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>x0</th>
      <th>x1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>-0.055221</td>
      <td>0.063784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[UNK]</td>
      <td>0.028813</td>
      <td>0.087702</td>
    </tr>
    <tr>
      <th>2</th>
      <td>trump</td>
      <td>-0.003126</td>
      <td>-0.052830</td>
    </tr>
    <tr>
      <th>3</th>
      <td>video</td>
      <td>-2.338849</td>
      <td>-1.493234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>says</td>
      <td>0.224585</td>
      <td>1.261130</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>jesse</td>
      <td>-1.443121</td>
      <td>-0.604883</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>invites</td>
      <td>1.180338</td>
      <td>0.147431</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>indian</td>
      <td>0.415695</td>
      <td>1.117235</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>hostage</td>
      <td>0.211893</td>
      <td>-0.504790</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>hopeful</td>
      <td>-0.465873</td>
      <td>1.111238</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-23c19791-70ce-468e-be86-c6ee4adf37f5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-23c19791-70ce-468e-be86-c6ee4adf37f5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-23c19791-70ce-468e-be86-c6ee4adf37f5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





Let's use plotly to create a visualization of the embedding.
```python
import plotly.express as px 
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                # size_max = 2,
                 hover_name = "word")

fig.show()
```
{% include embedding.html %}

The words "rohingya" (-0.148349,2.392929) and "myanmar" (0.5294677,2.076982) are located close to each other on this visualization, which can be explained by the extensive media coverage of the Rohingya refugee crisis in Myanmar beginning in 2015. The words "feds" (-0.4883741,-1.742767) and "terrorists" (-0.5416973) are also located close to each other, which can be explained by the fact that the FBI (colloquially known as "the Feds") is the law enforcement agency in charge of counterterrorism efforts and that any headline regarding terrorism is bound to mention them. On the other hand, we can explain the proximity between "denies" (2.151047,0.710892) and "allegations" (2.535659,0.6700212) in terms of linguistic convention, as the two words often go together in news reporting e.g. xx denies the allegations.