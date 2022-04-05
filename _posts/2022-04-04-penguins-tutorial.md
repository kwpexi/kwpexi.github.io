---
layout: post
title: Create a visualization from Palmer Penguins
---
We will go through the steps for creating a visualization from the Palmer Penguins dataset.

## Palmer Penguins dataset

We begin by importing the dataset.

```
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```
This assigns the contents of the palmer_penguins.csv to a pandas dataframe named penguins.

Let's take a look at what's inside penguins.

```
penguins.head()
```
python

## Data preprocessing

Before we do move on, let's shorten the species names to make things a little neater:

```
penguins["Species"] = penguins["Species"].str.split().str.get(0)
```
Now, a given penguin's species will be known by either Adelie, Gentoo or Chinstrap.

## Visualizing Palmer Penguins

Let's zoom in on two variables: the culmen length and the flipper length of any given penguin. I want to create a scatterplot of the two variables.
We can start by importing the necessary modules.

```
python
from matplotlib import pyplot as plt
import seaborn as sns
```

We first create a plot using `plt.subplots()` before using `sns.scatterplot()` to plot our scatterplot. We then use `.set()` to create a title for our visualization.

```
fig, ax = plt.subplots(figsize=(6,5))

viz = sns.scatterplot(x="Culmen Length (mm)", 
                      y="Flipper Length (mm)",
                      hue="Species",
                      palette="magma",
                      data=penguins)

viz.set(title = "Flipper length against culmen length by species")
```
![Flipper-culmen.png](/images/Flipper-culmen.png)



