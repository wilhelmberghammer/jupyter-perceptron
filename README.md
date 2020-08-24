# Neural Network (Perceptron) using NumPy
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QLvum2WgpK6Kv65XR77-BjethDvRueXH?usp=sharing)

This is an implementation of a neural network, with no hidden layers, just using [NumPy](https://numpy.org/). I made this project mainly to learn.

The NN is theoreticly capable of working with as many inputs as needed. The only thing that needs to be changed is the number of inputs (=number of weights).

![NN.png](https://github.com/wilhelmberghammer/np-neural-network/blob/master/readme_recources/NN.png?raw=true)

## Imported Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import random
```

## Data

This is the data I used to train and test the neural net. This data does not mean anything, every datapoint (`[0.5, 0.5, 0.5]` for example) represents a point in a three dimensional coordinate system.

The labels are either  `1` or `0` .

```python
data = [[0.5, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [1.0, 1.5, 1.5],
        [1.5, 1.0, 1.5],
        [1.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.5, 1.0, 1.5],
        [2.0, 1.5, 1.5],
        [2.5, 2.0, 1.5],
        [3.0, 0.5, 2.5],
        [2.5, 3.0, 3.0],
        [2.5, 3.5, 3.0],
        [3.0, 4.0, 3.5],
        [3.5, 1.5, 3.0],
        [3.5, 2.5, 3.5],
        [3.5, 3.5, 3.0],
        [4.0, 2.0, 5.0],
        [4.0, 3.0, 5.0],
        [4.5, 2.5, 4.0],
        [4.5, 3.5, 3.5]]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

test_data = [[4.5, 3.0, 4.0],
             [4.0, 3.5, 4.0],
             [4.0, 4.0, 4.0],
             [3.0, 3.0, 3.0],
             [3.5, 3.0, 4.0],
             [4.0, 2.5, 4.0],
             [1.0, 1.0, 1.0],
             [.25, 0.5, 3.0],
             [1.0, 0.5, 2.0],
             [1.5, 0.5, 2.0],
             [2.0, 1.0, 1.0],
             [2.5, 1.5, 4.0],]

test_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
```

This data can be plotted using `matplotlib`.

The green dots are data points with the label `0`. The red dots are data points with the label `1`.

![download.png](https://github.com/wilhelmberghammer/np-neural-network/blob/master/readme_recources/download.png?raw=true)

## Activation function

The labels (and the output) are between `0` and `1`. Therefore well use sigmoid as an activation function.

We'll also need the derivative of the sigmoid function.

```python
# Sigmoid activation function
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

# derivative of the sigmoid function
def d_sigmoid(x):
    y = sigmoid(x)*(1-sigmoid(x))
    return y
```

## Other functions

Calculating the average of a list:

```python
# Average of a list
def avg(list):
    sumOfNumbers = 0
    for t in list:
        sumOfNumbers = sumOfNumbers + t

    avg = sumOfNumbers / len(list)
    return avg
```

Plotting the loss:

```python
# Plot fuction to plot the loss
def plt_loss(loss):
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.grid()
    axes = plt.gca()
    #axes.set_ylim([0, 1])
    plt.plot(loss)
```

Calculating loss:

```python
def val_loss():
    val_loss_list = []
    val_loss = 0
    for i in range(len(test_data)):
        z = np.dot(w, test_data[i]) + b
        a = sigmoid(z)
        y = test_labels[i]
        c = np.square(a - y)
        val_loss_list.append(c)
    val_loss = avg(val_loss_list)
    return val_loss
```

## Back propagation

This is the main reason I started this project. I wanted to implement back propagation from scratch.

```python
def back_prop(w, b, iters, lr, batch_loss_size):
    w.shape = (num_weights,)
    loss = []
    batch_loss = []
		
    for i in range(iters):
        data_index = np.random.randint(len(data))
        data_point = data[data_index]
        label = labels[data_index]

        z = np.dot(w, data_point) + b
        a = sigmoid(z)
        y = label
        c = np.square(a - y)

        # derivatives
        dc_dw = []
        for j in range(len(data_point)):
            dc_dw.append(data_point[j]*d_sigmoid(z)*2*(a-y))
        dc_db = d_sigmoid(z)*2*(a-y)

        # change weights
        for k in range(len(w)):
            w[k] = w[k] - lr * dc_dw[k]
        b = b - lr * dc_db

        # to track loss and draw the loss graph
        batch_loss.append(c)
        if len(batch_loss) == batch_loss_size:
            loss.append(avg(batch_loss))
            batch_loss = []

    return loss, w, b, able_w
```

Assigning values to the parameters and running `back_prop()`
```python
# Number of weights (=number of imputs)
num_weights = 3
w = np.random.randn(num_weights,)
w = np.array(w)
b = [np.random.randn()]

# Hyperparameters Back.prop.
# interations:batch_size_loss >= 100 
iterations = 25000
learning_rate = .1
# has no inpact on the training - just to plot the loss graph better
batch_size_loss = 100

loss, w, b = back_prop(w, b, iterations, learning_rate, batch_size_loss)

plt_loss(loss)
print('Validation Loss: ', val_loss())
```

*Validation Loss: [0.07850451]*

*Loss:*

![loss.png](https://github.com/wilhelmberghammer/np-neural-network/blob/master/readme_recources/loss.png?raw=true)

![result1](https://github.com/wilhelmberghammer/np-neural-network/blob/master/readme_recources/result.png?raw=true)

![result2](https://github.com/wilhelmberghammer/np-neural-network/blob/master/readme_recources/result1.png?raw=true)
