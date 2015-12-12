__author__ = 'yutao'

import numpy
import theano
import theano.tensor as T
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        :param rng: Random number generator, for reproducable results
        :param input: Symbolic Theano variable for the input
        :param n_in: Number of incoming units
        :param n_out: Number of outgoing units
        :param W: Weight matrix
        :param b: Bias
        :param activation: Activation function to use
        """
        self.input = input
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.activation=activation


        if W is None: #Initialize Glorot Style
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid or activation == theano.tensor.nnet.hard_sigmoid or activation == theano.tensor.nnet.ultra_fast_sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')

        if b is None: #Initialize bias to zero
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        # Put your code here: Implement a function to compute activation(x*W+b)
        out = T.dot(input, self.W) + self.b

        if activation is None:
            self.output = out
        else:
            self.output = activation(out)

        self.params = [self.W, self.b]


import numpy
import theano
import theano.tensor as T


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                dtype=theano.config.floatX), name='b')

        # Put your code here, implement a function to compute softmax(x*W+b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.pred_y = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


import numpy
import theano
import theano.tensor as T

class MLP(object):
     def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        :param rng: Our random number generator
        :param input: Input variable (the data)
        :param n_in: Input dimension
        :param n_hidden: Hidden size
        :param n_out: Output size
        """
        #Put your code here to build the neural network
        self.hiddenLayer = HiddenLayer(rng, input, n_in, n_hidden, activation=T.tanh)
        self.softmaxLayer = SoftmaxLayer(self.hiddenLayer.output, n_hidden, n_out)

        self.negative_log_likelihood = self.softmaxLayer.negative_log_likelihood
        self.params = self.hiddenLayer.params + self.softmaxLayer.params


import pickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T


# Load the pickle file for the MNIST dataset.
dataset = 'data/mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, dev_set, test_set = pickle.load(f)
f.close()

#train_set contains 2 entries, first the X values, second the Y values
train_x, train_y = train_set
dev_x, dev_y = dev_set
test_x, test_y = test_set

#Created shared variables for these sets (for performance reasons)
train_x_shared = theano.shared(value=np.asarray(train_x, dtype='float32'), name='train_x')
train_y_shared = theano.shared(value=np.asarray(train_y, dtype='int32'), name='train_y')


print("Shape of train_x-Matrix: ",train_x_shared.get_value().shape)
print("Shape of train_y-vector: ",train_y_shared.get_value().shape)
print("Shape of dev_x-Matrix: ",dev_x.shape)
print("Shape of test_x-Matrix: ",test_x.shape)

###########################
#
# Start to build the model
#
###########################

# Hyper parameters
hidden_units = 50
learning_rate = 0.01
batch_size = 20

# Put your code here to build the training and predict_labels function
index = T.lscalar()
x = T.fmatrix('x')
y = T.ivector('y')
rng = numpy.random.RandomState(10)

classifer = MLP(rng, input=x, n_in=28*28, n_hidden=hidden_units, n_out=10)

cost = classifer.negative_log_likelihood(y)

updates = [(param, param - learning_rate * T.grad(cost, param)) for param in classifer.params]

train_model = theano.function(
    inputs = [index],
    outputs = cost,
    updates = updates,
    givens = {
        x: train_x_shared[index * batch_size: (index+1) * batch_size],
        y: train_y_shared[index * batch_size: (index+1) * batch_size]
    }
)

predict_labels = theano.function(inputs = [x], outputs=classifer.softmaxLayer.pred_y)

print(">> train- and predict-functions are compiled <<")