import lasagne
from theano import sparse
import numpy as np
import theano
import theano.tensor as T

EXP_SOFTMAX = True

class SparseLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_units, W = lasagne.init.GlorotUniform(), b = lasagne.init.Constant(0.), nonlinearity = lasagne.nonlinearities.rectify, **kwargs):
        super(SparseLayer, self).__init__(incoming, **kwargs)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_for(self, input, **kwargs):
        act = sparse.basic.structured_dot(input, self.W)
        if self.b is not None:
            act += self.b.dimshuffle('x', 0)
        if not EXP_SOFTMAX or self.nonlinearity != lasagne.nonlinearities.softmax:
            return self.nonlinearity(act)
        else:
            return T.exp(act) / (T.exp(act).sum(1, keepdims = True))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)