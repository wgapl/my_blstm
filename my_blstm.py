#! /usr/bin/env python
"""
file: my_blstm.py
author: thomas wood (thomas@wgapl.com)
description: Quick and dirty Bi-directional LSTM layer in Python.
"""
import numpy as np
from numpy import tanh
from numpy.random import random
from string import printable
from my_lstm import gen_bag_hashtable, \
    make_wordvector, \
    make_string, \
    LSTMLayer

def activation(z, method="tanh"):
    """
    Defaults to "tanh".
    Probably shouldn't ever neglect to use that, but whatever.
    """
    if method == "tanh":
        return tanh(z)
    elif method == "linear":
        return z
    elif method == "sigmoid":
        return 1./(1.+np.exp(-z))


class BLSTMLayer:
    def __init__(self, n_in, n_hidden, n_out, params, eps):
        """
        The number of parameters in a single LSTM layer is
        n_lstm =
        4*n_in*n_hidden  # four W_{i,c,f,o} input weight matrices
        + 4*n_hidden**2  # four U_* recurrent weight matrcies
        + 4*n_hidden     # four b_* bias vectors
        + n_hidden**2    # one V_o matrix of weights

        We use two matrices of size n_hidden*n_out along with
        a bias vector of size n_out to compute the output of
        the BLSTMLayer for a given input sequence, so the total
        number of parameters is

        n_total_params = 2*n_lstm + 2*n_hidden*n_out + n_out
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        n_lstm = 4*n_in*n_hidden + \
            4*n_hidden**2 + \
            4*n_hidden + \
            n_hidden**2


        # slice 'em and dice 'em
        ind_fwd = n_lstm # forward parameter index
        self.forward_params = params[:ind_fwd] # don't reshape
        ind_back = ind_fwd + n_lstm # backward parameter index
        self.backward_params = params[ind_fwd:ind_back] # don't reshape
        ind_W = ind_back + 2*n_hidden*n_out # Output weights
        self.W = params[ind_back:ind_W].reshape((n_out, 2*n_hidden))
        self.bias = params[ind_W:] # output bias

    def gen_sequence(self, X):
        n_in = self.n_in
        n_hidden = self.n_hidden
        n_out = self.n_out
        T = X.shape[1] # size of input sequence

        # big matrix of forward and backward hidden state values
        # We are going to use two LSTMs to populate this matrix
        H = np.zeros((2*n_hidden, T))


        # a single LSTMLayer for stepping forward
        # TODO!!!will look into multiple lstm layering inside
        # the bidirectional framework in a minute, but first
        # things first... !!!
        lstmFwd = LSTMLayer(n_in, n_hidden, self.forward_params, eps=0.0)
        # an LSTMLayer for stepping backward
        lstmBack = LSTMLayer(n_in, n_hidden, self.backward_params, eps=0.0)

        for k in range(T):
            # FORWARD: calculate forward hidden state values
            H[:n_hidden,k] = lstmFwd.step(X[:,k])
            # BACKWARD: calculate backward hidden state values
            H[n_hidden:,k] = lstmBack.step(X[:,T-1-k])

        return activation(np.dot(self.W, H), method="linear")

def rudimentary_test():
    s = """0 a is the quick fox who jumped over the lazy brown dog's new sentence."""
    table = gen_bag_hashtable()

    v = make_wordvector(s, table)

    n_in, T = v.shape
    n_out = n_in
    n_hidden = 100 # Learn a more complex representation?
    eps1 = 0.00001
    eps2 = 0.001

    n_lstm = 4*n_in*n_hidden + \
        4*n_hidden**2 + \
        4*n_hidden + \
        n_hidden**2

    n_params = 2*n_lstm + 2*n_hidden*n_out + n_out

    params1 = eps1*(2*random(n_params,)-1.)

    blstm = BLSTMLayer(n_in, n_hidden, n_out, params1, eps2)

    y1 = blstm.gen_sequence(v)

    s1 = make_string(y1)
    print s1

if __name__ == "__main__":
    rudimentary_test()
