'''
Created on 1 jul. 2014

@author: p257689
'''
import cPickle
import os
from os.path import join
import sys
import time

import numpy

import theano
import theano.tensor as T
from mlp import HiddenLayer
from copy import deepcopy

#from rprop import rprop_plus_updates
#from rprop import irprop_minus_updates

import matplotlib.pyplot as plt

#!/usr/bin/env python
import matplotlib.pyplot as plt
import time
import cPickle, sys
import math, numpy, cv2

import os
import Queue

class NN(object):
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        
        self.w = numpy.asarray(numpy.random.RandomState().uniform(
                    low=100,
                    high=100,
                    size=(self.n_input, self.n_output)), dtype=numpy.float64)
        self.b = numpy.zeros((self.n_output,1))
        
        self.learning_rate = 0.2
        
        self.batch_update = []
        self.batch_data = []
        self.batch_labels = []
        self.replay_data = []
        
    
    def out(self, input):
        return numpy.dot(input, self.w)
    
    def mse(self, input, target):
        return numpy.mean(target - (self.out(input)) ** 2)
    
    def online_update(self, input, target):
        #input = numpy.reshape(input.size1)
        delta_ey = (target - self.out(input))
        new_w = self.w + self.learning_rate * numpy.dot(input.T, delta_ey)
        self.w = new_w
        
    def batch(self, input, target):
        delta_ey = (target - self.out(input))
        self.batch_update.append(self.learning_rate * numpy.dot(input.T, delta_ey))
        self.batch_data.append(input)
        self.batch_labels.append(target)
        
    def update(self):
        
        data = zip(self.batch_data, self.batch_labels)
        for i in range(50):
            update = [self.learning_rate * numpy.dot(input.T, target - self.out(input)) for input, target in data]
            updates = numpy.asarray(self.batch_update)
            self.w += numpy.mean(updates, axis = 0)
        self.batch_update = []
        self.batch_data = []
        self.batch_labels = []
    
class NNRegression(object):
    '''
    classdocs
    '''


    def __init__(self, input_data, n_in, hidden_layer= 100, n_out = 4, weights = None,act_func = T.nnet.sigmoid, filename = None):
        print "Linear_Regression: From RL_METHODS"
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.input = input_data
        
        
        W1 = b1 = W2 = b2 = None
        print weights
        if weights != None:
            try:
                W1, b1 = weights
            except Exception as e:
                W1 = weights[0]
        
        #self.linearRegression = LinearRegression(self.sigmoid_layer.output, n_in = hidden_layer, n_out = n_out)
        numpy_rng = numpy.random.RandomState()
        self.linearRegression = HiddenLayer(rng=numpy_rng,
                                        input= self.input,
                                        n_in= n_in,
                                        n_out = n_out,
                                        W_values = W1,
                                        b_values = b1,
                                        activation=act_func)
        '''
        self.linearRegression2 = HiddenLayer(rng=numpy_rng,
                                        input= self.linearRegression.output,
                                        n_in= hidden_layer,
                                        n_out = n_out,
                                        W_values = W2,
                                        b_values = b2, 
                                        activation=None)
        '''
        self.L1 = abs(self.linearRegression.W).sum()
            

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = T.mean(self.linearRegression.W ** 2)

                    
        self.params = self.linearRegression.params
        
        self.output = self.linearRegression.output
        self.cost = self.linearRegression.mse
        
        
        
        if filename != None:
            self.load(filename)
            print "Network Loaded from %s" % (filename)
    #def output(self, x):
    #    l1 = T.nnet.sigmoid(T.dot(x, self.sigmoid_layer.W) + self.sigmoid_layer.b)
    #    return  T.dot(l1, self.linearRegression.W) + self.linearRegression.b
        
    #def mse(self, y):
    #    return T.mean(T.sum((self.output - y) ** 2, axis = 1)) 
    
    #def error(self, y):
    #    return T.sum((self.output - y) ** 2, axis = 1)
        
    def online_cost(self, x, y):
        out = T.dot(x, self.linearRegression.W)
        return T.mean((y - out) ** 2)
    
    def batch_cost(self, x, y):
        out = T.dot(x, self.params[0])
        return T.mean((out - y) ** 2)
    
    
    
def load_data():
    path = '/media/Datastation/nav-data-late14/BACKUPSTUFF'
    training_x = numpy.load(join(path, 'trainingEncOut_set.npy'))
    training_y = numpy.load(join(path, 'trainingEncOut_set_labels.npy'))
    
    val_input = numpy.load(join(path, 'validationEncOut_set.npy'))
    val_label = numpy.load(join(path, 'validationEncOut_set_labels.npy'))
    
    test_input = numpy.load(join(path, 'testingEncOut_set.npy'))
    test_label = numpy.load(join(path, 'testingEncOut_set_labels.npy'))
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y
 
    training_x, training_y = shared_dataset((training_x, training_y))
    val_input, val_label = shared_dataset((val_input, val_label))
    test_input, test_label = shared_dataset((test_input, test_label))
    return [(training_x, training_y), (val_input, val_label), (test_input, test_label)]
    

def test2():
    batch_size = 1
    test = numpy.asarray([0., 1., 1.,0.], dtype = theano.config.floatX)  # @UndefinedVariable
    print test.shape
    data = theano.shared(numpy.asarray([(0.,0.), (0.,1.), (1.,0.), (1.,1.)], dtype = theano.config.floatX))  # @UndefinedVariable
    label = theano.shared(numpy.asarray([(0.,1.), (1.,0.), (1.,0.),(0.,1.)], dtype = theano.config.floatX))  # @UndefinedVariable
    
    print data.get_value(borrow=True).shape
    print label.get_value(borrow=True).shape   
    index = T.lscalar()  # index to a [mini]batch
    x = T.fmatrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
    d = T.fmatrix('d')                   # [int] labels

    # construct the MLP class
    regressor = NNRegression(input_data=x, n_in=2,
                     hidden_layer=2, n_out=2)


    cost = regressor.linearRegression.mse(y)
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=regressor.linearRegression.error(y),
            givens={
                x: data[index * batch_size:(index + 1) * batch_size],
                y: label[index * batch_size:(index + 1) * batch_size]})
    
    validate_model = theano.function(inputs=[index],
            outputs=regressor.linearRegression.error(y),
            givens={
                x: data[index * batch_size:(index + 1) * batch_size],
                y: label[index * batch_size:(index + 1) * batch_size]})
    '''classifier
    output = theano.function(inputs=[index],
            outputs=regressor.linearRegression.output,
            givens={
                x: data[index * batch_size:(index + 1) * batch_size]})
    '''
    output = theano.function([x],
            outputs=regressor.output,
            allow_input_downcast=True)
    
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    
    for param in regressor.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    print "Gparams ", gparams
    print "Regressor Param Length: ", len(regressor.params)
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, gparam in zip(regressor.params, gparams):
        updates.append((param, param - 0.1 * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: data[index * batch_size:(index + 1) * batch_size],
                y: label[index * batch_size:(index + 1) * batch_size]})
    
    n_train_batches = data.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches =data.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = data.get_value(borrow=True).shape[0] / batch_size
    
    patience = 1000000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.999  # a relative improvement of this much is
                                    # considered significant
    validation_frequency = n_train_batches * 10
                                    # go through this many
                                    # minibatche before checking the network
                                    # on the validation set; in this case we
                                    # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    for j in range(len(regressor.params)):
        print regressor.params[j].get_value(), ", ",
    print " "
    while (epoch < 100000) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if this_validation_loss < 0.01:
                    for j in range(n_train_batches):
                        
                        print "for input ", data[j].eval(), ", output is:", output(numpy.reshape(data[j].eval(), (1,2)))
                    #for j in range(len(regressor.params)):
                    #    print regressor.params[j].get_value(), ", ",
                    print ""
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
    
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            #if patience <= iter:
            #        done_looping = True
            #        break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
if __name__ == '__main__':
    test2()
    #test_function()
