"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from linear_regression import LinearRegression, NNRegression
#from mlp import HiddenLayer
#from dA import dA


class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.1], fine_tune_type="nonlinear"):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.autoencoder_params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        
        # Modified By Amir: The label are presented as a vector of [floats] , each label has four values
        # X, Y, sin(angle), cos(angle)
        self.y = T.matrix('y')
        # This perhaps implied the use of sigmoid instead of logistic regression in the final layer of NN
        # self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
        # An effort to separate encoder weights and bias from the fine-tuning layer
        self.autoencoder_params = self.params
        
        # We now need to add a logistic layer on top of the MLP
        #self.logLayer = LogisticRegression(
        #                 input=self.sigmoid_layers[-1].output,
        #                 n_in=hidden_layers_sizes[-1], n_out=n_outs)
        
        # Added by Amir to replace the logistic layer with a linear layer for regression
        
        #TODO: Test with a sigmoid layer before the linear regression
        '''
        self.top_sigmoid_layer =  HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
        '''
        if fine_tune_type == "linear":
            self.top_linear_layer = LinearRegression(
                             input_data=self.sigmoid_layers[-1].output,
                             n_in=hidden_layers_sizes[-1], n_out=n_outs)
        else:
            self.top_linear_layer = NNRegression(
                             input_data=self.sigmoid_layers[-1].output,
                             n_in=hidden_layers_sizes[-1], hidden_layer = 50, n_out=n_outs)
        

        #self.params.extend(self.logLayer.params)
        self.params.extend(self.top_linear_layer.params)
        # construct a function that implements one step of fine tunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        #self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        
        self.finetune_cost = self.top_linear_layer.mse(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.top_linear_layer.error(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
              name='train')

        test_score_i = theano.function([index], self.finetune_cost,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]},
                      name='test')

        valid_score_i = theano.function([index], self.finetune_cost,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]},
                      name='valid')

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    # Give the output of the autoencoder minus the fine-tuning layer
    # TODO: directly use the final layer output instead of manually multiplying weights    
    def encode(self, input):
        i = 0
        while (i < self.n_layers * 2):
            W = self.params[i]
            b = self.params[i + 1]
            input = T.nnet.sigmoid(T.dot(input, W) + b)
            i += 2
            
        return input
    
    # Give the reconstructed input from the encoded result, and assign zero to bias when dimensions don't match. 
    # WARNING: It jumps over fine-tuning layer by popping parameters. 
    # TODO: directly use the final layer output instead of manually multiplying weights
    def decode(self, input):
        i = 0
        params = list(reversed(self.params))
        params.pop(0)
        params.pop(0)
        while (i < len(params)):
            b = params[i]
            W = T.transpose(params[i + 1])
            
            if i == len(params) - 2 or b.shape[0] != params[i + 3].shape[0]:
                input = T.nnet.sigmoid(T.dot(input, W))
            else:
                input = T.nnet.sigmoid(T.dot(input, W) + b)
            i += 2
        return input
    # Reconstructs the input
    def get_reconstructed_input(self, input):
        enc = self.encode(input)
        return self.decode(enc)

if __name__ == '__main__':
    reading_path = '/home/rik/nav-data/june-2014/2500/'
    path = '/home/rik/nav-data'
    model_path = '/home/rik/trainedModels/'
    #model = '/home/rik/nav-data/june-2014w/compressed/model'
    #model = '/home/rik/nav-data/june-2014w/compressed/model2014-07-13_09:57:06'
    #model = '/home/rik/nav-data/june-2014w/compressed/model2014-07-10_11:28:41'

    result = open(os.path.join(path, 'evaluation_results'), 'wb')
    
    f_test = open(os.path.join(path, 'test_set'), 'rb')
    test_set, _ = cPickle.load(f_test)
    
    for files in os.listdir(model_path):
        print "Evaluating Model, ", files
        if os.path.splitext(files)[1] == ".txt" or os.path.splitext(files)[1] == ".txt~":
            continue
        
        f = open(os.path.join(model_path, files), 'rb')
        model = cPickle.load(f)
        model_name = os.path.basename(files)
        
        x = T.fmatrix('x')
        out = model.get_reconstructed_input(x)
        #out = model.encode(x)
        get_f = theano.function([x], out)
        
        
        test_set = numpy.asarray(test_set, dtype = numpy.float32)
        test_set /= 255.
        error = 0
        for row_idx in range(test_set.shape[0]):
            feature = test_set[row_idx,:]
            feature = numpy.reshape(feature, (1, 28 * 28))
            reconstruct = get_f(feature)
            
            error += numpy.mean((feature - reconstruct) ** 2)
    
        error /= test_set.shape[0]
        print error
        cPickle.dump((model_name, error), result)
    
    

