"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, shuffle_rnn, conlleval, check_dir
from hw4_nn import myMLP, train_nn

from numpy import random
import copy

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(1500)

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.zeros((X.shape[1],X.shape[0]), dtype=numpy.int32)
    for i in range(X.shape[1]):
        Y[i] = numpy.sum(X[:,0:(i+1)],axis=1)
    Y = numpy.mod(numpy.transpose(Y), 2)
    
    return X, Y


#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit=8, n_epochs=5000, batch_size=10, n_hiddenLayers=2, n_hidden=[100, 100], learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000) #1000
    valid_set = gen_parity_pair(n_bit, 500)  #500
    test_set  = gen_parity_pair(n_bit, 100)  #100

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size  
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_out=2,
        n_hiddenLayers=n_hiddenLayers
    )
 
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    
    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    print('... training the model')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True

    while (epoch < n_epochs) and (not done_looping):
        seed = random.randint(0, sys.maxint)
        shuffle(train_set_x, seed)
        
        train_model = theano.function(
            inputs=[index],
            outputs=[cost],                      
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
        
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
  
                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))
                    
                    if test_score == 0:
                        done_looping = True
                        break

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    
#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, de, cs, normal=True, normal_layer=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        
        ###########################################################  add bias
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        

        self.G = theano.shared(name='G',
                               value=numpy.zeros(nh,
                               dtype=theano.config.floatX))
        
        # bundle
        self.params = [self.emb, self.wx, self.wh, self.w, self.h0, self.bh, self.b, self.G]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels


        def recurrence_normal(x_t, h_tm1):
            A = T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh)
            A_mean = T.mean(A)
            A_std  = T.std(A)
            A_normal = (self.G/A_std) * (A-A_mean + self.bh)
            h_t = T.nnet.sigmoid(A_normal)                   #######################################3
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)   ########################3
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        if normal_layer:
            [h, s], _ = theano.scan(fn=recurrence_normal,
                                    sequences=x,
                                    outputs_info=[self.h0, None],
                                    n_steps=x.shape[0])
        else:
            [h, s], _ = theano.scan(fn=recurrence,
                                    sequences=x,
                                    outputs_info=[self.h0, None],
                                    n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        error = self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()
        return error

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))            
                
    def get_param(self):
        return self.params

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(n_bit=8, nh=200, de=50, n_epoch=1000, learning_rate=0.1, normal_layer=True, decay_period=10):

    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    train_set_x = train_set[0]
    train_set_y = train_set[1]
    valid_set_x = valid_set[0]
    valid_set_y = valid_set[1]
    test_set_x  = test_set[0]
    test_set_y  = test_set[1]
    
    
    ''' T
    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    '''
    
    nc=2
    ne=2
    cs=1

    # instanciate the model
    seed = random.randint(0, 4294967295)
    numpy.random.seed(seed)
    random.seed(seed)
    
    print('... building the model')
    rnn = RNN(
        nh=nh,
        nc=nc,
        ne=ne,
        de=de,
        cs=cs,
        normal=False,
        normal_layer=normal_layer)

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    decay=True
    verbose=True
    decay_cnt=0
    
    for e in range(n_epoch):
        print("========== epoch %i ==========") % (e)
        print("===== learning_rate: %f =====") % (learning_rate)
        
        '''
        # shuffle
        seed = random.randint(0, 4294967295)
        shuffle_rnn(train_set_x, seed)
        '''
        
        current_epoch = e
        tic = timeit.default_timer()


        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            error = rnn.train(x, y, cs, learning_rate)
        print(error)
                    
             
        '''    
        ############################################################33
        A = rnn.get_param()
        print("A")
        print("self.emb")
        print(type(A[0].get_value()))
        print(A[0].get_value())
        
        print("self.wx")
        print(type(A[1].get_value()))
        print(A[1].get_value())  
        
        print("self.wh")
        print(type(A[2].get_value()))
        print(A[2].get_value())
        
        print("self.w")
        print(type(A[3].get_value()))
        print(A[3].get_value())
        
        print("self.h0")
        print(type(A[4].get_value()))
        print(A[4].get_value())
        raw_input()
        ''' 
            
        print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic))
        sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [rnn.classify(numpy.asarray(
                            contextwin(x, cs)).astype('int32'))
                            for x in test_set_x]       ################T .get_value()
        predictions_valid = [rnn.classify(numpy.asarray(
                             contextwin(x, cs)).astype('int32'))
                             for x in valid_set_x]      ###############T  .get_value()
        '''
        for x in test_set_x.get_value():
            print("x")
            print(type(x))
            print(x)
            
            print("numpy.asarray(contextwin(x, cs)).astype('int32')")
            print(type(numpy.asarray(contextwin(x, cs)).astype('int32')))
            print(numpy.asarray(contextwin(x, cs)).astype('int32'))
            
            print("rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32'))")
            print(type(rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32'))))
            print(rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32')))
            
            raw_input()
        
        print("predictions_test")
        print(type(predictions_test))
        print(predictions_test)
        
        print
        print("test_set_y.eval()")
        print(type(test_set_y.eval()))
        print(test_set_y.eval())
        '''
        
        
        # evaluation // compute the accuracy using conlleval.pl
        res_test = 0.0
        for i in range(len(predictions_test)):
            if(predictions_test[i][n_bit-1]==test_set_y[i][n_bit-1]):  #T .eval() 
                res_test += 1.0
        res_test /= float(len(predictions_test))
        
        res_valid = 0.0
        for i in range(len(predictions_valid)):
            if(predictions_valid[i][n_bit-1]==valid_set_y[i][n_bit-1]):  #T .eval()
                res_valid += 1.0
        res_valid /= float(len(predictions_valid))  
        
        #############################################3        
        print("res_valid")
        print(res_valid) 
        
        print("res_test")
        print(res_test)
        
        '''
        print("predictions_test")
        print(predictions_test)
        
        print("test_set_y.eval()")
        print(test_set_y.eval())        
        '''
      
        
        
        
        if res_valid > best_f1:

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid
            best_tf1 = res_test
            best_epoch = e

            if verbose:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid,
                      'best test F1', res_test)
            
            if res_test==1.0:
                break
               
        if decay==False:
            decay_cnt+=1
            if(decay_cnt>8):
                decay=True
                
        # learning rate decay if no improvement in "decay_period" epochs
        if decay and abs(best_epoch-current_epoch) >= decay_period:
            learning_rate *= 0.5
            rnn = best_rnn
            decay = False
            decay_cnt=0

        if learning_rate < 1e-5:
            break
       

    print('BEST RESULT: epoch', best_epoch,
           'valid F1', best_f1,
           'best test F1', best_tf1)
    
    return rnn


#TODO: implement LSTM class to learn parity function
class LSTM(object):

    def __init__(self, nh, nc, ne, de, cs, normal=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        
        
        ##### LSTM parameter
        self.wi = theano.shared(name='wi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.ui = theano.shared(name='ui',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.uo = theano.shared(name='uo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wc = theano.shared(name='wc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.uc = theano.shared(name='uc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
       
        self.wf = theano.shared(name='wf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.uf = theano.shared(name='uf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))

        self.c0 = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        
 
        # bundle
        self.params = [self.emb, self.w, self.h0, self.wi, self.ui, self.wo, self.uo, self.wc, self.uc, self.wf, self.uf]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels
        
        def recurrence(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wi) + T.dot(h_tm1, self.ui))
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wf) + T.dot(h_tm1, self.uf))
            c_t = i_t*T.tanh(T.dot(x_t, self.wc) + T.dot(h_tm1, self.uc)) + f_t*c_tm1
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wo) + T.dot(h_tm1, self.uo))
            h_t = o_t*T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w))  
            
            return [h_t, c_t, s_t]

        [h, c, s], _ = theano.scan(fn=recurrence,
                                   sequences=x,
                                   outputs_info=[self.h0, self.c0, None],
                                   n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        error = self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()
        return error
            
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(n_bit=8, nh=200, de=50, n_epoch=1000, learning_rate=0.1, decay_period=10):

    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    train_set_x = train_set[0]
    train_set_y = train_set[1]
    valid_set_x = valid_set[0]
    valid_set_y = valid_set[1]
    test_set_x  = test_set[0]
    test_set_y  = test_set[1]

    
    
    ''' T
    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    '''
    
    nc=2
    ne=2
    cs=1

    # instanciate the model
    seed = random.randint(0, 4294967295)
    numpy.random.seed(seed)
    random.seed(seed)
    
    print('... building the model')
    rnn = LSTM(
        nh=nh,
        nc=nc,
        ne=ne,
        de=de,
        cs=cs,
        normal=False)

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    decay=True
    verbose=True
    decay_cnt=0
    
    for e in range(n_epoch):
        print("========== epoch %i ==========") % (e)
        print("===== learning_rate: %f =====") % (learning_rate)
        
        '''
        # shuffle
        seed = random.randint(0, 4294967295)
        shuffle_rnn(train_set_x, seed)
        '''

        current_epoch = e
        tic = timeit.default_timer()



        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            error = rnn.train(x, y, cs, learning_rate)
        print(error)           
             
        '''    
        ############################################################33
        A = rnn.get_param()
        print("A")
        print("self.emb")
        print(type(A[0].get_value()))
        print(A[0].get_value())
        
        print("self.wx")
        print(type(A[1].get_value()))
        print(A[1].get_value())  
        
        print("self.wh")
        print(type(A[2].get_value()))
        print(A[2].get_value())
        
        print("self.w")
        print(type(A[3].get_value()))
        print(A[3].get_value())
        
        print("self.h0")
        print(type(A[4].get_value()))
        print(A[4].get_value())
        raw_input()
        ''' 
            
        print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic))
        sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [rnn.classify(numpy.asarray(
                            contextwin(x, cs)).astype('int32'))
                            for x in test_set_x]       ################T .get_value()
        predictions_valid = [rnn.classify(numpy.asarray(
                             contextwin(x, cs)).astype('int32'))
                             for x in valid_set_x]      ###############T  .get_value()
        '''
        for x in test_set_x.get_value():
            print("x")
            print(type(x))
            print(x)
            
            print("numpy.asarray(contextwin(x, cs)).astype('int32')")
            print(type(numpy.asarray(contextwin(x, cs)).astype('int32')))
            print(numpy.asarray(contextwin(x, cs)).astype('int32'))
            
            print("rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32'))")
            print(type(rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32'))))
            print(rnn.classify(numpy.asarray(contextwin(x, cs)).astype('int32')))
            
            raw_input()
        
        print("predictions_test")
        print(type(predictions_test))
        print(predictions_test)
        
        print
        print("test_set_y.eval()")
        print(type(test_set_y.eval()))
        print(test_set_y.eval())
        '''
        
        
        # evaluation // compute the accuracy using conlleval.pl
        res_test = 0.0
        for i in range(len(predictions_test)):
            if(predictions_test[i][n_bit-1]==test_set_y[i][n_bit-1]):  #T .eval() 
                res_test += 1.0
        res_test /= float(len(predictions_test))
        
        res_valid = 0.0
        for i in range(len(predictions_valid)):
            if(predictions_valid[i][n_bit-1]==valid_set_y[i][n_bit-1]):  #T .eval()
                res_valid += 1.0
        res_valid /= float(len(predictions_valid))  
        
        #############################################3        
        print("res_valid")
        print(res_valid) 
        
        print("res_test")
        print(res_test)
        
        '''
        print("predictions_test")
        print(predictions_test)
        
        print("test_set_y.eval()")
        print(test_set_y.eval())        
        '''
      
        
        
        
        if res_valid > best_f1:

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid
            best_tf1 = res_test
            best_epoch = e

            if verbose:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid,
                      'best test F1', res_test)
            
            if res_test==1.0:
                break
               
        if decay==False:
            decay_cnt+=1
            if(decay_cnt>8):
                decay=True
                
        # learning rate decay if no improvement in "decay_period" epochs
        if decay and abs(best_epoch-current_epoch) >= decay_period:
            learning_rate *= 0.5
            rnn = best_rnn
            decay = False
            decay_cnt=0

        if learning_rate < 1e-5:
            break
       

    print('BEST RESULT: epoch', best_epoch,
           'valid F1', best_f1,
           'best test F1', best_tf1)
    
    return rnn

    
if __name__ == '__main__':
    test_mlp_parity()
