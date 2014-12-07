__author__ = 'kentomasui'
import os
import gzip
import cPickle as pickle
import theano
import numpy
from theano import tensor as T
from itertools import islice

def loadMnistData(datasetPath):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(datasetPath)
    if data_dir == "" and not os.path.isfile(datasetPath):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            datasetPath
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            datasetPath = new_path

    if (not os.path.isfile(datasetPath)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, datasetPath)

    print '... loading data'

    # Load the dataset
    f = gzip.open(datasetPath, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

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
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def validationResultString(model,epoch, batchIndex, loss):
    return 'epoch %i, minibatch %i/%i, validation error %f %%' % (
        epoch,
        batchIndex + 1,
        model.nTrainSet,
        loss * 100.
    )

def testScoreString(model, epoch, batchIndex, score):
    return (
               '     epoch %i, minibatch %i/%i, test error of'
               ' best model %f %%'
           ) % (
               epoch,
               batchIndex + 1,
               model.nTrainSet,
               score * 100.
           )

def resultString(best, test):
    return ('Optimization complete with best validation score of %f %%,'
            'with test performance %f %%') % (best * 100., test * 100.)

def train(trainingItr,model):
    for epoch,batchIndex, loss,testScore,bestScore in trainingItr:
        str = validationResultString(model,epoch,batchIndex,loss)
        str += testScoreString(model,epoch,batchIndex,testScore)
        print(str)


def trainingIterator(model,
          patience = 5000,
          patienceIncrease = 2,
          MAX_EPOCH = 1000,
          improveThresh=0.995):
    validationFreq = min(model.nTrainSet, patience / 2)
    bestValidationLoss = numpy.inf
    epoch = 0
    done = False
    testLoss = 0

    while (epoch < MAX_EPOCH) and not done:
        epoch += 1
        for batchIndex in xrange(model.nTrainSet):
            avgCost = model.trainModel(batchIndex)
            iter = (epoch - 1) * model.nTrainSet + batchIndex
            if (iter + 1) % validationFreq == 0:
                loss = numpy.mean(map(model.validationModel, xrange(model.nValidSet)))
                if loss < bestValidationLoss:
                    if loss < bestValidationLoss * improveThresh:
                        patience = max(patience, iter * patienceIncrease)
                    bestValidationLoss = loss
                    testLoss = numpy.mean(map(model.testModel, xrange(model.nTestSet)))
                yield epoch,batchIndex,loss, testLoss, bestValidationLoss
            if patience <= iter:
                done = True
                break


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def autoClose(filename, param, f):
    print('autoclose')
    file = open(os.path.expanduser(filename), param)
    print('file')
    result = f(file)
    file.close()
    return result


def readAppend(filename, f):
    path = os.path.expanduser(filename)
    files = [open(path, p) for p in ['r', 'a']]
    f(files[0], files[1])
    for f in files:
        f.close()

def save(obj,fileName):
    autoClose(fileName,'wb',lambda f:pickle.dump(obj,f))

def load(fileName):
    autoClose(fileName,'rb',lambda f:pickle.load(f))