__author__ = 'kentomasui'
import theano
import numpy
from theano import tensor as T


class LogisticRegression:
    def __init__(self, input, nIn, nOut):
        self.x = T.dmatrix('x')
        self.W = theano.shared(
            value=numpy.zeros(
                (nIn, nOut),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (nOut,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.PYGivenX = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.YPrediction = T.argmax(self.PYGivenX, axis=1)
        self.params = [self.W, self.b]


    def negativeLogLikelihood(self, y):
        return -T.mean(T.log(self.PYGivenX)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.ndim != self.YPrediction.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.YPrediction.type)
            )
        if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
            return T.mean(T.neq(self.YPrediction, y))
        else:
            raise NotImplementedError()
