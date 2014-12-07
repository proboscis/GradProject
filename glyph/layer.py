__author__ = 'kentomasui'
from theano import tensor as T
import numpy,theano


class Layer():
    def __init__(self,rng,input,nIn,nOut,W=None,b=None,activation=T.tan):
        self.input = input
        if W is None:
            r = numpy.sqrt(6. / (nIn + nOut))
            values = numpy.asarray(
                rng.uniform(low= -r,high=r,size=(nIn,nOut)),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                values *= 4
            W = theano.shared(value=values,name='W',borrow=True)
        self.W = W
        if b is None:
            values = numpy.zeros((nOut,),dtype=theano.config.floatX)
            b = theano.shared(value=values,name='b',borrow=True)
        self.b = b
        outputs = T.dot(input,self.W) + self.b
        self.output = outputs if activation is None else activation(outputs)
        """output matrices"""
        self.params = [self.W,self.b]