__author__ = 'kentomasui'
from layer import Layer
from theano import tensor as T
from logistic_regression import LogisticRegression
class MultiLayerPerceptron:
    def __init__(self,rng,input,nIn,nHidden,nOut):
        self.hidden = Layer(rng=rng,input=input,nIn=nIn,nOut=nHidden,activation=T.tanh)
        self.regression = LogisticRegression(self.hidden.output,nIn=nHidden,nOut=nOut)
        self.L1 = abs(self.hidden.W).sum() + abs(self.regression.W).sum()
        self.L2 = abs(self.hidden.W ** 2).sum() + abs(self.regression.W ** 2).sum()
        self.negativeLogLikelihood = self.regression.negativeLogLikelihood
        self.errors = self.regression.errors
        self.params = self.hidden.params + self.regression.params