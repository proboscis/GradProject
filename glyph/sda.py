__author__ = 'kentomasui'

import theano
from theano import tensor as T
from glyph.layer import Layer
from glyph.dA import DenoisingAutoEncoder
from glyph.logistic_regression import LogisticRegression
from glyph.util import load,save
import glyph.util
class StackedDenoisingAutoencoder:
    def __init__(self,
                 numpyRng,
                 theanoRng=None,
                 nIn=28*28,
                 hiddenLayerSizes=[500,500],
                 nOut=10):
        self.nLayers = len(hiddenLayerSizes)
        if not theanoRng:
            theanoRng = theano.tensor.shared_randomstreams.RandomStreams(numpyRng.randint(2 ** 30))
        self.x = T.dmatrix('x')
        self.y = T.ivector('y')
        def makeSigmoidLayer(lastLayer,lastLayerSize,size):
            return Layer(rng=numpyRng,input=lastLayer,nIn=lastLayerSize,nOut=size,activation=T.nnet.sigmoid)
        def makeDALayer(lastLayer,lastLayerSize,size,sigmoidLayer):
            return DenoisingAutoEncoder(
                numpyRng=numpyRng,theanoRng=theanoRng,input=lastLayer,
                nVisible=lastLayerSize,
                nHidden=size,
                W=sigmoidLayer.W,
                bHidden=sigmoidLayer.b)
        def makeLayers(lastLayer,lastInputSize,nextLayerSizes):
            if nextLayerSizes:
                newList = list(nextLayerSizes)
                size = newList.pop()
                sigmoidLayer = makeSigmoidLayer(lastLayer,lastInputSize,size)
                daLayer = makeDALayer(lastLayer,lastInputSize,size,sigmoidLayer)
                yield (sigmoidLayer,daLayer)
                for layer in makeLayers(sigmoidLayer.output,size,newList):
                    yield layer
        self.sigmoidLayers,self.dALayers = zip(*makeLayers(self.x,nIn,hiddenLayerSizes))
        self.logLayer = LogisticRegression(self.sigmoidLayers[-1].output,hiddenLayerSizes[-1],nOut)
        self.params = [l.params for l in self.sigmoidLayers] + [self.logLayer.negativeLogLikelihood(self.y)]
        self.fineTuneCost = self.logLayer.negativeLogLikelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretrainingFunctions(self,trainSetX,batchSize):
        index = T.lscalar("index")
        corruptionLevel = T.scalar('corruption')
        learningRate = T.scalar("learning")
        batchBegin = batchSize * index
        batchEnd = batchBegin + batchSize
        for dA in self.dALayers:
            cost,updates = dA.costFunctionAndUpdates(corruptionLevel,learningRate)
            f = theano.function(
                inputs=[
                    index,
                    theano.Param(corruptionLevel,default=0.2),
                    theano.Param(learningRate,default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={self.x:trainSetX[batchBegin:batchEnd]}
            )
            yield f

    def fineTuneFunctions(self,datasets,batchSize,learningRate):
        index = T.lscalar('i')
        trainSetX,trainSetY = datasets[0]
        validSetX,validSetY = datasets[1]
        testSetX,testSetY = datasets[2]
        gparams = T.grad(self.fineTuneCost,self.params)
        updates = [
            (param,param-gparam*learningRate)
            for param,gparam in zip(self.params,gparams)
        ]
        def makeGivens(x,y):
            return {self.x:x[index*batchSize:(index+1)*batchSize],
                    self.y:y[index*batchSize:(index+1)*batchSize]}
        trainer = theano.function(
            inputs=[index],
            outputs=self.fineTuneCost,
            updates=updates,
            givens=makeGivens(trainSetX,trainSetY),
            name='train'
        )
        testScoreI=theano.function(
            inputs=[index],
            outputs=self.errors,
            givens=makeGivens(testSetX,testSetY),
            name='test'
        )
        validScoreI=theano.function(
            inputs=[index],
            outputs=self.errors,
            givens=makeGivens(validSetX,validSetY),
            name='valid'
        )

        def validationScore():
            return [validScoreI(i) for i in xrange(validSetX.get_value(borrow=True).shape[0]/batchSize)]

        def testScore():
            return [testScoreI(i) for i in xrange(validSetX.get_value(borrow=True).shape[0]/batchSize)]

        return trainer,validationScore,testScore

    def preTrain(self,
                 data=glyph.util.loadMnistData("mnist.pkl.gz"),
                 batchSize=20,
                 preLearningRate=0.1,
                 corruptionLevels=(.1,.2,.3)):
        import numpy,glyph.util

        preTrainer = self.pretrainingFunctions(data[0][0],batchSize=batchSize)
        for i,(trainer,corruptionLevel) in enumerate(zip(preTrainer,corruptionLevels)):
            for epoch in xrange(15):
                trainScores = [trainer(batchIndex,corruptionLevel,preLearningRate) for batchIndex in xrange(data[0][0].get_value(borrow=True).shape[0]/batchSize)]
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),numpy.mean(trainScores)




if __name__ == '__main__':
    import numpy,util


    numpyRng = numpy.random.RandomState(89677)
    sda = StackedDenoisingAutoencoder(numpyRng,hiddenLayerSizes=[1000,1000,1000])
    sda.preTrain()
    save(sda,'data/pre_trained_sda.pkl')
    # data = util.loadMnistData("mnist.pkl.gz")
    # batchSize = 20
    # preLearningRate = 0.1
    # pretrainer = sda.pretrainingFunctions(data[0][0],batchSize=batchSize)
    # corruptionLevels = [0.1,0.2,0.3]
    # print list(sda.dALayers)
    # for i,(trainer,corruptionLevel) in enumerate(zip(pretrainer,corruptionLevels)):
    #     for epoch in xrange(15):
    #         trainScores = [trainer(batchIndex,corruptionLevel,preLearningRate) for batchIndex in xrange(data[0][0].get_value(borrow=True).shape[0]/batchSize)]
    #         numpy.mean(trainScores)
    #         print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch)

