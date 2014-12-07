__author__ = 'kentomasui'

class LRTest:
    def __init__(self):
        import theano
        import util
        from theano import tensor as T
        from logistic_regression import LogisticRegression
        self.index = T.iscalar('index')
        self.BATCH_SIZE = 100
        self.LEARNING_RATE = 0.12
        self.dataSets = util.loadMnistData("mnist.pkl.gz")
        self.x = T.dmatrix('x')
        self.y = T.ivector('y')
        self.index = T.iscalar('index')
        self.classifier = LogisticRegression(input=self.x, nIn=28 * 28, nOut=10)
        self.cost = self.classifier.negativeLogLikelihood(self.y)
        self.gW = T.grad(cost=self.cost, wrt=self.classifier.W)
        self.gB = T.grad(cost=self.cost, wrt=self.classifier.b)
        self.trainSet, self.validSet, self.testSet = self.dataSets
        self.nTrainSet, self.nValidSet, self.nTestSet = map(self.numBatches, self.dataSets)
        updates = [
            (self.classifier.W, self.classifier.W - self.LEARNING_RATE * self.gW),
            (self.classifier.b, self.classifier.b - self.LEARNING_RATE * self.gB)
        ]

        def makeGivens(data):
            return {
                self.x: data[0][self.index * self.BATCH_SIZE:(self.index + 1) * self.BATCH_SIZE],
                self.y: data[1][self.index * self.BATCH_SIZE:(self.index + 1) * self.BATCH_SIZE]
            }

        self.testModel = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens=makeGivens(self.dataSets[2])
        )
        self.validationModel = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens=makeGivens(self.dataSets[1])
        )
        self.trainModel = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=updates,
            givens=makeGivens(self.dataSets[0])
        )

    def numBatches(self, dataSet):
        return dataSet[0].get_value(borrow=True).shape[0] / self.BATCH_SIZE

    def printValid(self, epoch, batchIndex, loss):
        return 'epoch %i, minibatch %i/%i, validation error %f %%' % (
            epoch,
            batchIndex + 1,
            self.nTrainSet,
            loss * 100.
        )

    def printTestScore(self, epoch, batchIndex, score):
        return (
                   '     epoch %i, minibatch %i/%i, test error of'
                   ' best model %f %%'
               ) % (
                   epoch,
                   batchIndex + 1,
                   self.nTrainSet,
                   score * 100.
               )

    def resultString(self, best, test):
        return ('Optimization complete with best validation score of %f %%,'
                'with test performance %f %%') % (best * 100., test * 100.)

    def train(self):
        import numpy

        patience = 5000
        patienceIncrease = 2
        MAX_EPOCH = 1000
        improveThresh = 0.995
        validationFreq = min(self.nTrainSet, patience / 2)
        bestValidationLoss = numpy.inf
        epoch = 0
        done = False
        testLoss = 0

        while (epoch < MAX_EPOCH) and not done:
            epoch += 1
            for batchIndex in xrange(self.nTrainSet):
                avgCost = self.trainModel(batchIndex)
                iter = (epoch - 1) * self.nTrainSet + batchIndex
                if (iter + 1) % validationFreq == 0:
                    loss = numpy.mean(map(self.validationModel, xrange(self.nValidSet)))
                    if loss < bestValidationLoss:
                        if loss < bestValidationLoss * improveThresh:
                            patience = max(patience, iter * patienceIncrease)
                        bestValidationLoss = loss
                        testLoss = numpy.mean(map(self.testModel, xrange(self.nTestSet)))
                    yield epoch,batchIndex,loss, testLoss, bestValidationLoss
                if patience <= iter:
                    done = True
                    break

    def doTrain(self):
        for epoch,batchIndex, loss,testScore,bestScore in self.train():
            str = self.printValid(epoch,batchIndex,loss)
            str += self.printTestScore(epoch,batchIndex,testScore)
            print(str)

        print(self.resultString(bestScore,testScore))

