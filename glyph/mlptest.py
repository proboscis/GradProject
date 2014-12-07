__author__ = 'kentomasui'

def createTestMlp():
    import mlp,util,numpy,theano
    from theano import tensor as T
    data = util.loadMnistData("mnist.pkl.gz")

    random = numpy.random.RandomState(1234)
    x = T.dmatrix('x')
    y = T.ivector('y')
    index = T.iscalar("index")
    L1 = 0.00
    L2 = 0.0001
    batchSize = 20
    learningRate = 0.01
    classifier = mlp.MultiLayerPerceptron(
        rng=random,
        input=x,
        nIn=28*28,
        nHidden=500,
        nOut=10
    )
    cost = classifier.negativeLogLikelihood(y) + L1*classifier.L1 + L2*classifier.L2
    def batchNum(set):
        return set[0].get_value(borrow=True).shape[0] / batchSize

    def givens(set):
        return {
            x: set[0][index * batchSize:(index + 1) * batchSize],
            y: set[1][index * batchSize:(index + 1) * batchSize]
        }
    def costFunction(set):
        return theano.function(inputs=[index],outputs=classifier.errors(y),givens=givens(set))
    testModel,validationModel = map(costFunction,list(reversed(data))[:2])
    gparams = [T.grad(cost,param) for param in classifier.params]
    updates = [(param,param-learningRate*gparam) for param,gparam in zip(classifier.params,gparams)]
    trainModel = theano.function(inputs=[index],outputs=classifier.errors(y),givens=givens(data[2]),updates=updates)
    class Object():
        pass
    result = Object()
    result.testModel = testModel
    result.validationModel = validationModel
    result.trainModel = trainModel
    result.nTrainSet,result.nValidSet,result.nTestSet = map(batchNum,data)
    return result

def mlpTest():
    import util
    model = createTestMlp()
    itr = util.trainingIterator(model=model,patience=10000)
    util.train(itr,model)

if __name__ == '__main__':
    mlpTest()