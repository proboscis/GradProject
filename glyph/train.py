__author__ = 'kento'
import util,numpy
def modelTable():
    return {
        "sda":createSDA
    }

def dataSetTable():
    return {
        "ebook":createEbookDataSet
    }

def createModel(modelInfo):
    return modelTable()[modelInfo["kind"]](modelInfo)

def createSDA(modelInfo):
    from sda import StackedDenoisingAutoencoder
    model = StackedDenoisingAutoencoder(
        numpyRng=numpy.random.RandomState(modelInfo["randomSeed"]),
        nIn = modelInfo["nIn"],
        hiddenLayerSizes=modelInfo["hiddenLayerSizes"],
        nOut=modelInfo["nOut"]
    )
    corruptionLevels = modelInfo["corruptionLevels"]
    #preLearningRates = modelInfo["preLearningRates"]
    preLearningRate = 0.2
    batchSize = modelInfo["batchSize"]
    print "batchSize",batchSize
    def trainer(data):
        print "dataShape", data.get_value(borrow=True).shape
        preTrainer = list(model.pretrainingFunctions(data,batchSize=batchSize))
        assert len(corruptionLevels) == len(preTrainer) , "given corruption levels do not correspond to the layers!!!"
        for i,(trainer,corruptionLevel) in enumerate(zip(preTrainer,corruptionLevels)):
            print corruptionLevel,preLearningRate
            for epoch in xrange(modelInfo["pretrainingEpochs"][i]):
                print 'Pre-training layer %i, epoch %d start with learning rate of %f' % (i,epoch,preLearningRate)
                trainScores = [trainer(batchIndex,corruptionLevel,preLearningRate) for batchIndex in xrange(data.get_value(borrow=True).shape[0]/batchSize)]
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),numpy.mean(trainScores)
    return model,trainer
    #return something

def createDataSet(dataSetInfo):
    return dataSetTable()[dataSetInfo["kind"]](dataSetInfo)


def createEbookDataSet(info):
    folder = info["path"]
    from os import listdir
    from scipy import misc
    from itertools import islice
    import random, theano
    imgFiles = listdir(folder)
    random.shuffle(imgFiles)#destructive operation... and makes operation so slow
    shape = info["shape"]
    print "shape",shape
    size = info["shape"][0]
    nInput = min(info["shape"][1],len(imgFiles))
    images = (misc.imread(folder+"/"+imgFile).reshape(size) for imgFile in imgFiles)
    print "loading %d out of %d images" % (nInput , len(imgFiles))
    result = numpy.zeros((nInput,size),dtype=theano.config.floatX)
    for i,img in enumerate(islice(images,nInput)):
        if i % 1000 == 0:
            print "loading %dth image" % i
        result[i] = img
    return theano.shared(result.reshape(nInput,size),borrow=True)

def train(info):
    dataSet = createDataSet(info["dataSet"])
    model,train = createModel(info["model"])
    train(dataSet)
    return model

#TODO
if __name__ == '__main__':
    import sys,json
    info = json.loads(sys.argv[1])
    dst = info["dst"]
    def l():
        return train(info)
    model = util.loadOrCall(dst,l)