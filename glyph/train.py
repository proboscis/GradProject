__author__ = 'kento'
import util,numpy
def modelTable():
    return {
        "sda":createSDA
    }

def dataSetTable():
    return {
        "ebook":createEbookDataSet,
        "mnist":createMnistDataSet
    }

def createModel(info,trainset):
    return modelTable()[info["model"]["kind"]](info,trainset)

def createSDA(info,trainset):
    modelInfo = info["model"]
    from sda import StackedDenoisingAutoencoder
    visualize = modelInfo.get("visualize",True)
    model = StackedDenoisingAutoencoder(
        numpyRng=numpy.random.RandomState(modelInfo["randomSeed"]),
        nIn = modelInfo["nIn"],
        hiddenLayerSizes=modelInfo["hiddenLayerSizes"],
        nOut=modelInfo["nOut"]
    )
    corruptionLevels = modelInfo["corruptionLevels"]
    preLearningRates = modelInfo["preLearningRates"]
    batchSize = modelInfo["batchSize"]
    print "batchSize",batchSize
    def ensureSaveImage(img,dst):
        util.ensurePathExists(dst)
        img.save(dst)
    if visualize:
        #save input image
        import visualize
        print type(trainset)
        print trainset.shape
        img = visualize.makeImageOfData(trainset.get_value(borrow=True))
        dst = info["dst"].replace("pkl","")+"/inputs.png"
        ensureSaveImage(img,dst)

    def trainer(data):
        print "training dataShape", data.get_value(borrow=True).shape
        if visualize:
            trainSet = data.get_value(borrow=True)
            trainSetImg = visualize.makeImageOfData(trainSet)
            dst = info["dst"].replace("pkl","")+"/train_set.png"
            ensureSaveImage(trainSetImg,dst)
        preTrainer = list(model.pretrainingFunctions(data,batchSize=batchSize))
        assert len(corruptionLevels) == len(preTrainer) , "given corruption levels do not correspond to the layers!!!"
        for i,(trainer,corruptionLevel,preLearningRate) in enumerate(zip(preTrainer,corruptionLevels,preLearningRates)):
            for epoch in xrange(modelInfo["pretrainingEpochs"][i]):
                print 'Pre-training layer %i, epoch %d start with learning rate of %f' % (i,epoch,preLearningRate)
                trainScores = [trainer(batchIndex,corruptionLevel,preLearningRate) for batchIndex in xrange(data.get_value(borrow=True).shape[0]/batchSize)]
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),numpy.mean(trainScores)
                if visualize:
                    for name,img in visualize.createSdaImages(model,trainset.get_value(borrow=True)):
                        dst = info["dst"].replace("pkl","")+"/layer%depoch%d/" % (i,epoch) + name
                        ensureSaveImage(img,dst)


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
    size = shape[0] * shape[1]
    nInput = min(info["numData"],len(imgFiles))
    images = (misc.imread(folder+"/"+imgFile).reshape(size) for imgFile in imgFiles)
    print "loading %d out of %d images" % (nInput , len(imgFiles))
    result = numpy.zeros((nInput,size),dtype=theano.config.floatX)
    for i,img in enumerate(islice(images,nInput)):
        if i % 1000 == 0:
            print "loading %dth image" % i
        result[i] = img
    return theano.shared(result.reshape(nInput,size),borrow=True)

def createMnistDataSet(info):
    data = util.loadMnistData()
    return data[0][0] #train_set_x

def train(info):
    dataSet = createDataSet(info["dataSet"])
    model,train = createModel(info,dataSet)
    train(dataSet)
    return model

if __name__ == '__main__':
    import sys,json
    info = json.loads(sys.argv[1])
    dst = info["dst"]
    def l():
        return info,train(info)
    model = util.saveIfNotExist(dst,l)