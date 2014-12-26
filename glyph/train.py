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

def createModel(info):
    return modelTable()[info["model"]["kind"]](info)

def createSDA(info):
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
    if visualize:
        import visualize
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        compressors = list(visualize.genCompressors(model))
        inputs,size = visualize.loadRandomImages("../gray",100,None)
        plots = {}
        plt.ion()

    def trainer(data):
        print "dataShape", data.get_value(borrow=True).shape
        preTrainer = list(model.pretrainingFunctions(data,batchSize=batchSize))
        assert len(corruptionLevels) == len(preTrainer) , "given corruption levels do not correspond to the layers!!!"
        for i,(trainer,corruptionLevel,preLearningRate) in enumerate(zip(preTrainer,corruptionLevels,preLearningRates)):
            for epoch in xrange(modelInfo["pretrainingEpochs"][i]):
                print 'Pre-training layer %i, epoch %d start with learning rate of %f' % (i,epoch,preLearningRate)
                trainScores = [trainer(batchIndex,corruptionLevel,preLearningRate) for batchIndex in xrange(data.get_value(borrow=True).shape[0]/batchSize)]
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),numpy.mean(trainScores)
                if visualize:
                    compressed = map(lambda f:f(inputs),compressors)
                    images = map(visualize.makeImageOfData,compressed)
                    for j,img in enumerate(images):
                        dst = info["dst"]+"/epoch%dlayer%d.png" % (epoch,j)
                        util.ensurePathExists(dst)
                        img.save(dst)
                    #
                    # plt.figure(0)
                    # if 0 not in plots.keys():
                    #     print "show image!"
                    #     plots[0] = plt.imshow(images[0], cmap = cm.Greys_r,interpolation='nearest')
                    # print "set data for figure %d" % 0
                    # print "number of images %d" % len(images)
                    # plots[0].set_data(images[0])
                    #
                    # print "update figure %d" % 0
                    # plt.draw()

                    # for j,img in enumerate(images):
                    #     plt.figure(j)
                    #     plt.ion()
                    #     if j not in plots.keys():
                    #         plots[j] = plt.imshow(img, cmap = cm.Greys_r,interpolation='nearest')
                    #         #plt.show()
                    #     print "set data for figure %d" % j
                    #     plots[j].set_data(img)
                    #
                    #     print "update figure %d" %j
                    #     plt.draw()
                        #so, how dows this end up?
                    #show the activation of each layers.

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
    model,train = createModel(info)
    train(dataSet)
    return model

if __name__ == '__main__':
    import sys,json
    info = json.loads(sys.argv[1])
    dst = info["dst"]
    def l():
        return info,train(info)
    model = util.saveIfNotExist(dst,l)