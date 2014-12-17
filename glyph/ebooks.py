__author__ = 'kentomasui'

from util import fileLines

import numpy,util,theano
from util import load,save ,checkTime
from sda import StackedDenoisingAutoencoder
import math

def purchaseData():
    return (line.split(",") for line in fileLines("purchase.csv"))

def bookInfo():
    return (line.split(",") for line in fileLines("boookinfo.csv"))

def rgb2Gray(rgb):
    import numpy
    return numpy.dot(rgb[...,:3],[0.299, 0.587, 0.144])

def convert2GrayAll(srcFolder,dstFolder):
    #this took an hour to write... orz
    from os import listdir,path,makedirs
    from scipy import misc
    if not path.exists(dstFolder):
        makedirs(dstFolder)
    imgFiles = listdir(srcFolder)
    for i, imgFile in enumerate(listdir(srcFolder),start=1):
        print i , " of " , len(imgFiles)
        dstFile = dstFolder+'/'+imgFile
        if not path.exists(dstFile):
            src = misc.imread(srcFolder+'/'+imgFile)
            gray = rgb2Gray(src)
            misc.imsave(dstFolder+'/'+imgFile,gray)
        else:
            print "skipped"

def loadImages(folder):
    from os import listdir
    from scipy import misc
    from itertools import islice
    import numpy,itertools,random
    imgFiles = listdir(folder)
    # random.shuffle(imgFiles)#destructive operation...
    images = (misc.imread(folder+"/"+imgFile).reshape(10000) for imgFile in imgFiles)
    size = len(imgFiles)/3
    print "required ram:%dMb" % (size * 10000 * 64 / 8 / 1024 / 1024)
    result = numpy.zeros((size,10000),dtype=theano.config.floatX)
    for i,img in enumerate(islice(images,size)):
        if i % 1000 == 0:
            print "loading %dth image" % i
        result[i] = img
    return result,size

def createSDA2(data,nIn,hiddenLayerSizes,corruptionLevels):
    print "start data transportation to gpu..."
    size = len(data)/nIn
    def makeX():
        return theano.shared(data.reshape(size,nIn),borrow=True)
    gpuSendTime,x = checkTime(makeX)
    data = None # free memory
    print "data transportation to gpu finished in ", gpuSendTime , " seconds"

    def train():
        numpyRng = numpy.random.RandomState(89677)
        model = StackedDenoisingAutoencoder(
            numpyRng,
            nIn=nIn,
            hiddenLayerSizes=hiddenLayerSizes)
        model.preTrain(data=x,corruptionLevels=corruptionLevels)
        return model
    print "start compilation and training..."
    trainingTime,sda = checkTime(train)
    print "training finished in ", trainingTime , " seconds"
    return sda

def createSDA(params):
    def l1():
        return loadImages("../gray")
    print "start converting images to a single list..."
    loadingTime,(data,size) = checkTime(l1)
    print "data conversion took ",loadingTime ," seconds"
    print "start data transportation to gpu..."

    def makeX():
        return theano.shared(data.reshape(size,10000),borrow=True)
    gpuSendTime,x = checkTime(makeX)
    data = None # free memory
    print "data transportation to gpu finished in ", gpuSendTime , " seconds"
    hiddenLayerSizes = params["hiddenLayerSizes"]
    corruptionLevels = params["corruptionLevels"]
    def train():
        numpyRng = numpy.random.RandomState(89677)
        model = StackedDenoisingAutoencoder(
            numpyRng,
            nIn=100*100,
        hiddenLayerSizes=hiddenLayerSizes)
        model.preTrain(data=x,corruptionLevels=corruptionLevels)
        return model
    print "start compilation and training..."
    trainingTime,sda = checkTime(train)
    print "training finished in ", trainingTime , " seconds"
    return sda

def saveSdaImages(sda,path):
    print("saving layer images...")
    for i,da in enumerate(sda.dALayers):
        print da.W.get_value(borrow=True).shape
        x = int(math.sqrt(da.nVisible))
        y = int(math.sqrt(da.nHidden))
        da.saveLayerImage(
            (path +"/layer_%d" % i) + ".png",
            resolution=(x,x),
            tileShape=(y,y))
    print "saving layers done"

def compressImages(sda,images):
    from theano import tensor as T
    compress = theano.function(
        inputs = [sda.x],
        outputs = sda.sigmoidLayers[-1].output)
    return compress(images)

def clustering(data):
    from sklearn.cluster import KMeans
    kmeans = KMeans()
    kmeans.fit(data)
    labels = kmeans.labels_
    uniqueLabels = numpy.unique(labels)
    nCluster = len(uniqueLabels)
    centers = kmeans.cluster_centers_
    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    plt.figure(1)
    plt.clf()
    for center in centers:
        print center
    for k,col in zip(range(nCluster),colors):
        members = labels == k
        center = centers[k]
        plt.plot(data[members,0],data[members,1],col +'.')
        plt.plot(center[0],center[1],'o',markerfacecolor=col,
                 markeredgecolor = 'k',markersize = 14)
    plt.title("clusters")
    plt.show()


if __name__ == '__main__':
    #loadImage("../resized/60000000.jpg")
    #convert2GrayAll("../resized","../gray")
    genSDA = util.fileMemo(createSDA,"../data/pre_trained_sda_ebooks.pkl")
    sda = genSDA({
        "hiddenLayerSizes":(2500, 400, 100,10),
        "corruptionLevels":(0.3,0.2,0.1,0.1)
    })
    def l1():
        result,size = loadImages("../gray")
        return compressImages(sda,result)
    compressed = util.loadOrCall("../data/compressed.pkl",l1)
    # numpy.random.shuffle(compressed)#destructive
    # image = util.makeImage(compressed,(50,50),(10,10))
    # import matplotlib.pylab as plb
    # plb.imshow(image)
    # plb.show()

    import itertools

    # compressedTuples = map(tuple,compressed)
    # sets = set(compressedTuples)
    # for g in sets:
    #     print g
    #
    # for vec in compressed:
    #     print vec
    # groups = itertools.groupby(compressed,lambda a:a)
    # for k,v in groups:
    #      print "key:",k
    #saveSdaImages(sda,"../data/ebooks")
    #clustering(compressed)


