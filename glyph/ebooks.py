__author__ = 'kentomasui'

from util import fileLines
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
    import numpy,itertools
    images = (misc.imread(folder+"/"+imgFile) for imgFile in listdir(folder))
    arrays = (ary.reshape(10000).tolist() for ary in images)
    result = []
    for ary in arrays:
        result.extend(ary)
    return result

if __name__ == '__main__':
    #loadImage("../resized/60000000.jpg")
    #convert2GrayAll("../resized","../gray")


    import numpy,util,theano
    from glyph.util import load,save ,checkTime
    from glyph.sda import StackedDenoisingAutoencoder
    print "start converting images to a single list..."
    def loadData():
        return loadImages("../gray")
    loadingTime,data = checkTime(loadData)
    print "data conversion took ",loadingTime ," seconds"
    print "start data transportation to gpu..."
    def makeX():
        size = len(data)/10000
        return theano.shared(numpy.asarray(data,dtype=theano.config.floatX).reshape(size,10000),borrow=True)
    gpuSendTime,x = checkTime(makeX)
    print "data transportation to gpu finished in ", gpuSendTime , " seconds"
    def train():
        numpyRng = numpy.random.RandomState(89677)
        model = StackedDenoisingAutoencoder(numpyRng,nIn=100*100,hiddenLayerSizes=[3000,500,100])
        model.preTrain(data=x)
        return model
    print "start compilation and training..."
    trainingTime,sda = checkTime(train)
    print "training finished in ", trainingTime , " seconds"
    save(sda,'../data/pre_trained_sda_ebooks.pkl')
    # sda = load('../data/pre_trained_sda_ebooks.pkl')
    print("saving layer images...")
    #TODO reimplement
    for i,da in enumerate(sda.dALayers):
        da.saveLayerImage(
            "../data/ebooks/layer_%d.png" % i,
            resolution=(100,100))
    print "program finish"
