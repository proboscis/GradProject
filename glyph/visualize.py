import util,theano

def compressImages(sda,images,i):
    """
    :param sda:
    :param images:
    :param i :
    :return: images compressed by an ith sigmoid layer
    """
    compress = theano.function(
        inputs = [sda.x],
        outputs = sda.sigmoidLayers[i].output)
    return compress(images)

def loadRandomImages(folder,number,resolution):
    from os import listdir
    from scipy import misc
    from itertools import islice
    import numpy,itertools,random
    imgFiles = listdir(folder)
    size = min(len(imgFiles),number)
    random.shuffle(imgFiles)
    linear = resolution[0] * resolution[1]
    # random.shuffle(imgFiles)#destructive operation...
    print "imgShape",linear
    def loadImage(path):
        img = misc.imread(path)
        return img
    images = (misc.imread(folder+"/"+imgFile).reshape(linear) for imgFile in imgFiles)
    result = numpy.zeros((size,linear),dtype=theano.config.floatX)
    for i,img in enumerate(islice(images,size)):
        if i % 1000 == 0:
            print "loading %dth image of %d images" % (i,size)
        result[i] = img
    return result,size

def smallestSquare(area):
    import math
    sq = int(math.sqrt(area))
    numbers = (i for i in xrange(1,sq)
               if area % i == 0)
    l = max(numbers)
    return l, area/l

def genCompressors(sda):
    layers = sda.sigmoidLayers
    for layer in layers:
        f = theano.function(
            inputs = [sda.x],
            outputs = layer.output)
        yield f

def dataResolution(data):
    import numpy
    if isinstance(data,numpy.ndarray):
        size = data.shape[1]
        return smallestSquare(size)
    else :
        size = data.get_value(borrow=True).shape[1]
        return smallestSquare(size)

def makeImageOfData(data):
    import numpy
    shape = data.shape
    res = dataResolution(data)
    print shape,res
    return util.makeImage(data,dataResolution(data),(10,10))

def findModels():
    import os , os.path
    from os import path
    root = "../models/ebooks"
    for fn in os.listdir(root):
        print fn
        abspath = path.join(root,fn)
        if not path.isdir(abspath):
            print abspath
            yield abspath





def reconstructor(sda):
    top = sda.dALayers[0]
    reconstructed = top.reconstructedInput(top.hiddenValues(sda.x))
    return theano.function(
        inputs=[sda.x],
        outputs=reconstructed
    )

def printDict(d):
    if type(d) == dict:
        for k,v in d.iteritems():
            print "key:%s value:%s" % (k,v)
            printDict(v)

def convertModelDataToImages(path):
    print "loading a model at " + path
    info,sda = util.load(path)
    print "params info:"
    printDict(info)
    print "loading random images..."
    if (info["dataSet"]["kind"] == "mnist"):
        data = util.loadMnistData()[0][0].get_value(borrow=False)[0:100]
        print "shape", data.shape
        data = data.reshape(100,784)
    elif(info["dataSet"]["kind"] == "ebook"):
        data,size = loadRandomImages("../gray/",100,info["dataSet"]["shape"])

    # create compressors
    print "generating compressors..."
    compressors = genCompressors(sda)
    # compress all images
    print "compressing images..."
    compressedData = map(lambda f:f(data),compressors)
    # convert to images
    print "making images..."
    compressedImages = map(makeImageOfData,compressedData)
    # save to folders
    folder = "../images/"+info["dataSet"]["kind"]+"/"+path.replace(".pkl","")

    print "saving images to: "+ folder
    for i,img in enumerate(compressedImages):
        dst = folder + ("/layer%d.png" % i)
        util.ensurePathExists(dst)
        img.save(dst)

    #reconstruct images
    reconstructed = reconstructor(sda)(data)
    reconImage = makeImageOfData(reconstructed)
    reconImage.save(folder + "/reconstructed.png")
    #save learned features as an image
    sda.dALayers[0].saveLayerImage(folder + "/layer_weight.png",info["dataSet"]["shape"],(10,10))

if __name__ == '__main__':
    import os
    models = list(findModels())
    l = len(models)
    print "start converting models:",l
    for i,modelPath in enumerate(models):
        if not os.path.isdir(modelPath):
            print modelPath
            print "converting %dth model of %d models" % (i,l)
            convertModelDataToImages(modelPath)
    print "done! yay!"
