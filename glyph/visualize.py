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
    print "loadRandomImages at:" + folder
    from os import listdir
    from scipy import misc
    from itertools import islice
    import numpy,itertools,random
    imgFiles = listdir(folder)
    size = min(len(imgFiles),number)
    random.shuffle(imgFiles)
    shape = misc.imread(folder+"/"+imgFiles[0]).shape
    linear = shape[0]*shape[1]
    #linear = resolution[0] * resolution[1]
    # random.shuffle(imgFiles)#destructive operation...
    print "imgShape",linear
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
    numbers = (i for i in xrange(1,sq+1)
               if area % i == 0)
    l = max(numbers)
    return l, area/l

def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
def genCompressors(sda):
    layers = sda.sigmoidLayers
    print "genCompressor(sda),sda.x.shape = ", sda.x.shape
    #this does not determine the result shape, right?
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

def makeImageOfData(data,resolution=None):
    import numpy
    shape = data.shape
    res = dataResolution(data)
    print "makeImageOfData:inputShape",shape,"output shape:",res
    if resolution is None:
        resolution = dataResolution(data)
    return util.makeImage(data,resolution,(10,10))

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

def reconstructors(sda):
    for layer in sda.dALayers:
        reconstructed = layer.reconstructedInput(layer.hiddenValues(sda.x))
        yield theano.function(
            inputs=[sda.x],
            outputs = reconstructed
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

def imageArray2(data,resolution,tileShape=(10,10)):
    try:
        import PIL.Image as Image
    except ImportError:
        import Image
    from utils import tile_raster_images
    image = Image.fromarray(tile_raster_images(X=self.W.get_value(borrow=True).T,
                           img_shape=resolution, tile_shape=tileShape,
                           tile_spacing=(1, 1)))
    return image

def sdaLayerImages(sda,color=False):
    #layer weights
    layers = list(sda.dALayers)
    if color:
        first = layers[0]
        weight = first.W.get_value(borrow=True).T
        print "first layer shape:",weight.shape
        fs = weight.shape
        weight = weight.reshape(fs[0],fs[1]/3,3)
        weight = weight.swapaxes(1,2)
        weight = weight.swapaxes(0,1)
        print "reshaped shape:",weight.shape
        for c,ch in zip(['r','g','b'],weight):
            yield ("weight%s.png" % c),util.makeImage(ch,dataResolution(ch),(10,10))
        layers = layers[1:]
    for i,layer in enumerate(layers):
        weight = layer.W.get_value(borrow=True).T
        resolution = dataResolution(weight)
        yield ("weight%d.png" % i) , layer.genLayerImage(resolution,(10,10))
def isqrtSet(n):
    w = isqrt(n)
    h = n/w
    return w,h
    
def sdaLayerImages2(sda,nChannel):
    #layer weights
    layers = list(sda.dALayers)
    for i,layer in enumerate(layers):
        W = layer.W.get_value(borrow =True).T
        shape = W.shape
        name =("layer%d" % i)
        nInput,nOutput = shape
        print name,shape
        if nChannel == 1:
            w,h = isqrtSet(nOutput)
            yield name,imageArray((W.flatten()[:nInput*w*h]).reshape(nInput,w,h))
        else:
            elements = nOutput/nChannel
            w,h = isqrtSet(elements)
            print w , h
            print w * h * nChannel
            yield name,imageArray((W.flatten()[:nInput*w*h*nChannel]).reshape(nInput,w,h,nChannel))
            
    
def createSdaImages(sda,inputs,color=False):
    """
    return Seq[ImageName,Image]
    """
    print "createSdaImages"
    #layer activations
    for i,comp in enumerate(genCompressors(sda)):
        compressed = comp(inputs)
        img = makeImageOfData(compressed)
        name ="layer%d.png" % i
        yield name,img
    #reconstructed image TODO
    # for i,recon in enumerate(reconstructors(sda)):
    #     yield ("reconstructed%d.png"%i), makeImageOfData(recon(inputs))
    yield "reconstructed.png",makeImageOfData(reconstructor(sda)(inputs))
    for pair in sdaLayerImages(sda,color):
        yield pair

def saveModelImages(modelPath,dstPath,color = False):
    info,sda = util.load(modelPath)
    import train
    x = train.createDataSet(info["dataSet"]).get_value(borrow=True)
    for name,img in createSdaImages(sda,x,color):
        dst = dstPath + "/" + name
        util.ensurePathExists(dst)
        img.save(dst)

def imgScatter(coords,images):
    """
    coords: seq[(x,y)]
    images: seq[Image]
    """
    from   matplotlib.offsetbox import OffsetImage,AnnotationBbox
    print "imageScatter:",images.shape
    offsetImages = map(lambda i: OffsetImage(i,zoom=1),images)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x,y = zip(*coords)
    ax.plot(x,y,'o')
    def genArtists():
        for (x,y),img in zip(coords ,offsetImages):
            ab = AnnotationBbox(img,(x,y),xycoords='data',frameon=False)
            yield ax.add_artist(ab)
    artists = list(genArtists())
    return fig

def genClusterFigures(images,labels):
    """
    images : Seq[Image]
    labels: Seq[Int]
    resolution : tuple[Int] // (28,28,3)
    return : Iterator[Figure]
    """
    from itertools import groupby
    from matplotlib import pyplot as plt, cm as cm
    nIn = len(images)
    print "input shape:",images.shape
    sets = zip(labels,images)
    sets.sort(key = lambda a:a[0])
    count = 0
    groups = [(k,zip(*list(g))[1]) for k,g in groupby(sets,lambda a : a[0])]
    groups.sort(key = lambda p: len(p[1]))
    
    for label,group in groups:
        title = str(label)
        yield imageArray(group,title = title)
        

def imageArray(images,row = 5, col = 5,title = "no title"):

    """
    images:Seq[Image[28,28,3]]
    make a image matrix figure.
    """
    from matplotlib import pyplot as plt, cm as cm
    fig = plt.figure()
    for i, img in enumerate(images[:row*col]):
        ax = fig.add_subplot(row,col,i+1)
        fig.subplots_adjust(hspace = 0.001,wspace = 0.001)
        ax.set_xticklabels(())
        ax.set_yticklabels(())
        ax.tick_params(labelbottom='off',labelleft='off',labelright='off',labeltop='off')
        if len(img.shape) >= 3 :
            ax.imshow(img)
        else:
            ax.imshow(img,cmap = cm.Greys_r)
    return fig

def MDSPlots(images,compressed):
    """
    generator of pyplot figures
    """
    from sklearn.manifold import MDS
    mds = MDS(n_components = 2,dissimilarity = "precomputed")
    print "calculating similarities"
    from scipy.spatial.distance import squareform, pdist
    #similarities = squareform(pdist(compressed,'mahalanobis'))
    similarities = squareform(pdist(compressed,'euclidean'))
    print "fitting mds"
    coords = mds.fit_transform(similarities)

    import visualize as viz
    print "create figure"
    fig = viz.imgScatter(coords,images)
    return fig

    
def MDSPlotTest():
    import json
    import experiment
    resPath = "../experiments/ebook_color_pca_3"
    experiment.experimentCase("../params/ebook_color_pca_28x28_3.json",resPath)
    info = json.loads(util.fileString("../params/ebook_color_pca_28x28_3.json"))
    info = util.dotdict(info)
    x = util.load(resPath+"/x.pkl")
    print x.dtype
    compressed = util.load(resPath+"/compressed.pkl")
    MDSPlots(x,compressed,info.dataSet.shape)
    import matplotlib.pyplot as plt
    fig.savefig()
    print("show figure")
    plt.show()

def sdaImageTest(modelPath):
    info,model = util.load(modelPath)
    print info
    print model
    shape = info["dataSet"]["shape"]
    color = len(shape) == 3
    import matplotlib.pyplot as plt
    for name,fig in sdaLayerImages2(model,3):
        fig.show()
        raw_input()
        
    plt.show()
        
if __name__ == '__main__':
    import sys
    sdaImageTest(sys.argv[1])
    # saveModelImages(sys.argv[1],sys.argv[2],bool(sys.argv[3]))
    # import os
    # models = list(findModels())
    # l = len(models)
    # print "start converting models:",l
    # for i,modelPath in enumerate(models):
    #     if not os.path.isdir(modelPath):
    #         print modelPath
    #         print "converting %dth model of %d models" % (i,l)
    #         convertModelDataToImages(modelPath)
    # print "done! yay!"
