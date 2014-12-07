__author__ = 'kentomasui'
import dA,util,theano,numpy,os,sys,time
from theano import tensor as T
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image
from theano.tensor.shared_randomstreams import RandomStreams


if __name__ == '__main__':
    trainingEpochs = 15
    batchSize = 20
    data = util.loadMnistData("mnist.pkl.gz")
    nTrainBatches = data[0][0].get_value(borrow=True).shape[0]/batchSize
    index = T.lscalar('index')
    x = T.dmatrix('x')
    rng = numpy.random.RandomState(123)
    theanoRng = RandomStreams(rng.randint(2 ** 30))
    da = dA.DenoisingAutoEncoder(
        numpyRng=rng,
        theanoRng=theanoRng,
        input = x,
        nVisible=28*28,
        nHidden=500
    )
    cost,updates = da.costFunctionAndUpdates(corruptionLevel=0,learningRate=0.1)
    trainModel = theano.function([index],cost,updates=updates,givens={
        x:data[0][0][index*batchSize:(index+1)*batchSize]
    })

    start_time = time.clock()


    for epoch in xrange(trainingEpochs):
        c = []
        for i in xrange(nTrainBatches):
            c.append(trainModel(i))
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = time.clock()
    training_time = (end_time - start_time)
    print >> sys.stderr, ('The no corruption code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                       img_shape=(28, 28), tile_shape=(10, 10),
                       tile_spacing=(1, 1)))
    image.save('filters_corruption_03.png')