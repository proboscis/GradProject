__author__ = 'kentomasui'
import theano,numpy
from theano import tensor as T


class DenoisingAutoEncoder():
    def __init__(self,
                 numpyRng,
                 theanoRng=None,
                 input=None,
                 nVisible=784,
                 nHidden=500,
                 W=None,
                 bHidden=None,
                 bVisible=None):
        self.nVisible = nVisible
        self.nHidden = nHidden
        if not theanoRng:
            theanoRng = theano.RandomStreams(numpyRng.randint(2**30))
        if not W:
            initialW = numpy.asarray(
                numpyRng.uniform(
                    low=-4 * numpy.sqrt(6. / (nHidden + nVisible)),
                    high=4 * numpy.sqrt(6. / (nHidden + nVisible)),
                    size=(nVisible, nHidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initialW, name='W', borrow=True)

        def shared(name,size):
            return theano.shared(
                value=numpy.zeros(
                    size,
                    dtype=theano.config.floatX
                ),
                name=name,
                borrow=True
            )
        if not bVisible:
            bVisible = shared("bVisible",nVisible)
        if not bHidden:
            bHidden = shared("bHidden",nHidden)

        self.W = W
        self.b = bHidden
        self.bPrime = bVisible
        self.WPrime = self.W.T
        self.theanoRng = theanoRng
        if input is None:
            self.x = T.dmatrix(name="input")
        else:
            self.x = input

        self.params = [self.W,self.b,self.bPrime]

    def hiddenValues(self,input):
        return T.nnet.sigmoid(T.dot(input,self.W) + self.b)

    def reconstructedInput(self,hidden):
        return T.nnet.sigmoid(T.dot(hidden,self.WPrime) + self.bPrime)

    def corruptedInput(self,input,corruptionLevel):
        return self.theanoRng.binomial(size=input.shape,n=1,p=1-corruptionLevel,dtype=theano.config.floatX) * input

    def costFunctionAndUpdates(self,corruptionLevel,learningRate):
        tildeX = self.corruptedInput(self.x,corruptionLevel)
        y = self.hiddenValues(tildeX)
        z = self.reconstructedInput(y)
        L = - T.sum(self.x * T.log(z) + ( 1- self.x) * T.log(1-z),axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost,self.params)
        updates = [
            (param,param - learningRate*gparam)
            for param,gparam in zip(self.params, gparams)
        ]
        return (cost,updates)

    def saveLayerImage(self,filename,resolution=(100,100),tileShape=(10,10)):
        try:
            import PIL.Image as Image
        except ImportError:
            import Image
        from utils import tile_raster_images

        image = Image.fromarray(
            tile_raster_images(X=self.W.get_value(borrow=True).T,
                               img_shape=resolution, tile_shape=tileShape,
                               tile_spacing=(1, 1)))
        import util
        util.ensurePathExists(filename)
        image.save(filename)
