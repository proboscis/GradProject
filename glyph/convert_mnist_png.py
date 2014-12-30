import util
from os import path,makedirs
from scipy import misc
if __name__ == '__main__' :
	dst = "../data/mnist/28x28"
	trainSet = util.loadMnistData()[0][0].get_value(borrow=True)
	print "%d images to be converted" % trainSet.shape[0] 
	for i in xrange(trainSet.shape[0]):
		print "saving %dth image" % i
		dstFile = dst + '/%d.png' % i
		util.ensurePathExists(dstFile)
		misc.imsave(dstFile,trainSet[i].reshape(28,28))
