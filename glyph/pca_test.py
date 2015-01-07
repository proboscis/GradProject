__author__ = 'kento'
import train
from sklearn.decomposition import PCA
if __name__ == '__main__':
    import numpy as np
    import clustering
    import util
    #x = util.loadMnistData()[0][0].get_value(borrow=True)
    x = train.createEbookDataSet({
        "path":"../resized/100x100",
        "shape":(100,100,3),
        "numData":10000,
    }).get_value(borrow=True)
    def l():
        print "create pca"
        pca = PCA(2)
        print "fitting pca"
        pca.fit(x)
        return pca
    pca = util.loadOrCall("../models/pca.pkl",l)
    print "transforming data"
    y = pca.transform(x)
    print "clustering data"
    #clustering.showClustering(y)
    clustering.showInputImageAndClass(x,y,clustering.applyDBSCAN,(100,100,3))
