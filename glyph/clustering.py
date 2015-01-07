__author__ = 'kento'
import util
import train
import visualize
import numpy
import matplotlib
from sklearn.cluster import KMeans
import itertools
def showClustering(data):
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
        print "plotting %dth cluster" % k
        print "label type" ,labels, type(labels)
        print "members are:", members, type(members)
        print "data[members,0]",data[members,0],type(data[members,0])
        center = centers[k]
        plt.plot(data[members,0],data[members,1],col +'.')
        plt.plot(center[0],center[1],'o',markerfacecolor=col,
                 markeredgecolor = 'k',markersize = 14)
    plt.title("clusters")
    plt.show()

def applyKMeans(data):
    kmeans = KMeans()
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def applyDBSCAN(data):
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN()
    dbscan.fit(data)
    return dbscan.labels_

def showInputImageAndClass(x,y,clusterizer,resolution,gray=False):
    from itertools import groupby
    from matplotlib import pyplot as plt, cm as cm
    labels = clusterizer(y)
    nIn,length = x.shape
    print "input shape:",x.shape
    print "given resolution:",resolution
    sets = zip(labels,x)
    sets.sort(key = lambda a:a[0])
    count = 0
    for k,g in groupby(sets,lambda a : a[0]):
        count += 1
        if count > 25:
            break
        fig = plt.figure(str(k))
        group = list(g)
        for i,img in enumerate(group[:25]):
            fig.add_subplot(5,5,i)
            if gray:
                plt.imshow(img[1].reshape(resolution),cmap=cm.Greys_r)
            else:
                plt.imshow(img[1].reshape(resolution))
    plt.title("clustering result")
    plt.show()

def showResult(groups):
    import matplotlib
    from scipy import misc
    from matplotlib import pyplot as plt, cm as cm
    for k,g in groups:
        fig = plt.figure(k)
        for i,imgPath in enumerate(list(g)[:25]):
            img = misc.imread("../thumbnails/"+imgPath[1])
            fig.add_subplot(5,5,i)
            plt.imshow(img,cmap=cm.Greys_r)
    plt.title("clustering result")
    plt.show()

def showPlots(data):
    from pylab import *
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data[0],data[1],'ro')
    plt.show()

if __name__ == '__main__':
    from os import listdir
    from itertools import groupby,islice
    #pkl = "../models/ebooks/s1000c0.0l0.0e10s1000c0.0l0.0e45s10c0.0l0.0e45b1.pkl"
    pkl = "/home/kento/Documents/GradProject/models/ebooks/color/colors_28x28_2.pkl"
    info,model = util.load(pkl)
    inputs = train.createEbookDataSet(info["dataSet"]).get_value(borrow=True)
    inputPaths = listdir(info["dataSet"]["path"])
    compress = list(visualize.genCompressors(model))[-1]
    compressed = compress(inputs)

    # from sklearn.decomposition import PCA
    # pca = PCA(2)
    # pca.fit(compressed)
    # y = pca.transform(compressed)
    print "data shape",compressed.shape
    #showInputImageAndClass(inputs,y,applyKMeans,(28,28),True)
    showInputImageAndClass(inputs,compressed,applyDBSCAN,(28,28,3),False)
    #showClustering(y)
    #showPlots(y.swapaxes(0,1))
    # labels = applyKMeans(compressed)
    # zipped = zip(labels,inputPaths)
    # zipped.sort(key=lambda a: a[0])
    # grouped = groupby(zipped,lambda a : a[0])
    # showResult(grouped)

