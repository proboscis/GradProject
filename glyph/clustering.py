__author__ = 'kento'
import util
import train
import visualize
import numpy
import matplotlib
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import squareform,pdist
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
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

def showInputImageAndClass(x,y,clusterizer,resolution,gray=False,dstFolder=None):
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
        if dstFolder is not None:
            dst = dstFolder+"/%s.png" % str(k)
            util.ensurePathExists(dst)
            plt.savefig(dst ,bbox_inches='tight')
    if dstFolder is None:
        plt.title("clustering result")
        plt.show()
    plt.close("all")

def showImages(images,tile = (5,5)):
    from itertools import groupby
    from matplotlib import pyplot as plt, cm as cm
    #assume (nImage,w,h,ch)
    fig = plt.figure()

    for i,img in enumerate(images[:tile[0]*tile[1]]):
        fig.add_subplot(tile[0],tile[1],i)
        plt.imshow(img)
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
    print "plotting points.."
    plt.figure()
    plt.plot(data[0],data[1],'ro')
    plt.show()

def savePlots(dst,data):
    print "saving plotted points.."
    util.ensurePathExists(dst)
    plt.figure()
    plt.plot(data[0],data[1],'ro')
    plt.savefig(dst,bbox_inches='tight')

def saveMDSPlots(dst,data):
    mds = MDS(dissimilarity="precomputed")
    distances = squareform(pdist(data[:1000],'euclidean'))
    points = mds.fit_transform(distances)
    print "mds done!"
    print "points shape", points.shape
    savePlots(dst,points.swapaxes(0,1))

def showMDSPlots(data):
    mds = MDS(dissimilarity="precomputed")
    distances = squareform(pdist(data[:1000],'euclidean'))
    points = mds.fit_transform(distances)
    print "mds done!"
    print "points shape", points.shape
    showPlots(points.swapaxes(0,1))

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
    #showInputImageAndClass(inputs,compressed,applyDBSCAN,(28,28,3),False)
    #showClustering(y)
    #showPlots(y.swapaxes(0,1))
    showMDSPlots(compressed)
    print "score:",metrics.silhouette_score(compressed, applyDBSCAN(compressed), metric='euclidean')

    # labels = applyKMeans(compressed)
    # zipped = zip(labels,inputPaths)
    # zipped.sort(key=lambda a: a[0])
    # grouped = groupby(zipped,lambda a : a[0])
    # showResult(grouped)

