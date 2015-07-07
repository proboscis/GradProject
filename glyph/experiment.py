__author__ = 'kento'
import util
import train
import visualize
import numpy
import matplotlib
import json
import clustering
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import squareform,pdist
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics

def experiments():
    return [
        ("../params/ebook_color_sda_28x28_2.json","../experiments/ebook_color_sda_2"),
        ("../params/ebook_color_pca_28x28_2.json","../experiments/ebook_color_pca_2"),
        ("../params/mnist_sda_e15.json","../experiments/mnist_sda"),
        ("../params/mnist_pca.json","../experiments/mnist_pca")
    ]

def experimentAll():
    for paramPath,dstPath in experiments().iteritems():
        experimentCase(paramPath,dstPath)

def modelToCompressor(info,model):
    if info["model"]["kind"] == "pca" :
        return model.transform
    if info["model"]["kind"] == "sda" :
        return list(visualize.genCompressors(model))[-1]

def experimentCase(paramPath,resultPath):
    print "experiment param:",paramPath
    print "experiment result:",resultPath
    print "load param"
    info = json.loads(util.fileString(paramPath))
    print "create model"
    print (info)
    info,model = train.evalModel(info)
    print "load dataset"
    info["dataSet"]["numData"] = 1000
    x = util.loadOrCall(resultPath+"/x.pkl",lambda :train.createDataSet(info["dataSet"]).get_value(borrow=True),force = True)
    numClustering = 1000
    x = x[:numClustering]
    print "create compressor"
    print "compressing input shape:",x.shape
    compress = modelToCompressor(info,model)
    compressed = util.loadOrCall(resultPath+"/compressed.pkl",lambda :compress(x),force=True)
    print "create clustering result images"
    #clustering.showInputImageAndClass(x,compressed,clustering.applyDBSCAN,info["dataSet"]["shape"],dstFolder=resultPath+"/clusters")

    labels = clustering.applyDBSCAN(compressed)
    images = x.reshape((numClustering,)+tuple(info["dataSet"]["shape"]))
    clusterImages = visualize.genClusterFigures(images,labels)
    for i,fig in enumerate(clusterImages):
        fig.savefig(resultPath + "/cluster"+str(i))
    print "create mds distribution image"
    print "image shape",images.shape
    print "x shape",x.shape
    print "compressed shape", compressed.shape

    mdsFig = visualize.MDSPlots(images,compressed)
    mdsFig.savefig(resultPath+"/mds")
    #clustering.saveMDSPlots(resultPath+"/mds.png", compressed)
    print "calculate clustering score"
    #score = metrics.silhouette_score(compressed, clustering.applyDBSCAN(compressed), metric='euclidean')
    #util.writeFileStr(resultPath+"/score.txt",str(score))
    print "experiment case done!"

if __name__ == '__main__':
    #experimentCase("../params/tiny.json","../experiments/tiny")
    experimentCase("../params/newsda.json","../experiments/newsda")
    #experimentCase("../params/tinysda.json","../experiments/tinysda")
    #experimentCase(*experiments()[0])
    #experimentAll()
