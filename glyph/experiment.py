#! /usr/bin/env python
__author__ = 'kento'
import util
import train
import visualize
import numpy
import matplotlib
matplotlib.use('Agg')
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

def visualizeModel(info,model,path):
    if info["model"]["kind"] == "pca":
        print "no visualization for PCA"
    if info["model"]["kind"] == "sda":
        nChannel = 3 if len(info["dataSet"]["shape"]) == 3 else 1
        for name,fig in visualize.sdaLayerImages2(model,nChannel):
            fig.savefig(path+"/"+name)

def experimentCase(paramPath,resultPath,useCache = True):
    print "experiment param:",paramPath
    print "experiment result:",resultPath
    print "useCache:",useCache
    print "load param"
    info = json.loads(util.fileString(paramPath))
    print "create model"
    print (info)
    info,model = train.evalModel(info,useCache)
    print "load dataset"
    numClustering = 300
    info["dataSet"]["numData"] = numClustering
    x = util.loadOrCall(resultPath+"/x.pkl",lambda :train.createDataSet(info["dataSet"]).get_value(borrow=True),force = not useCache)
    x = x[:numClustering]
    print "create compressor"
    print "compressing input shape:",x.shape
    compress = modelToCompressor(info,model)
    compressed = util.loadOrCall(resultPath+"/compressed.pkl",lambda :compress(x),force=not useCache)
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
    print "visualize layers"
    visualizeModel(info,model,resultPath)
    print "experiment case done!"

if __name__ == '__main__':
    import sys
    args = sys.argv
    experimentCase(args[1],args[2],args[3] == "True" if len(args) > 3 else True)
    #experimentCase("../params/tiny.json","../experiments/tiny")
    #experimentCase("../params/newsda.json","../experiments/newsda")
    #experimentCase("../params/newsda_mnist.json","../experiments/newsda_mnist")
    #experimentCase("../params/tinysda.json","../experiments/tinysda")
    #experimentCase(*experiments()[0])
    #experimentAll()