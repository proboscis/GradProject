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
        return list(model.transform)
    if info["model"]["kind"] == "sda" :
        return list(visualize.genCompressors(model))

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
    util.ensurePathExists(resultPath+"/param.json")
    with open(resultPath+"/param.json","w") as f:
        json.dump(info,f)
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
    compressors = modelToCompressor(info,model)
    for layer, compressor in enumerate(compressors):
        layer = str(layer)
        compressed = util.loadOrCall(resultPath+"/layer"+layer+"/compressed.pkl",lambda :compressor(x),force=not useCache)
        print "create clustering result images"
    #clustering.showInputImageAndClass(x,compressed,clustering.applyDBSCAN,info["dataSet"]["shape"],dstFolder=resultPath+"/clusters")

        #estimate eps from knn
        from sklearn.neighbors import NearestNeighbors
        def knn():
            nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(compressed)
            return nbrs.kneighbors(compressed)
        dist,indices = util.loadOrCall(resultPath+"/layer"+layer+"/nn.pkl",knn,force = not useCache)
        dist = numpy.swapaxes(dist,0,1)
        avgDist =  numpy.mean(dist[1])
        avgDist = max(0,avgDist)
        print "average dist:" ,avgDist
        for eps in numpy.arange(avgDist/3,avgDist*2,avgDist/3):
            print "clustering eps:",eps
            labels = clustering.applyDBSCAN(compressed,eps)
            images = x.reshape((numClustering,)+tuple(info["dataSet"]["shape"]))
            clusterImages = visualize.genClusterFigures(images,labels)
            level = ("%.3f" % eps).replace(".","_")
            for i,fig in enumerate(clusterImages):
                dirName = resultPath +"/layer"+layer+"/"+level+ "/cluster"+str(i)
                util.ensurePathExists(dirName)
                fig.savefig(dirName)
        print "create mds distribution image"
        print "image shape",images.shape
        print "x shape",x.shape
        print "compressed shape", compressed.shape
        mdsFig = visualize.MDSPlots(images,compressed)
        figPath = resultPath+"/layer"+layer+"/mds"
        util.ensurePathExists(figPath)
        mdsFig.savefig(figPath)
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
