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

def experimentAll():
	experiments = {
		"../params/ebook_color_sda_28x28_2.json":"../experiments/ebook_color_sda_2",
		"../params/ebook_color_pca_28x28_2.json":"../experiments/ebook_color_pca_2",
		"../params/mnist_sda_e15.json":"../experiments/mnist_sda",
		"../params/mnist_pca.json":"../experiments/msnit_pca"
	}
	for paramPath,dstPath in experiments.iteritems():
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
	info,model = train.evalModel(info)
	print "load dataset"
	x = train.createDataSet(info["dataSet"]).get_value(borrow=True)
	print "create compressor"
	compress = modelToCompressor(info,model)
	compressed = compress(x)
	print "create clustering result images"
	clustering.showInputImageAndClass(x,compressed,clustering.applyDBSCAN,info["dataSet"]["shape"],dstFolder=resultPath+"/clusters")
	print "create mds distribution image"
	clustering.saveMDSPlots(resultPath+"/mds.png", compressed)
	print "calculate clustering score"
	score = metrics.silhouette_score(compressed, clustering.applyDBSCAN(compressed), metric='euclidean')
	util.writeFileStr(resultPath+"/score.txt",str(score))
	print "experiment case done!"

if __name__ == '__main__':
	experimentAll()
