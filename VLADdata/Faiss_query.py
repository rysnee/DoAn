import itertools
import argparse
import glob
import cv2
import pickle
from sklearn.neighbors.kd_tree import KDTree
import numpy as np
import os

from bagofvisualwords import BagOfVisualWords

from VLADlib.VLAD import *
from VLADlib.Descriptors import *

import faiss

def get_query(groundtruth_dir):
	image_dir = 'dataset'
	imgs = []
	names = []
	
	for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
		name = f.replace("query", "ranklist")
		name = name.replace(groundtruth_dir, "output")
		query_name, x, y, w, h = open(f).read().strip().split(' ')
		query_name = query_name.replace('oxc1_', '')
		
		img = cv2.imread(os.path.join(image_dir, '%s.jpg' % query_name)) # BGR
        
		# Crop
		x, y, w, h = map(float, (x, y, w, h))
		x, y, w, h = map(lambda d: int(round(d)), (x, y, w, h))
		
		img = img[y:y+h, x:x+w]
		# img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
		
		names.append(name)
		imgs.append(img)
		
	return names, imgs

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-gt", "--groundtruth", required = True,
	help = "Path of a groundtruth image")
ap.add_argument("-r", "--retrieve", required = True,
	help = "number of images to retrieve")
ap.add_argument("-d", "--descriptor", required = True,
	help = "descriptors: SURF, SIFT or ORB")
ap.add_argument("-dV", "--visualDictionary", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-vw", "--visualWord", required = True,
	help = "Path of the visualWord")
ap.add_argument("-l", "--leafsize", required = True,
	help = "Leaf Size")

args = vars(ap.parse_args())

# arg
pathVD = args["visualWord"]
k = int(args["retrieve"])
descriptorName = args["descriptor"]
pathGT = args["groundtruth"]
pathV = args["visualDictionary"]
l = args["leafsize"]

# load
with open(pathVD, 'rb') as f:
    vocab = pickle.load(f)   

# training KD-tree
training = np.asarray([i[1].toarray()[0].tolist() for i in vocab]).astype('float32')

# faiss inital
dimension = 512
nlist = 10

quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)

print(len(training[0]))
index.train(training)
index.add(training) 


# initial BoW
with open(pathV, 'rb') as g:
    visualDictionary = pickle.load(g) 

bovw = BagOfVisualWords(visualDictionary.cluster_centers_)

names, queries = get_query(pathGT)

for i, q in enumerate(queries):

    #compute descriptors
    kp, descriptor = describeORB(q)

    # represent at BoW
    hist = bovw.describe(descriptor)
    query = np.asarray([hist.toarray()[0].tolist()]).astype('float32')

    print(query)
    dist, ind = index.search(query, k)  
    
    ind = list(itertools.chain.from_iterable(ind))
    
    for item in ind:
        print(vocab[item][0])

    name = names[i]
    with open(name, 'w') as f:
        print(vocab[item][0])
        for item in ind:
            re = vocab[item][0].replace("dataset/", "")
            re = re.replace(".jpg", "")
            f.write("%s\n" % re) 






