import itertools
import argparse
import glob
import cv2
import pickle
from sklearn.neighbors.kd_tree import KDTree
import numpy as np

from bagofvisualwords import BagOfVisualWords

from VLADlib.VLAD import *
from VLADlib.Descriptors import *

pathVD = "visualWords/visualWords.pickle"
with open(pathVD, 'rb') as f:
    vocab = pickle.load(f)   

training = np.asarray([i.toarray()[0].tolist() for i in vocab])
tree = KDTree(training, leaf_size=2)

image = 'dataset/3.jpg'
im = cv2.imread(image)

# initial BoW
pathVD = 'visualDictionary/visualDictionary2ORB.pickle'
with open(pathVD, 'rb') as g:
    visualDictionary = pickle.load(g) 

bovw = BagOfVisualWords(visualDictionary.cluster_centers_)

#compute descriptors
kp, descriptor = describeORB(im)

# represent at BoW
hist = bovw.describe(descriptor)
query = np.asarray(hist.toarray()[0].tolist())

print("Query = ", query)

dist, ind = tree.query(query.reshape(1, -1), k=3) 
print(ind)




