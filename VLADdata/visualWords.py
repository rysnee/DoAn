from bagofvisualwords import BagOfVisualWords
from bovwindexer import BOVWIndexer
import numpy as np
import argparse
import glob
import cv2
import os, os.path

from VLADlib.VLAD import *
from VLADlib.Descriptors import *

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to image dataset")
ap.add_argument("-dV", "--visualDictionaryPath", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-n", "--descriptor", required = True,
	help = "descriptor = SURF, SIFT or  ORB")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where BoW will be stored")
args = vars(ap.parse_args())

#args
path = args["dataset"]
pathVD = args["visualDictionaryPath"]
descriptorName = args["descriptor"]
output = args["output"]

#estimating VLAD descriptors for the whole dataset
print("Estimating VLAD descriptors using " + descriptorName + " for dataset: /" + path + " and visual dictionary: /" + pathVD)

with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f) 

bovw = BagOfVisualWords(visualDictionary.cluster_centers_)

dict = {"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  

re = []
for imagePath in glob.glob(path + "/*.jpg"):
    print(imagePath)
    im = cv2.imread(imagePath)
    kp, des = dict[descriptorName](im)

    if des is not None:
        hist = bovw.describe(des)
        re.append(hist)

#output
file = output + ".pickle"

with open(file, 'wb') as g:
    pickle.dump(re, g)


