from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import pickle
import argparse
import glob
import cv2


#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--VLADdescriptorPath", required = True,
	help = "Path to the file that contains the VLAD descriptors")
ap.add_argument("-l", "--leafSize", required = True,
	help = "Size of the leafs of the Ball tree")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())


#args
path = args["VLADdescriptorPath"]
leafSize = int(args["leafSize"])
output = args["output"]

#estimating VLAD descriptors for the whole dataset
print("indexing VLAD descriptors from " + path + " with a ball tree:")

#load the vlad descriptors VD=[imageID, VLADdescriptors, pathToImageDataSet]
with open(path, 'rb') as f:
    VLAD_DS = pickle.load(f)

imageID = VLAD_DS[0]
V = VLAD_DS[1]
pathImageData = VLAD_DS[2]

tree = indexBallTree(V, leafSize)
#output
file = output + ".pickle"

with open(file, 'wb') as f:
	pickle.dump([imageID, tree, pathImageData], f, pickle.HIGHEST_PROTOCOL)

print("The ball tree index is saved at " + file)

