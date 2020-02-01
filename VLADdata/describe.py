# Feature Extraction

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import mlcrate as mlc
import cv2

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help = "Path to the directory that contains the images")
ap.add_argument("-n", "--descriptor", required=True,
	help = "descriptor = SURF, SIFT or  ORB")
ap.add_argument("-o", "--output", required=True,
	help = "Path to where the computed descriptors will be stored")
ap.add_argument("-t", "--threads", default=1, help="Number of threads to use for descriptor extraction")
args = vars(ap.parse_args())

#reading arguments
path = args["dataset"]
descriptorName = args["descriptor"]
output = args["output"]
threads = int(args["threads"])

#computing the descriptors
dict = {"SURF": describeSURF, "SIFT": describeSIFT, "ORB": describeORB}
descriptors = getDescriptors(path, dict[descriptorName], threads)

print('Writing descriptors to disk ...')

# Write descriptors to disk
mlc.save(descriptors, output + '.pickle')
