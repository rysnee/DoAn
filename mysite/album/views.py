from django.shortcuts import render, redirect
from django.contrib import messages

from .models import Photo, ResultPhoto
from .forms import PhotoForm

from .VLADlib.VLAD import *
from .VLADlib.Descriptors import *
import itertools
import argparse
import glob
import cv2

def retrieval_k(img, k):
    descriptorName = "ORB"
    pathVD = "VLADdata/visualDictionary/visualDictionary2ORB.pickle"
    treeIndex = "VLADdata/ballTreeIndexes/index_ORB_W2.pickle"

    #load the index
    with open(treeIndex, 'rb') as f:
        indexStructure = pickle.load(f)

    #load the visual dictionary
    with open(pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)     

    imageID = indexStructure[0]
    tree = indexStructure[1]
    pathImageData = indexStructure[2]

    # computing descriptors
    dist, ind = query(img, k, descriptorName, visualDictionary, tree)
    dist = dist.flatten()
    ind = ind.flatten()
    if dist is 0:
        return

    print(dist)
    print(ind)
    #ind = list(itertools.chain.from_iterable(ind))

    k_image = []
    # loop over the results
    for i in ind:
        # load the result image and display it
        k_image.append(imageID[i])
        
    return k_image, dist

def photo_list(request):
    photos = Photo.objects.all()   
    result_photos = ResultPhoto.objects.all()
    if request.method == 'POST':
        ResultPhoto.objects.all().delete()
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():     
            img = form.save()
            image = img.file.path
          
            #Query the image
            result, scores = retrieval_k(image, 10)
            print('result =', result)
            print('scores =', scores)
            if(result == None):                       
                messages.info(request, 'Image is too small!')
            else:               
                result_photos = ResultPhoto.objects.all()
                print(ResultPhoto.objects.all().count())
                for i in range(len(result)):
                    r = ResultPhoto(index=result[i], score=scores[i])
                    r.save()               
                      
            return render(request, 'album/result_photo_list.html', {'form': form, 'photos': photos, 'result_photos': result_photos})
    else:
        form = PhotoForm()
    return render(request, 'album/photo_list.html', {'form': form, 'photos': photos, 'result_photos': result_photos})
    
def result_photo_list(request):
    return 
