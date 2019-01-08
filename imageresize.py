import pydicom
import os
import numpy
from matplotlib import pyplot, cm
import cv2

PathDicom = "C://Users//arvin//Documents//CNN Project//cleaned"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

lstFilesDCM = lstFilesDCM[0:100]

RefDs = pydicom.read_file(lstFilesDCM[0])
ConstPixelDims = (len(lstFilesDCM),368, 368)

# The array is sized based on 'ConstPixelDims'
arraydicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

def img_resize(img,size):
    desired_size = size
    im = img
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_im

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = pydicom.read_file(filenameDCM).pixel_array
    res = img_resize(ds,368)
    
    # store the raw image data
    arraydicom[:,:,:] = res

print(arraydicom[0])
pyplot.figure(dpi=150)
pyplot.imshow(arraydicom[0],cmap = 'gray')
pyplot.show()
