from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, SGD
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pydicom
import os
from matplotlib import pyplot, cm
import cv2
from sklearn import preprocessing

# path loading
PathDicom = "C://Users//arvin//Documents//CNN Project//ctscan_cleaned"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

test_array = []

#scaler = MinMaxScaler()
# Loading and resizing image
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
    res = img_resize(pydicom.read_file(filenameDCM).pixel_array,368)
    
    # store the raw image data
    test_array.append(preprocessing.scale(res))

test_finale = np.asarray(test_array).reshape(len(lstFilesDCM),368,368)
test_finale = test_finale.reshape(len(lstFilesDCM),368,368,1).astype('float32')
sequence_list = [[test_finale[i-2],test_finale[i-1],test_finale[i]] for i in range(2,test_finale.shape[0],3)]
sequence_list = np.asarray(sequence_list)
#sequence_list = sequence_list[:269]
labels = np.zeros(685, dtype=int)
labels[:188] = 1
#labels = to_categorical(labels, num_classes = 2)

X_train, X_test, y_train, y_test = train_test_split(sequence_list,labels, test_size=0.4, random_state=1)
print(len(X_train))
print(len(X_test))


model = load_model('C:/Users/arvin/Documents/new_2_1_0_inverse_cate_5050_sgd_GRU_0.0001.h5')
#print(model.evaluate(X_test[:30],y_test[:30]))
#print(model.metrics_names)
#label_hat=model.predict(X_test[60:80]),axis=1
#print(label_hat)
#print(y_test[40:60])
#print(label_hat.flatten())

temp = []
sumation = 0
batch = 0
for i in range(2,len(X_test),2):
    batch +=1
    scores = model.evaluate(X_test[i-2:i],y_test[i-2:i])
    sumation+=scores[1]

print(sumation/batch)

for i in range(2,len(X_test),2):
    label_hat=model.predict(X_test[i-2:i])
    flat = label_hat.flatten()
    temp.extend(flat)

y_pred = []
for i in temp:
    if i<0.5:
        y_pred.append(0)
    else:
        y_pred.append(1)

print(y_pred)
    



