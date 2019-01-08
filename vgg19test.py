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
    res = img_resize(pydicom.read_file(filenameDCM).pixel_array,512)
    
    # store the raw image data
    test_array.append(preprocessing.scale(res))

test_finale = np.asarray(test_array).reshape(len(lstFilesDCM),512,512)
test_finale = test_finale.reshape(len(lstFilesDCM),512,512,1).astype('float32')
sequence_list = [[test_finale[i-2],test_finale[i-1],test_finale[i]] for i in range(2,test_finale.shape[0],3)]
sequence_list = np.asarray(sequence_list)
#sequence_list = sequence_list[:269]
labels = np.zeros(685, dtype=np.int32)
labels[:188] = 1
labels = np.asarray(labels,dtype=np.int32)
#labels = to_categorical(labels, num_classes = 2)

X_train, X_test, y_train, y_test = train_test_split(sequence_list,labels, test_size=0.4, random_state=1)
#scaler = StandardScaler()

#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit(X_test)
#y_train = y
#y_test = y


model = Sequential()
model.add(TimeDistributed(Conv2D(64,(3,3),input_shape = (3,None,None,1), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(Conv2D(64,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = 2)))

model.add(TimeDistributed(Conv2D(128,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(Conv2D(128,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = 2)))
#model.add(TimeDistributed(Dropout(0.1))

model.add(TimeDistributed(Conv2D(256,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(Conv2D(256,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(Conv2D(256,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(Conv2D(256,(3,3), activation = 'relu',strides = 1, padding = 'same')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = 2)))


#model.add(TimeDistributed(Dropout(0.2))

model.add(TimeDistributed(Dense(128,activation ='relu')))
#model.add(TimeDistributed(Dropout(0.2))

#model.add(TimeDistributed(Dense(128, activation = 'relu')))

model.add(TimeDistributed(GlobalAveragePooling2D()))

model.add(GRU(16))
#model.add(Dense(256, activation = 'relu'))
model.add(Dense(1,activation='sigmoid'))

opt = SGD(lr=0.0001)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'] )


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 4, batch_size = 2)

save_dir = "C:/Users/arvin/Documents"
model_name = 'new_2_1_0_inverse_binary_5050_sgd_GRU_0.0001.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

#model = load_model('C:/temp/sgd_0.0001.h5')
#print(model.evaluate(X_test[:20],y_test[:20]))
#print(model.metrics_names)