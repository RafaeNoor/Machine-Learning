import os
import numpy as np
import cv2
import pickle

dataSet = {} # dictionary to write as pkl file 


rootDir = './'+'101_ObjectCategories'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % os.path.basename(dirName))
    count = 0
    dataSet[os.path.basename(dirName)] = []

    for fname in fileList:
        
        print('\t%s' % dirName+'/'+fname)
        
        if(not(fname == '.DS_Store')):
            img = cv2.imread(dirName+'/'+fname,0)
            rowSize = len(img)
            colSize = len(img[0])
            rowScale = 32.0/rowSize
            colScale = 32.0/colSize
            #  grayscale each image then resize it into a 32x 32 square
            resized = cv2.resize(img,None,fx = colScale,fy = rowScale,interpolation = cv2.INTER_CUBIC) 

            linear = []
            for arr in resized:
                for val in arr:
                    linear.append(val)

            lol = dataSet[os.path.basename(dirName)]
            newList = []
            for ls in lol:
                newList.append(ls)

            newList.append(linear)

            dataSet[os.path.basename(dirName)]= newList 



dataSet.pop("101_ObjectCategories",None)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


print("Begin saving...")
save_obj(dataSet,'dict')
print("Begin reading...")
readSet = load_obj('dict')
print("Done reading...")
