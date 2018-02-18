import os
import math

from PIL import Image
from pymongo import MongoClient

import cv2
import numpy as np
from matplotlib import pyplot as plt

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)

client = MongoClient()
ImageDb = client['photoMosaicTest']
#os.chdir(r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images')
data_dir = r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\thumbnails'
data_dir2 = r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images'

SIZE = 256
tileSize = 32
img = cv2.imread(os.path.join(data_dir2, '4 - Fyt8Egq.jpg'))
mosaicBase = cv2.resize( img,(SIZE, SIZE), interpolation = cv2.INTER_AREA)
# for i in range(res.shape[0]):
#     for j in range(res.shape[1]):
#         for k in range(res.shape[2]):
#             res[j,i,k]

fullDb = ImageDb.Images.find()
dbrgb = []
imageFiles = []
fullDb = list(fullDb)
for doc in range(fullDb.__len__()):
    dbrgb.append([fullDb[doc]['r'],fullDb[doc]['g'],fullDb[doc]['b']])
    imageFiles.append(fullDb[doc]['filename'])

rgb_array = np.array(dbrgb)
print(rgb_array.shape)
print (len(imageFiles))


r,g,b = rgb_array[123,:]
rgb = np.array([r,g,b])
print(rgb.shape)
mainImage = np.zeros((SIZE*tileSize,SIZE*tileSize,3))

finalImageArray = []
for i in range(mosaicBase.shape[0]):
    for j in range(mosaicBase.shape[1]):
        b,g,r = mosaicBase[j,i,:]
        rgb = np.array([r, g, b])
        diff_array = np.sqrt(np.sum(np.square(rgb_array - rgb),axis=1))
        min_ele = np.argmin(diff_array)
        img = cv2.imread(os.path.join(data_dir, imageFiles[min_ele]))
        res = cv2.resize(img, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
        mainImage[j*tileSize:j*tileSize+tileSize,i*tileSize:i*tileSize+tileSize,:] = res

        print("min is= ", min_ele)


# def getdiff(argArr):
#     imgColor = argArr[0]
#     dbColor = argArr[1]
#     r1 = imgColor[0]
#     g1 = imgColor[1]
#     b1 = imgColor[2]
#     r2 = dbColor[0]
#     g2 = dbColor[1]
#     b2 = dbColor[2]
#     return math.sqrt(pow((r1-r2),2) + pow((g1-g2),2) + pow((b1-b2),2))


# imageColorArray = []
# finalImageArray = []
# for i in range(mosaicBase.shape[0]):
#     for j in range(mosaicBase.shape[1]):
#         r,g,b = mosaicBase[j,i,:]
#         imageColorArray.append([r,g,b])
#
# for color in imageColorArray:
#     diffArr = []
#     for dbcolor in dbrgb:
#         diffArr.append([list(color), list(dbcolor)])
#     diffArr = list(map(getdiff, diffArr))
#     print(color)
#     finalImageArray.append(imageFiles[diffArr.index(min(diffArr))])
# print('finished', finalImageArray.__len__())



# init = True
# for i in range(mosaicBase.shape[0]):
#     for j in range(mosaicBase.shape[1]):
#         r,g,b = mosaicBase[j,i,:]
#         fileData = ImageDb.Images.aggregate([{
#                                     '$project': {
#                                         "diff": {
#                                             "$sqrt": {
#                                                 "$add": [
#                                                     { "$pow": [{ "$subtract": [int(r), "$b"] }, 2] },
#                                                     { "$pow": [{ "$subtract": [int(g), "$g"] }, 2] },
#                                                     { "$pow": [{ "$subtract": [int(b), "$r"] }, 2] }
#                                                 ]
#                                             }
#
#                                         },
#                                         "doc": "$$ROOT"
#                                     },
#
#                                 },
#                                 { "$sort": { "diff": 1 } },
#                                 { "$limit": 1 }
#                             ])
#         if j == 0:
#             print(i)
#         currentfile = list(fileData)[0]['doc']['filename']
#         img = cv2.imread(os.path.join(data_dir, currentfile))
#         res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
#         mainImage[j*64:j*64+64,i*64:i*64+64,:] = res
#
saveFileName = os.path.abspath(os.path.join(data_dir,'..', 'newMosaicImage'+str(SIZE)+'2.jpg'))
print(saveFileName)
cv2.imwrite(saveFileName,mainImage)
print('saved')