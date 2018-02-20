import os
import datetime
from pymongo import MongoClient
import matplotlib as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)

client = MongoClient()
ImageDb = client['photoMosaicTest']
#os.chdir(r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images')
data_dir = r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\thumbnails'
data_dir2 = r'D:\Alex laptop\source_images'

# image settings
SIZE = 128
tileSize = 32
weight = np.array([1.0,1.0,1.0])
# targetImageFile = '1 - PA15xHi.jpg'
# targetImageFile = '6 - cYsggBH.jpg'
targetImageFile = 'Image uploaded from iOS.jpg'
targetColorSpace = cv2.COLOR_BGR2YCR_CB
databaseColorSpace = cv2.COLOR_RGB2YCR_CB

img = cv2.imread(os.path.join(data_dir2, targetImageFile))

width = int(img.shape[0]/tileSize)
height = int(img.shape[1]/tileSize)
img = img[:width*tileSize, :height*tileSize, :]

mosaicBase = cv2.resize( img,(height, width), interpolation = cv2.INTER_AREA)
mosaicBase = cv2.cvtColor(mosaicBase, targetColorSpace)
print (mosaicBase.shape)
mosaicBase = mosaicBase.reshape(-1,3)
print (mosaicBase.shape)
fullDb = ImageDb.Images.find()
dbrgb = []
imageFiles = []
fullDb = list(fullDb)
for doc in range(fullDb.__len__()):
    dbrgb.append([fullDb[doc]['r'],fullDb[doc]['g'],fullDb[doc]['b']])
    imageFiles.append(fullDb[doc]['filename'])

rgb_array = np.uint8(dbrgb)
lab_arr = rgb_array.reshape((1,-1,3))

lab_arr = cv2.cvtColor(lab_arr, databaseColorSpace)
pixel_arr = np.float32(lab_arr.reshape((-1,3)))

mainImage = np.zeros((width*tileSize,height*tileSize,3))
errorImage = np.zeros((height,width),dtype=np.float32)
finalImageArray = []

def addImage(mainImage, errorImage, pixel_arr, mosaicBase):
    def returnFunc(index):
        r, g, b = mosaicBase[index,:]
        i, j = np.unravel_index(index, (width, height))
        rgb = np.array([r, g, b])
        diff_array = np.sqrt(np.sum(np.square(pixel_arr - rgb) * weight, axis=1))
        min_ele = np.argmin(diff_array)
        #errorImage[i, j] = diff_array.min()
        img = cv2.imread(os.path.join(data_dir, imageFiles[min_ele]))
        res = cv2.resize(img, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
        mainImage[i * tileSize:i * tileSize + tileSize, j * tileSize:j * tileSize + tileSize, :] = res
        return True
    return returnFunc

pool.map(addImage(mainImage, errorImage, pixel_arr,mosaicBase), range(mosaicBase.shape[0]))

# for i in tqdm(range(mosaicBase.shape[0])):
#     for j in range(mosaicBase.shape[1]):
#         r,g,b = mosaicBase[i,j,:]
#         rgb = np.array([r, g, b])
#         diff_array = np.sqrt(np.sum(np.square(pixel_arr - rgb) * weight,axis=1))
#         min_ele = np.argmin(diff_array)
#         errorImage[j,i] = diff_array.min()
#         img = cv2.imread(os.path.join(data_dir, imageFiles[min_ele]))
#         res = cv2.resize(img, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
#         mainImage[i*tileSize:i*tileSize+tileSize,j*tileSize:j*tileSize+tileSize,:] = res

datstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
saveFileName = os.path.abspath(os.path.join(data_dir,'..', 'mosaicBuild3_'+str(SIZE)+'_'+datstr+'_'+str(weight)+'.jpg'))
print(saveFileName)
cv2.imwrite(saveFileName,mainImage)
# plt.imshow(np.uint8(mainImage[:,:,::-1]))
# plt.show()
# print(errorImage.min(), errorImage.max())
# scaledError = (errorImage - errorImage.min())/(errorImage.max() - errorImage.min())
# plt.imshow(np.uint8(255.*scaledError))
# plt.show()
# plt.hist(errorImage.flatten())
# plt.show()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(.1)
print(s.getvalue())