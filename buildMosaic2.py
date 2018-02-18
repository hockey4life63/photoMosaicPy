import os
import datetime
from pymongo import MongoClient
import matplotlib as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

# from multiprocessing.dummy import Pool as ThreadPool
# pool = ThreadPool(4)

client = MongoClient()
ImageDb = client['photoMosaicTest']
#os.chdir(r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images')
data_dir = r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\thumbnails'
data_dir2 = r'D:\Alex laptop\source_images'

# image settings
SIZE = 256
tileSize = 16
weight = np.array([1.0,0.0,0.0])
# targetImageFile = '1 - PA15xHi.jpg'
targetImageFile = '3693 - rafwmZt.jpg'

img = cv2.imread(os.path.join(data_dir2, targetImageFile))

mosaicBase = cv2.resize( img,(SIZE, SIZE), interpolation = cv2.INTER_AREA)
mosaicBase = cv2.cvtColor(mosaicBase, cv2.COLOR_BGR2Lab)
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

rgb_array = np.uint8(dbrgb)
lab_arr = rgb_array.reshape((1,-1,3))

rgb_array = cv2.cvtColor(lab_arr, cv2.COLOR_RGB2Lab)
print(rgb_array)
rgb_array = np.float32(rgb_array.reshape((-1,3)))

print(rgb_array.shape)
print (len(imageFiles))


r,g,b = rgb_array[123,:]
rgb = np.array([r,g,b])
print(rgb.shape)
mainImage = np.zeros((SIZE*tileSize,SIZE*tileSize,3))
errorImage = np.zeros((SIZE,SIZE),dtype=np.float32)

finalImageArray = []
for i in range(mosaicBase.shape[0]):
    for j in range(mosaicBase.shape[1]):
        r,g,b = mosaicBase[j,i,:]
        rgb = np.array([r, g, b])
        diff_array = np.sqrt(np.sum(np.square(rgb_array - rgb) * weight,axis=1))
        min_ele = np.argmin(diff_array)
        errorImage[j,i] = diff_array.min()
        img = cv2.imread(os.path.join(data_dir, imageFiles[min_ele]))
        res = cv2.resize(img, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
        mainImage[j*tileSize:j*tileSize+tileSize,i*tileSize:i*tileSize+tileSize,:] = res

        print("min is= ", min_ele)

datstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
saveFileName = os.path.abspath(os.path.join(data_dir,'..', 'newMosaicImageLab_'+str(SIZE)+'_'+datstr+'_'+str(weight)+'.jpg'))
print(saveFileName)
cv2.imwrite(saveFileName,mainImage)
plt.imshow(np.uint8(mainImage[:,:,::-1]))
plt.show()
print(errorImage.min(), errorImage.max())
scaledError = (errorImage - errorImage.min())/(errorImage.max() - errorImage.min())
plt.imshow(np.uint8(255.*scaledError))
plt.show()
plt.hist(errorImage.flatten())
plt.show()

print(mainImage.shape)
print(mainImage.dtype)
print('saved')
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(.1)
print(s.getvalue())