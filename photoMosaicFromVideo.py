import numpy as np
import threading
import cv2
import matplotlib as plt
import time
import os
# steps
# load target image
# flatten image
# build mirrored array for tile diff value with max int value to ensure its over written on first frame
# build the mosaic base image
# play video and mosaicBase
#   grab frame
#   compare to analyzed image
#   compare frame diff to diff arry to get all better match locations
#   for better match
#       update diff array
#       update image
#
#   every x frames or seconds update mosaicBase on screen
#on video end, time limit reached, or terimnation save file and exit

tileSize = 64
weight = np.array([1.0,0.0,0.0])
# targetImageFile = '1 - PA15xHi.jpg'
targetImageFile = '4 - Fyt8Egq.jpg'
targetColorSpace = cv2.COLOR_BGR2YCR_CB
databaseColorSpace = cv2.COLOR_RGB2YCR_CB
TIME_PER_FRAME = 1
WIN_NAME = 'video'
data_dir2 = r'D:\Alex laptop\source_images'


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(WIN_NAME)

# load target image
img = cv2.imread(os.path.join(data_dir2, targetImageFile))
# figure out even height and width for tile size that best keeps aspect
width = int(img.shape[1]/tileSize)
height = int(img.shape[0]/tileSize)
print('img.shape:', img.shape)
img = img[:height*tileSize, :width*tileSize, :]

# resize image thrun cv2 an change color space
mosaicBase = cv2.resize( img,(height, width), interpolation = cv2.INTER_AREA)
mosaicBase = cv2.cvtColor(mosaicBase, targetColorSpace)

# flatten to be single array of rgb arrays
mosaicBase = mosaicBase.reshape(-1,3)
##TODO:create a mirror array to mosaicBase to store current diff from needed color
diffArr = np.empty(mosaicBase.shape[0],dtype=np.float64)
diffArr.fill(np.finfo(diffArr.dtype).max/4)

mainImage = np.zeros((height*tileSize,width*tileSize,3),dtype=np.uint8)

TIME_PER_FRAME = 1
oldTime = time.time()
counter = 0

def findBetterPoints(currentDiffArr, mosaicBase, frameColor):
    rgb = frameColor
    #rgb = np.array([r, g, b])
    testDiffArr = np.sqrt(np.sum(np.square(mosaicBase - rgb) * weight, axis=1))
    relativeDiff = currentDiffArr - testDiffArr
    #print("srel shape is: ", relativeDiff.shape)
    indices = np.argwhere(relativeDiff>0)
    for index in indices:
        currentDiffArr[index] = testDiffArr[index]
    return indices

def changeImage(mainImage, frame, index):
    #print('imageChanged at index', index)
    #print('frame.shape:',frame.shape, 'height:', height, 'widith:', width)
    i, j = np.unravel_index(index, (height, width))
    i = i[0]
    j = j[0]
    #print('mainImage.shape, i, j, tilesize:', mainImage.shape, i, j, tileSize)
    mainImage[i * tileSize:i * tileSize + tileSize, j * tileSize:j * tileSize + tileSize, :] = frame

def resizeAndAvgImage(frame):
    res = cv2.resize(frame, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
    avg = np.mean(frame, axis=0)
    avg = np.mean(avg, axis=0)
    r, g, b = avg
    return res, (r, g, b)

# frame = cap.read()
# frame ,(r,g,b) = resizeAndAvgImage(frame)
# replaceIndices = findBetterPoints(diffArr, mosaicBase, np.array((r,g,b)))
# for index in replaceIndices:
#   changeImage(mainImage, frame, index)
#

while(counter < 6):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (time.time() - oldTime) > TIME_PER_FRAME:
        oldTime = time.time()
        counter+=1
        print(counter)
        #print('frame.shape=', frame.shape)
        #diffArr += diffArr.max()/60
        cv2.imshow(WIN_NAME, mainImage)
        cv2.waitKey(1)
        frame ,(r,g,b) = resizeAndAvgImage(frame)
        replaceIndices = findBetterPoints(diffArr, mosaicBase, np.array((r,g,b)))
        for index in replaceIndices:
            changeImage(mainImage, frame, index)
    if( cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


saveFileName = os.path.abspath(os.path.join(os.getcwd(), 'VideoMosaicBuild_'+ str(time.time())+'_'+str(weight)+'.jpg'))
print(saveFileName)
cv2.imwrite(saveFileName,mainImage)
