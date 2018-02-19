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
# play video and targetImage
#   grab frame
#   compare to analyzed image
#   compare frame diff to diff arry to get all better match locations
#   for better match
#       update diff array
#       update image
#
#   every x frames or seconds update targetImage on screen
#on video end, time limit reached, or terimnation save file and exit

weight = np.array([1.0,0.0,0.0])
# targetImageFile = '1 - PA15xHi.jpg'
targetImageFile = '4 - Fyt8Egq.jpg'
targetColorSpace = cv2.COLOR_BGR2RGB
#databaseColorSpace = cv2.COLOR_RGB2YCR_CB
TIME_PER_FRAME = 1
WIN_NAME = 'video'
data_dir2 = r'D:\Alex laptop\source_images'


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(WIN_NAME)

tileSize=32
# load target image
img = cv2.imread(os.path.join(data_dir2, targetImageFile))
#img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)
# figure out even height and width for tile size that best keeps aspect
print("orig img", img.shape)
width = int(img.shape[1]/tileSize)*tileSize
height = int(img.shape[0]/tileSize)*tileSize
img = img[:height, :width, :]
print('img.shape:', img.shape)

targetImage = np.empty((img.shape[0]*img.shape[1]//(tileSize*tileSize),tileSize,tileSize, 3),dtype=img.dtype)
index = 0
chris = -1
for i in range(0,img.shape[0],tileSize):
    for j in range(0, img.shape[1], tileSize):
        #targetImage[index,:,:,:] = cv2.cvtColor(img[i:i+tileSize,j:j+tileSize,:], targetColorSpace)
        targetImage[index, :, :, :] = img[i:i + tileSize, j:j + tileSize, :]
        if i == 6*tileSize and j == 5*tileSize: chris = index
        index += 1
        #cv2.imshow(WIN_NAME, img[i:i+tileSize,j:j+tileSize,:])
        #cv2.waitKey()
        #chris = targetImage[index,:,:,:]
        #cv2.imshow(WIN_NAME, chris.reshape(tileSize, tileSize, -1))
        #cv2.waitKey()
        #exit()

# i = 6*tileSize
# j= 5*tileSize
# cv2.imshow(WIN_NAME, img[i:i+tileSize,j:j+tileSize,:])
# cv2.waitKey()
#
# print(chris)
# cv2.imshow(WIN_NAME, targetImage[chris,:,:,:].reshape(tileSize, tileSize, 3))
# cv2.waitKey()

##TODO:create a mirror array to targetImage to store current diff from needed color
diffArr = np.empty(targetImage.shape[:-1],dtype=np.float64)
diffArr.fill(np.finfo(diffArr.dtype).max/4)

mainImage = np.zeros((height,width,3),dtype=np.uint8)

TIME_PER_FRAME = 1
oldTime = time.time()
counter = 0

def findBetterPoints(currentDiffArr, targetImage, frameColor):
    #frameColor
    #rgb = np.array([r, g, b])
    imgDiff = targetImage
    testDiffArr = np.sqrt(np.sum(np.square(np.float64(targetImage) - np.float64(frameColor), dtype=np.float64), axis=-1))
    relativeDiff = currentDiffArr - testDiffArr
    tileError = np.sum(np.sum(relativeDiff, axis=-1),axis=-1)
    print("tileError, currentDiff testDiff ", tileError.shape, currentDiffArr.shape, testDiffArr.shape)
    #print("srel shape is: ", relativeDiff.shape)
    indices = np.argwhere(tileError>0)
    print("number of new tiles = ",len(indices))
    for index in indices:
        currentDiffArr[index,:,:] = testDiffArr[index,:,:]
    return indices

def changeImage(mainImage, frame, index):
    #print('imageChanged at index', index)
    #print('frame.shape:',frame.shape, 'height:', height, 'widith:', width)
    i, j = np.unravel_index(index, (height//tileSize, width//tileSize))
    i = i[0]
    j = j[0]
    #print('mainImage.shape, i, j, tilesize:', mainImage.shape, i, j, tileSize)
    mainImage[i * tileSize:i * tileSize + tileSize, j * tileSize:j * tileSize + tileSize, :] = frame

def resizeAndAvgImage(frame):
    print('frame is ', frame.shape)
    res = cv2.resize(frame, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
    print('frame is after', res.shape)
    avg = np.mean(frame, axis=0)
    avg = np.mean(avg, axis=0)
    r, g, b = avg
    return res, (r, g, b)

# frame = cap.read()
# frame ,(r,g,b) = resizeAndAvgImage(frame)
# replaceIndices = findBetterPoints(diffArr, targetImage, np.array((r,g,b)))
# for index in replaceIndices:
#   changeImage(mainImage, frame, index)
#

while(counter < 60):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (time.time() - oldTime) > TIME_PER_FRAME:
        oldTime = time.time()
        counter+=1
        print(counter)
        #print('frame.shape=', frame.shape)
        #diffArr += diffArr.max()/60
        cv2.imshow(WIN_NAME, mainImage[::2,::2,:])
        cv2.waitKey(1)
        frame ,(r,g,b) = resizeAndAvgImage(frame)
        replaceIndices = findBetterPoints(diffArr, targetImage, frame)
        for index in replaceIndices:
            changeImage(mainImage, frame, index)
    if( cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


saveFileName = os.path.abspath(os.path.join(os.getcwd(), 'VideoMosaicBuild_'+ str(time.time())+'_'+str(weight)+'.jpg'))
print(saveFileName)
cv2.imwrite(saveFileName,mainImage)
