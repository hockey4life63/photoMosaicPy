import numpy as np
import cv2
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
# targetImageFile = '2 - TyyFBOz.jpg'
targetImageFile = '4 - Fyt8Egq.jpg'
targetColorSpace = cv2.COLOR_BGR2RGB
TIME_PER_FRAME = 0
WIN_NAME = 'video'
data_dir2 = r'D:\Alex laptop\source_images'
RUNTIME = 300

cap = cv2.VideoCapture(0)

cv2.namedWindow(WIN_NAME)

tileSize=64
# load target image
img = cv2.imread(os.path.join(data_dir2, targetImageFile))
# figure out even height and width for tile size that best keeps aspect
width = int(img.shape[1]/tileSize)*tileSize
height = int(img.shape[0]/tileSize)*tileSize
# crop exccess pixels to be even for tileSize
img = img[:height, :width, :]
# reshape image into tileSize chunks
targetImage = np.empty((img.shape[0]*img.shape[1]//(tileSize*tileSize),tileSize,tileSize, 3),dtype=img.dtype)
index = 0
for i in range(0,img.shape[0],tileSize):
    for j in range(0, img.shape[1], tileSize):
        targetImage[index, :, :, :] = img[i:i + tileSize, j:j + tileSize, :]
        index += 1

diffArr = np.empty(targetImage.shape[:-1],dtype=np.float64)
diffArr.fill(np.finfo(diffArr.dtype).max/4)

mainImage = np.zeros((height,width,3),dtype=np.uint8)

oldTime = time.time()
counter = 0

def findBetterPoints(currentDiffArr, targetImage, frameColor):
    imgDiff = targetImage
    testDiffArr = np.sqrt(np.sum(np.square(np.float64(targetImage) - np.float64(frameColor), dtype=np.float64), axis=-1))
    relativeDiff = currentDiffArr - testDiffArr
    tileError = np.sum(np.sum(relativeDiff, axis=-1),axis=-1)
    indices = np.argwhere(tileError>0)
    for index in indices:
        currentDiffArr[index,:,:] = testDiffArr[index,:,:]
    return indices

def changeImage(mainImage, frame, index):
    i, j = np.unravel_index(index, (height//tileSize, width//tileSize))
    i = i[0]
    j = j[0]
    mainImage[i * tileSize:i * tileSize + tileSize, j * tileSize:j * tileSize + tileSize, :] = frame

def resizeAndAvgImage(frame):
    res = cv2.resize(frame, (tileSize, tileSize), interpolation=cv2.INTER_AREA)
    avg = np.mean(frame, axis=0)
    avg = np.mean(avg, axis=0)
    r, g, b = avg
    return res, (r, g, b)
timerTime = time.time()
while(counter < RUNTIME):
    ret, frame = cap.read()
    if (time.time() - oldTime) > TIME_PER_FRAME:
        if(time.time()- timerTime >=1):
            counter+=1
            print(counter)
            timerTime = time.time()
        oldTime = time.time()
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