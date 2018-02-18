import os

from PIL import Image
from pymongo import MongoClient
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)
# pool.map(myFunc, myArr) )

client = MongoClient()
ImageDb = client['photoMosaicTest']

fileNames = []
for root, dirs, files in os.walk(r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images'):
    for file in files:
        fileNames.append(file)

print(fileNames.__len__())

os.chdir(r'C:\Users\Alex laptop\Work Projects\personal\photo-mosaic\source_images')
def avgImage(file):
    'file=fileName gets average color and adds to db'
    try:
        testImage = Image.open(file)
        thumbnail = testImage.resize((128,128),Image.ANTIALIAS)
        thumbnail.save('../thumbnails/'+file)
        avg = testImage.resize((1,1), Image.ANTIALIAS)
        px = avg.load()
        rgb = px[0,0]
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        dbDoc = {'r':r,
                 'g':g,
                 'b':b,
                 'filename':file}
        ImageDb.Images.insert_one(dbDoc)
    except:
        print('error')


pool.map(avgImage, fileNames)

