import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

import numpy as np
import cv2

def jpg2txt(pic=None, txt=None):
    img = cv2.imread(pic)
    size = img.shape
    with open(txt, 'w') as f:
        # separated by ', ', use two lines, related to below
        f.write(f"{size[0]}, {size[1]}\n\n")
        for x in range(size[0]):
            for y in range(size[1]):
                f.write(f"{img[x][y][0]} {img[x][y][1]} {img[x][y][2]}\n")
    return    
def txt2jpg(txt=None, testflg=False, pic=None):
    # get dimention of RGB
    logger.info("load img size")
    with open(txt, 'r') as f:
        mystr=f.readline()
    # split by ', ', related to above
    size=mystr.strip().split(', ')
    # load RGB
    # skip two lines, related to above
    logger.info("load img from txt file")
    data=np.loadtxt(txt, skiprows=2)
    # reshape from 2D to 3D
    logger.info("reshape")
    img=np.reshape(data,(int(size[0]), int(size[1]), 3))
    if testflg:
        cv2.imwrite(pic,img)
    return

def txt2jpgreverse(txt=None, testflg=False, pic=None):
    # get dimention of RGB
    logger.info("load img size")
    with open(txt, 'r') as f:
        mystr=f.readline()
    # split by ', ', related to above
    size=mystr.strip().split(', ')
    # load RGB
    # skip two lines, related to above
    logger.info("load img from txt file")
    data=np.loadtxt(txt, skiprows=2)
    # reshape from 2D to 3D
    logger.info("reshape")
    img=np.reshape(data,(int(size[0]), int(size[1]), 3))
    logger.info("make copy")
    newimg=img.copy()
    # reverse it
    logger.info("start reverse")
    for x in range(int(size[0])):
        for y in range(int(size[1])):
            for z in range(3):
                newimg[x][y][z]=255-img[x][y][z]
    logger.info("finish reverse")
    if testflg:
        cv2.imwrite(pic,newimg)
    return

def txt2jpgredrect(txt=None, testflg=False, pic=None, poslist=None):
    # get dimention of RGB
    logger.info("load img size")
    with open(txt, 'r') as f:
        mystr=f.readline()
    # split by ', ', related to above
    size=mystr.strip().split(', ')
    # load RGB
    # skip two lines, related to above
    logger.info("load img from txt file")
    data=np.loadtxt(txt, skiprows=2)
    # reshape from 2D to 3D
    logger.info("reshape")
    img=np.reshape(data,(int(size[0]), int(size[1]), 3))
    logger.info("make copy")
    newimg=img.copy()
    # reverse it
    logger.info("start reverse")
    for x in range(int(size[0])):
        for y in range(int(size[1])):
            if x>poslist[0] and x<poslist[1] and y>poslist[2] and y<poslist[3]:
                newimg[x][y][0]=0    # blue
                newimg[x][y][1]=0
                newimg[x][y][2]=255  # red
    logger.info("finish reverse")
    if testflg:
        cv2.imwrite(pic,newimg)
    return

if __name__ == '__main__':
    mylog = os.path.realpath(__file__).replace('.py', '.log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)

    logger.info(f'start python code {__file__}.\n')
    #logger.info("Convert picture to text file")
    #jpg2txt(pic='sue.jpg', txt='t.txt')
    # logger.info("Import text file, reverse and save to picture")
    # txt2jpgreverse(txt='t.txt', testflg=True, pic='t.jpg')
    logger.info("Import text file, red block and save to picture")
    pl=[440,490,400,450]
    txt2jpgredrect(txt='t.txt', testflg=True, pic='tr.jpg', poslist=pl)
    logger.info("Done")
