#!c:/user/ldu/apps/Python36/python.exe -B

import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

import numpy as np
import cv2
import dlib
'''

    The mouth can be accessed through points [48, 68].
    The right eyebrow through points [17, 22].
    The left eyebrow through points [22, 27].
    The right eye using [36, 42].
    The left eye with [42, 48].
    The nose using [27, 36].
    And the jaw via [0, 17].
'''
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
def jpg2pnt68(pic=None):
    logger.info(f"start program jpg2pnt68...")
    detector = dlib.get_frontal_face_detector()
    
    logger.info(f"load data...")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    logger.info(f"load picture...")
    img = cv2.imread(pic)

    logger.info(f"capture gray...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    logger.info(f"detect...")
    rects = detector(img_gray, 1)
    logger.info(f"picture length: {len(rects)}")

    rstlist=list()
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        logger.info(f"The {i+1}-th rect, number of landmarks: {len(landmarks)}")
        rstlist.append(landmarks)

    logger.info(f"Finished program jpg2pnt68")
    return rstlist

def outtxt68(pic=None, pnts=None, out68=None, outlt=None):
    logger.info(f"start program outtxt68...")
    # Create an array of points.
    points = [];
    with open(out68, 'w') as f:
        for landmark in pnts:
            for idx, point in enumerate(landmark):
                f.write(f"({point[0, 0]}, {point[0, 1]}) # {idx+1}-th point\n")
                points.append((int(point[0, 0]), int(point[0, 1])))
    logger.info(f"Finished 68 point coordinates")
    # Rectangle to be used with Subdiv2D
    img = cv2.imread(pic)
    size = img.shape
    rect = (0, 0, size[1], size[0])
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList();
    with open(outlt, 'w') as f:
        for t in triangleList :
            logger.info(t)
            f.write(f"({t[0]}, {t[1]}), ({t[2]}, {t[3]}), ({t[4]}, {t[5]})\n")
    logger.info(f"Finished program outtxt68")
    return    

def showjpg68(pic=None, pnts=None, outpic=None, showflg=True):
    logger.info(f"start program showjpg68...")
    img = cv2.imread(pic)

    logger.info(f"add circles and texts to picture...")
    for landmark in pnts:
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(img, pos, 5, color=(0, 255, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)

    if showflg:
        logger.info(f"show edited picture and wait for key...")
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    if outpic != None:
        cv2.imwrite(outpic, img)
    logger.info(f"Finished program showjpg68")
    return

def showonly68(pic=None, pnts=None, outpic=None, showflg=True):
    logger.info(f"start program showonly68...")
    img = cv2.imread(pic)
    height, width, channels = img.shape
    logger.info(f"Picture '{pic}' with height {height} and width {width}.")
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image[:,:] = (255,255,255) # white background
    logger.info(f"add circles and texts to blank picture...")
    for landmark in pnts:
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(blank_image, pos, 5, color=(0, 0, 0), thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blank_image, str(idx+1), pos, font, 0.5, (0, 0, 255), 1,cv2.LINE_AA)

    if showflg:
        logger.info(f"show edited picture and wait for key...")
        cv2.namedWindow("img", 2)
        cv2.imshow("img", blank_image)
        cv2.waitKey(0)

    if outpic != None:
        cv2.imwrite(outpic, blank_image)
    logger.info(f"Finished program showonly68")
    return

def show68dt(pic=None, pnts=None, outpic=None, showflg=True):
    logger.info(f"start program showjpg68...")
    img = cv2.imread(pic)

    logger.info(f"add delaunay triangle to picture...")
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    # Create an array of points.
    points = [];
    for landmark in pnts:
        for idx, point in enumerate(landmark):
            points.append((int(point[0, 0]), int(point[0, 1])))
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(r);
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList();
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)

    if showflg:
        logger.info(f"show edited picture and wait for key...")
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    if outpic != None:
        cv2.imwrite(outpic, img)
    logger.info(f"Finished program show68dt")
    return

def jpg2bit(pic=None, outbit=None):
    logger.info(f"start program jpg2bit...")
    # Rectangle to be used with Subdiv2D
    img = cv2.imread(pic)
    size = img.shape
    with open(outbit, 'w') as f:
        for x in range(size[0]):
            for y in range(size[1]):
                f.write(f"{x} {y} {img[x][y][0]} {img[x][y][1]} {img[x][y][2]}\n")
    logger.info(f"Finished program jpg2bit")
    return    

def main(pic=None, showflg=True):
    filename, file_extension = os.path.splitext(os.path.basename(pic))
    jpg2bit(pic=pic, outbit=f'{filename}.txt')
    rst=jpg2pnt68(pic=pic)
    outtxt68(pic=pic, pnts=rst, out68=f'{filename}68.txt', outlt=f'{filename}_dt.txt')
    # showjpg68(pic=pic, pnts=rst, outpic=f"{filename}_edited.jpg", showflg=showflg)
    show68dt(pic=pic, pnts=rst, outpic=f"{filename}_dt.jpg", showflg=showflg)
    #showonly68(pic=pic, pnts=rst, outpic=f"{filename}_empty.jpg", showflg=showflg)

if __name__ == '__main__':
    mylog = os.path.realpath(__file__).replace('.py', '.log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)

    logger.info(f'start python code {__file__}.\n')
    # main(pic='sue.jpg', showflg=False)
    # main(pic='kevin.jpg', showflg=False)
    # main(pic='david.jpg', showflg=False)
    # main(pic='david1.jpg')
    main(pic='lukesmile.jpg', showflg=False)
