import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

import numpy as np
import cv2
import dlib

from geo_int import GeoInt

''' the class FaceSmile will
1. load face picture into numpy array;
2. create 68 points to characterize face;
3. create delaunay triangle using 68 points;
4. try to move one out of 68 points to modify face picture;
to be continue
'''
class FaceSmile(object):
    ''' load face picture into numpy 3d array '''
    def __init__(self
                 , pic=None      # original picture file
                 , img=None      # image numpy format data file
                 , model_data='shape_predictor_68_face_landmarks.dat'
                 , pnt68=None    # 68 point coordinate
                 , dt=None       # delaunay triangle based on 68 points
                ):
        self.model_data=model_data
        if pic == None and img == None:
            logger.error(f"No pictures and image data!!!")
            exit(1)
        elif pic == None:
            logger.debug(f"FaceSmile.__init__:load img size")
            with open(img, 'r') as f:
                mystr=f.readline()
            # split by ', ', related to above
            size=mystr.strip().split(', ')
            # load RGB
            # skip two lines, related to above
            logger.debug(f"FaceSmile.__init__:load img from txt file {img}")
            data=np.loadtxt(img, skiprows=2, dtype=int)
            # reshape from 2D to 3D
            logger.debug(f"FaceSmile.__init__:reshape to ({int(size[0])}, {int(size[1])}, 3)")
            self.img=np.reshape(data,(int(size[0]), int(size[1]), 3))
            logger.debug(f"FaceSmile.__init__:finish loading img {img}")
        else:
            self.pic=pic
            logger.debug(f"FaceSmile.__init__:load picture...")
            self.img=cv2.imread(self.pic)
            logger.debug(f"FaceSmile.__init__:finish loading picture")
        self.pnt68=list()
        if pnt68 != None:
            logger.debug(f"FaceSmile.__init__:load 68 points...")
            data=np.loadtxt(pnt68, dtype=int)
            for i in range(data.shape[0]):
                self.pnt68.append((data[i][0], data[i][1]))
            logger.debug(f"FaceSmile.__init__:finish load 68 points {len(self.pnt68)}")
        self.dt=list()
        if dt != None:
            logger.debug(f"FaceSmile.__init__:load delaunay triangle...")
            data=np.loadtxt(dt, dtype=int)
            for i in range(data.shape[0]):
                self.dt.append((data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]))
            logger.debug(f"FaceSmile.__init__:finish load delaunay triangle")
        # morphing point list
        self.mpnt=None
        # morphing delaunay triangle list
        self.mdt=None
        # new point list (according to morphing points)
        self.newpnt=None
        # new delaunay triangle list (according to morphing triangles)
        self.newdt=None
        # global rectangle. No change if out of this rectangle
        self.grect=None
        # rectangle for each new delaunay triangle
        self.rectlist=None
        # except for 68 points, create boundry points
        self.withbound=False
        # for 68 points, we may remove points too close to each other
        self.dropclose=False
        # for large triangle, we may split by adding new points
        self.trisplit=False
        # use flag to indicate each point
        # P: original 68 points
        # B: added boundary point
        # D: dropped points which are too close to other point
        # S: added point to split the large triangle
        self.pntflag=['P' for i in range(len(self.pnt68))]

    ''' create 68 points to characterize face '''
    def face68pnt(self):
        detector = dlib.get_frontal_face_detector()

        logger.debug(f"FaceSmile.face68pnt:load model data...")
        predictor = dlib.shape_predictor(self.model_data)

        logger.debug(f"FaceSmile.face68pnt:capture gray...")
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        logger.debug(f"FaceSmile.face68pnt:detect...")
        rects = detector(img_gray, 1)
        logger.debug(f"FaceSmile.face68pnt:picture length: {len(rects)}")

        rstlist=list()
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(self.img,rects[i]).parts()])
            logger.debug(f"FaceSmile.face68pnt:The {i+1}-th rect, number of landmarks: {len(landmarks)}")
            rstlist.append(landmarks)

        logger.debug(f"FaceSmile.face68pnt:append to pnt68...")
        if len(self.pnt68)>0:
            logger.debug(f"FaceSmile.face68pnt:pnt68 not empty, will be removed...")
            self.pnt68=list()
        for landmark in rstlist:
            for idx, point in enumerate(landmark, start=1):
                self.pnt68.append((int(point[0, 0]), int(point[0, 1])))
                # logger.debug(f"FaceSmile.face68pnt:The {idx}-th point is ({int(point[0, 0])}, {int(point[0, 1])})")
        logger.debug(f"FaceSmile.face68pnt:Finished.")
        return

    ''' create delaunay triangle using 68 points of the face '''
    def face2dt(self, boundary=False
                , pntdrop=False, distratio=100, morphlist=None
                , bigtrianglesplit=False, arearatio=81
        ):
        logger.debug(f"FaceSmile.face2dt:start create delaunay triangle...")
        if len(self.pnt68) == 0:
            logger.error(f"No 68 points, check you submit face68pnt first!!!")
            exit(1)
        gi=GeoInt()
        logger.debug(f"FaceSmile.face2dt:get image size...")
        size=self.img.shape
        rect = (0, 0, size[1], size[0])
        if boundary and not self.withbound:
            logger.debug(f"FaceSmile.face2dt:create rectangle bound...")
            rect68=gi.rect4pnts(pntlst=self.pnt68)
            #logger.debug(rect)
            #logger.debug(rect68)
            rect10pct=gi.rectexpand(rect=rect68, pct=10)
            #logger.debug(rect10pct)
            thisrect=gi.rectintersect(rect1=rect, rect2=rect10pct)
            #logger.debug(rect10pct)
            L, T, R, B=thisrect
            for X in range(10):
                self.pnt68.append((int((R-L)/10*X)+L,T))
                self.pntflag.append('B')
                self.pnt68.append((int((R-L)/10*(X+1))+L-1,B-1))
                self.pntflag.append('B')
                self.pnt68.append((L,int((B-T)/10*(X+1))+T-1))
                self.pntflag.append('B')
                self.pnt68.append((R-1,int((B-T)/10*X)+T))
                self.pntflag.append('B')
            self.withbound=True
        logger.debug(f"FaceSmile.face2dt:Create an instance of Subdiv2D")
        subdiv = cv2.Subdiv2D(rect);
        logger.debug(f"FaceSmile.face2dt:Insert points into subdiv")
        try:
            for p in self.pnt68:
                subdiv.insert(p)
        except:
            logger.error(rect)
            logger.error(p)
            exit(1)
        logger.debug(f"FaceSmile.face2dt:Create delaunay triangle")
        self.dt = subdiv.getTriangleList().astype(int)
        if pntdrop:
            logger.debug(f"FaceSmile.face2dt:Drop points if too close")
            edgeset=set()
            for t in self.dt:
                for e in gi.triangle2edge(t):
                    oe=(e[2], e[3], e[0], e[1]) # another order of edge
                    if e not in edgeset and oe not in edgeset:
                        edgeset.add(e)

            edgelist=list(sorted(edgeset))
            edgeshort=list()
            piclen2=(size[0])*(size[0])+(size[1])*(size[1])
            for idx, e in enumerate(edgelist, start=1):
                if ((e[0]-e[2])*(e[0]-e[2])+(e[1]-e[3])*(e[1]-e[3]))*distratio*distratio<piclen2:
                    edgeshort.append(e) # less than pic size divided by distratio
            logger.debug(f"    Closest two point are {edgeshort}")

            # check if two short edges sharing same point
            droppnt=set()
            for e in edgeshort:
                droppnt.add((e[0], e[1]))
                droppnt.add((e[2], e[3]))
            if len(droppnt) < len(edgeshort)*2:
                logger.error(f"Three points together!!!")
                logger.error(edgeshort)
                logger.error(droppnt)
                exit(1)

            # flag the dropped point in pair from pnt68
            for e in edgeshort:
                pnt=(e[0], e[1])
                if pnt not in self.pnt68:
                    logger.error(f"Point {pnt} not in object!!!")
                    exit(1)
                pntidx=self.pnt68.index(pnt)
                if morphlist != None and pntidx+1 in morphlist:
                    logger.debug(f"WARNING: the closed point is selected as morphing point {pntidx+1}!")
                    pnt=(e[2], e[3])
                    if pnt not in self.pnt68:
                        logger.error(f"The second point {pnt} not in object!!!")
                        exit(1)
                    pntidx=self.pnt68.index(pnt)
                    logger.debug(f"WARNING: the other point {pntidx+1} will be used to drop...")
                    if pntidx+1 in morphlist:
                        logger.error(f"The two ends of edge {e} are selected as morphing points!!!")
                        exit(1)
                logger.debug(f"    Drop {pntidx+1}-th point {(e[0], e[1])} with flag {self.pntflag[pntidx]}")
                self.pntflag[pntidx]='D'
            subdiv = cv2.Subdiv2D(rect)
            for idx, p in enumerate(self.pnt68):
                if self.pntflag[idx] != 'D':
                    subdiv.insert(p)
            self.dt = subdiv.getTriangleList().astype(int)
            
        if bigtrianglesplit:
            logger.debug(f"FaceSmile.face2dt:split big triangle...")
            numofbigt=10 # iteration start
            while numofbigt>0:
                logger.debug(f"    loop with numofbigt={numofbigt}")
                bigtlist=list()
                rectarea=size[0]*size[1]*2 # since area of triangle didn't divied by 2
                for t in self.dt:
                    s=gi.areaoftriangle(p0=(t[0], t[1]), p1=(t[2], t[3]), p2=(t[4], t[5]))
                    if s * arearatio > rectarea:
                        logger.debug(f"    triangle {t} area {s} rect {rectarea}")
                        bigtlist.append(t) # larger than pic area divided by arearatio
                numofbigt=len(bigtlist)
                if numofbigt==0:
                    break
                for t in bigtlist:
                    centert=(int((t[0]+t[2]+t[4])/3),int((t[1]+t[3]+t[5])/3))
                    logger.debug(f"    triangle {t} center {centert}")
                    self.pnt68.append(centert)
                    self.pntflag.append('S')
                subdiv = cv2.Subdiv2D(rect)
                for idx, p in enumerate(self.pnt68):
                    if self.pntflag[idx] != 'D':
                        subdiv.insert(p)
                self.dt = subdiv.getTriangleList().astype(int)
        logger.debug(f"FaceSmile.face2dt:Finish create delaunay triangle")

    ''' preparing point morphing based on delaunay triangles '''
    def faceprep(self
                 , morphlist=None           # morphing point index
                 , newpntlist=None          # according new point coordinate
                 , boundary=False           # if true, create boundary for morphing
                 , pntdrop=False            # if true, delete points too close
                 , distratio=100            # when pntdrop, the distance ratio
                 , bigtrianglesplit=False   # if true, split big triangle
                 , arearatio=81             # when bigtrianglesplit, the area ratio
                    ):
        logger.debug(f"FaceSmile.faceprep:create morphing data structure...")
        if len(self.pnt68) != 68:
            logger.error(f"No 68 points, check you submit face68pnt first!!!")
            exit(1)
        if len(self.dt) == 0:
            logger.error(f"No delaunay triangle, check you submit face2dt first!!!")
            exit(1)
        logger.debug(f"FaceSmile.faceprep:image shape {self.img.shape}...")
        if boundary or pntdrop or bigtrianglesplit:
            logger.debug(f"FaceSmile.faceprep:Re-calculate points and triangle based on option...")
            self.face2dt(boundary=boundary
                         , pntdrop=pntdrop, distratio=distratio, morphlist=morphlist
                         , bigtrianglesplit=bigtrianglesplit, arearatio=arearatio
                        )
        self.mpnt=list()
        for idx, mphidx in enumerate(morphlist, start=1):
            self.mpnt.append((self.pnt68[mphidx-1][0], self.pnt68[mphidx-1][1]))
            logger.debug(f"    The {idx}-th morphing point {mphidx} with coordinate {(self.pnt68[mphidx-1][0], self.pnt68[mphidx-1][1])}")
        logger.debug(f"FaceSmile.faceprep:triangles with vertex at morphpnt...")
        logger.debug(f"FaceSmile.faceprep:compute morphing delaunay triangles...")
        self.mdt=list()
        for t in self.dt:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            for p in self.mpnt:
                if p == pt1 or p == pt2 or p == pt3:
                    logger.info(f"    Morphing triangle point coordinates {t}")
                    self.mdt.append(t)
                    break
        logger.debug(f"FaceSmile.faceprep:sign new point according to morphing point...")
        self.newpnt=newpntlist
        logger.debug(f"FaceSmile.faceprep:compute new delaunay triangles...")
        self.newdt=list()
        for t in self.mdt:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            for p, q in zip(self.mpnt, self.newpnt):
                if p == pt1:
                    pt1=q
                if p == pt2:
                    pt2=q
                if p == pt3:
                    pt3=q
            self.newdt.append(tuple(list(pt1)+list(pt2)+list(pt3)))
        logger.debug(f"FaceSmile.faceprep:compute global rectangle...")
        gi=GeoInt()
        self.grect=gi.rect4trilist(tl=self.newdt)
        logger.debug(f"FaceSmile.faceprep:compute rectangle list of new triangles...")
        self.rectlist=list()
        for t in self.newdt:
            self.rectlist.append(gi.rect4triangle(t=t))
        logger.debug(f"FaceSmile.faceprep:Finished")
        return

    ''' one point morphing execution '''
    def facemorphing(self, outpic=None):
        logger.debug(f"FaceSmile.facemorphing:image copy...")
        newimg = self.img.copy()
        logger.debug(f"FaceSmile.facemorphing:start morphing...")
        size=newimg.shape
        gi=GeoInt()
        for x in range(size[0]):
            if x<self.grect[1]:
                continue
            if x>self.grect[3]:
                continue
            for y in range(size[1]):
                if y<self.grect[0]:
                    continue
                if y>self.grect[2]:
                    continue
                outrect=True
                for idx, r in enumerate(self.rectlist, start=1):
                    # if idx not in(6,5):continue #debug
                    if x>r[1] and x<r[3] and y>r[0] and y<r[2]:
                        outrect=False
                        break
                if outrect: continue
                for idx, t in enumerate(self.newdt, start=1):
                    # if idx not in(6,5):continue #debug
                    if gi.pnt_in_triangle(pnt=(y,x), t=t):
                        pnt4=gi.origpntlist(newpnt=(y,x), newt=t, orit=self.mdt[idx-1])
                        newimg[x][y][0]=int(self.img[pnt4[0][1]][pnt4[0][0]][0])
                        newimg[x][y][1]=int(self.img[pnt4[0][1]][pnt4[0][0]][1])
                        newimg[x][y][2]=int(self.img[pnt4[0][1]][pnt4[0][0]][2])
                        # newimg[x][y][0]=int((self.img[pnt4[0][0]][pnt4[0][1]][0]
                        #                  + self.img[pnt4[1][0]][pnt4[1][1]][0]
                        #                  + self.img[pnt4[2][0]][pnt4[2][1]][0]
                        #                  + self.img[pnt4[3][0]][pnt4[3][1]][0])/4)
                        # newimg[x][y][1]=int((self.img[pnt4[0][0]][pnt4[0][1]][1]
                        #                  + self.img[pnt4[1][0]][pnt4[1][1]][1]
                        #                  + self.img[pnt4[2][0]][pnt4[2][1]][1]
                        #                  + self.img[pnt4[3][0]][pnt4[3][1]][1])/4)
                        # newimg[x][y][2]=int((self.img[pnt4[0][0]][pnt4[0][1]][2]
                        #                  + self.img[pnt4[1][0]][pnt4[1][1]][2]
                        #                  + self.img[pnt4[2][0]][pnt4[2][1]][2]
                        #                  + self.img[pnt4[3][0]][pnt4[3][1]][2])/4)
                        break
        logger.debug(f"FaceSmile.facemorphing:finish morphing.")
        cv2.imwrite(outpic, newimg)
        logger.debug(f"FaceSmile.facemorphing:finish output to {outpic}")

    ''' output picture '''
    def img2txt(self, outtxtimg=None):
        logger.debug(f"FaceSmile.img2txt:output to {outtxtimg}...")
        size = self.img.shape
        with open(outtxtimg, 'w') as f:
            # separated by ', ', use two lines, related to __init__
            f.write(f"{size[0]}, {size[1]}\n\n")
            for x in range(size[0]):
                for y in range(size[1]):
                    f.write(f"{self.img[x][y][0]} {self.img[x][y][1]} {self.img[x][y][2]}\n")
        logger.debug(f"FaceSmile.img2txt:finish output to {outtxtimg}")

    ''' output 68 points '''
    def img68totxt(self, outtxt68=None):
        logger.debug(f"FaceSmile.img68totxt:output to {outtxt68}...")
        if len(self.pnt68) == 0:
            logger.error(f"No 68 points, check you submit face68pnt first!!! {len(self.pnt68)}")
            exit(1)
        with open(outtxt68, 'w') as f:
            if self.pntnum != 0 and self.pntnum != 68:
                f.write(f"{self.pntnum} # number of points after remove the following:\n")
                f.write(f"{self.pntdelsorted}\n")
            for idx, pnt in enumerate(self.pnt68, start=1):
                f.write(f"{pnt[0]} {pnt[1]}\n")
        logger.debug(f"FaceSmile.img68totxt:finish output to {outtxt68}.")

    ''' output delaunay triangle '''
    def imgdt2txt(self, outtxtdt=None):
        logger.debug(f"FaceSmile.imgdt2txt:output to {outtxtdt}...")
        if len(self.dt) == 0:
            logger.error(f"No delaunay triangle, check you submit face2dt first!!!")
            exit(1)
        with open(outtxtdt, 'w') as f:
            for idx, pnt in enumerate(self.dt, start=1):
                f.write(f"{int(pnt[0])} {int(pnt[1])} {int(pnt[2])} {int(pnt[3])} {int(pnt[4])} {int(pnt[5])}\n")
        logger.debug(f"FaceSmile.imgdt2txt:finish output to {outtxtdt}.")

    ''' output delaunay triangle '''
    def img1morphdbg(self, outtxtdbg=None):
        logger.debug(f"FaceSmile.img1morphdbg:output to {outtxtdbg}...")
        with open(outtxtdbg, 'w') as f:
            f.write(f"Morphing point : {self.mpnt}\n")
            f.write(f"\nNew point : {self.newpnt}\n")
            f.write(f"\nMorphing triangles :\n")
            for idx, pnt in enumerate(self.mdt, start=1):
                f.write(f"{pnt}\n")
            f.write(f"\nNew triangles :\n")
            for idx, pnt in enumerate(self.newdt, start=1):
                f.write(f"{pnt}\n")
            f.write(f"\nGlobal rectangle : {self.grect}\n")
            f.write(f"\nRectangles :\n")
            for idx, pnt in enumerate(self.rectlist, start=1):
                f.write(f"{pnt}\n")
        logger.debug(f"FaceSmile.imgdt2txt:finish output to {outtxtdbg}.")

    ''' draw 68 points with number '''
    def img68pnt2jpg(self, outpic=None
                     , pntcolor=None, font=None, fontsize=None, fontcolor=None
                     , bpntcolor=None, bfont=None, bfontsize=None, bfontcolor=None
                     , spntcolor=None, sfont=None, sfontsize=None, sfontcolor=None
        ):
        logger.debug(f"FaceSmile.img68pnt2jpg:output to {outpic}...")
        if len(self.pnt68) == 0:
            logger.error(f"No 68 points, check you submit face68pnt first!!!")
            exit(1)
        logger.debug(f"FaceSmile.img68pnt2jpg:make copy of picture...")
        newimg=self.img.copy()
        logger.debug(f"FaceSmile.img68pnt2jpg:draw 68 points ...")
        for idx, p in enumerate(self.pnt68, start=1):
            if self.pntflag[idx-1] == 'D':
                continue
            if self.pntflag[idx-1] == 'P':
                if pntcolor == None:
                    pntcolor=(0, 255, 0) # green
                if font == None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                if fontsize == None:
                    fontsize = 0.8
                if fontcolor == None:
                    fontcolor=(0, 0, 255) # red
            elif self.pntflag[idx-1] == 'B':
                if bpntcolor == None:
                    pntcolor=(0, 255, 0) # green
                else:
                    pntcolor=bpntcolor
                if bfont == None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                else:
                    font=bfont
                if bfontsize == None:
                    fontsize = 0.8
                else:
                    fontsize=bfontsize
                if bfontcolor == None:
                    fontcolor=(0, 0, 255) # red
                else:
                    fontcolor=bfontcolor
            elif self.pntflag[idx-1] == 'S':
                if spntcolor == None:
                    pntcolor=(0, 128, 0) # green
                else:
                    pntcolor=spntcolor
                if sfont == None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                else:
                    font=sfont
                if sfontsize == None:
                    fontsize = 0.4
                else:
                    fontsize=sfontsize
                if sfontcolor == None:
                    fontcolor=(0, 0, 128) # red
                else:
                    fontcolor=sfontcolor
            cv2.circle( newimg, p, 2, pntcolor, cv2.FILLED, cv2.LINE_AA, 0 )
            cv2.putText(newimg, str(idx), p, font, fontsize, fontcolor, 1,cv2.LINE_AA)
        logger.debug(f"FaceSmile.img68pnt2jpg:output to jpg file...")
        cv2.imwrite(outpic, newimg)
        logger.debug(f"FaceSmile.img68pnt2jpg:finish output to {outpic}.")

    ''' draw delaunay triangle on picture '''
    def imgdt2jpg(self, outpic=None, delaunay_color=None
                  , morph_color=None
                  , new_color=None
                 ):
        logger.debug(f"FaceSmile.imgdt2jpg:output to {outpic}...")
        if len(self.pnt68) == 0:
            logger.error(f"No 68 points, check you submit face68pnt first!!!")
            exit(1)
        if len(self.dt) == 0:
            logger.error(f"No delaunay triangle, check you submit face2dt first!!!")
            exit(1)
        logger.debug(f"FaceSmile.imgdt2jpg:make copy of picture...")
        newimg=self.img.copy()
        if new_color == None:
            logger.debug(f"FaceSmile.imgdt2jpg:draw delaunay triangle...")
            for t in self.dt:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                cv2.line(newimg, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(newimg, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(newimg, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
            logger.debug(f"FaceSmile.imgdt2jpg:output to jpg file...")
            cv2.imwrite(outpic, newimg)
            logger.debug(f"FaceSmile.imgdt2jpg:finish output to {outpic}.")
            return
        if len(self.mdt) == 0:
            logger.error(f"No morphing delaunay triangle, check you submit face1pnt first!!!")
            exit(1)
        logger.debug(f"FaceSmile.imgdt2jpg:draw other triangles not related to morphing point...")
        for t in self.dt:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(newimg, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(newimg, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(newimg, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
        logger.debug(f"FaceSmile.imgdt2jpg:draw triangles with vertex at morphpnt...")
        for t in self.mdt:
            # gi=GeoInt()
            # if gi.pnt_in_triangle(pnt=self.newpnt,t=t):
            #     dbg_color=(0,0,0)
            #     thick=5
            # else:
            #     dbg_color=morph_color
            #     thick=1
            dbg_color=morph_color
            thick=1
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(newimg, pt1, pt2, dbg_color, thick, cv2.LINE_AA, 0)
            cv2.line(newimg, pt2, pt3, dbg_color, thick, cv2.LINE_AA, 0)
            cv2.line(newimg, pt3, pt1, dbg_color, thick, cv2.LINE_AA, 0)
        logger.debug(f"FaceSmile.imgdt2jpg:draw new triangles with vertex at newpnt...")
        for t in self.newdt:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(newimg, pt1, pt2, new_color, 1, cv2.LINE_AA, 0)
            cv2.line(newimg, pt2, pt3, new_color, 1, cv2.LINE_AA, 0)
            cv2.line(newimg, pt3, pt1, new_color, 1, cv2.LINE_AA, 0)
        logger.debug(f"FaceSmile.imgdt2jpg:output to jpg file...")
        cv2.imwrite(outpic, newimg)
        logger.debug(f"FaceSmile.imgdt2jpg:finish output to {outpic}.")

if __name__ == '__main__':
    mylog = os.path.realpath(__file__).replace('.py', '.log').replace('python', 'log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)

    logger.info(f'start python code {__file__}.\n')
    jpgpath=os.path.dirname(os.path.realpath(__file__)).replace('python', 'jpg')
    modelpath=os.path.dirname(os.path.realpath(__file__)).replace('python', 'modeldata')
    txtpath=os.path.dirname(os.path.realpath(__file__)).replace('python', 'txt')
    outpath=os.path.dirname(os.path.realpath(__file__)).replace('python', 'out')

    '''
    # initialization code, just run once
    fs=FaceSmile(pic=f'{jpgpath}/sue.jpg'
                 , model_data=f'{modelpath}/shape_predictor_68_face_landmarks.dat')
    fs.face68pnt()
    fs.face2dt()
    fs.img68totxt(outtxt68=f'{txtpath}/sue68.txt')
    fs.imgdt2txt(outtxtdt=f'{txtpath}/sue_dt.txt')
    '''
    # Assume you have finished initialization, just load existing model
    fs=FaceSmile(pic=f'{jpgpath}/sue.jpg'
                 , pnt68=f'{txtpath}/sue68.txt'
                 , dt=f'{txtpath}/sue_dt.txt')

    fs.img68pnt2jpg(outpic=f'{outpath}/sue68pnt.jpg'
                    , pntcolor=(255, 255, 255) # white
                    , font=None # default
                    , fontsize=0.4
                    , fontcolor=(255, 0, 0)
    )
    fs.imgdt2jpg(outpic=f'{outpath}/sue_dt.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
    #             , new_color=(255,0,0)
    )
    # if you need boundary to move points
    fs.face2dt(boundary=True)
    fs.img68pnt2jpg(outpic=f'{outpath}/suebdy.jpg'
                    , pntcolor=(255, 255, 255) # white
                    , font=None # default
                    , fontsize=0.4
                    , fontcolor=(255, 0, 0)
    )
    fs.imgdt2jpg(outpic=f'{outpath}/sue_dtb.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
    #             , new_color=(255,0,0)
    )
    '''
    fs.faceprep(morphlist=[49, 50, 60, 61, 54, 55, 56, 65]
                , newpntlist=[(170,660), (200, 650), (204, 700), (193,670)
                              , (330, 642), (385, 652), (338,688), (364, 659)]
                , boundary=False
                , pntdrop=False
                , bigtrianglesplit=False
               )
    fs.imgdt2jpg(outpic=f'{outpath}/sue_plan.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
                 , new_color=(255,0,0)
    )
    
    # if you like boundary points to generate delaunay triangles:
    # fs.face2dt(boundary=True, pntdrop=True)
    # fs.img68totxt(outtxt68=f'{txtpath}/sue681.txt')
    # fs.imgdt2txt(outtxtdt=f'{txtpath}/sue_dt1.txt')
    # fs.imgdt2jpg(outpic=f'{outpath}/sue_dt1.jpg', delaunay_color=(255,255,255)
    #              , morph_color=(0,255,255)
    # #             , new_color=(255,0,0)
    # )

    
    # if you like split big delaunay triangles:
    fs.faceprep(morphlist=[49, 50, 60, 61, 54, 55, 56, 65]
                  , newpntlist=[(170,660), (200, 650), (204, 700), (193,670)
                                , (330, 642), (385, 652), (338,688), (364, 659)]
                  , boundary=True
                  , pntdrop=True)
    fs.imgdt2jpg(outpic=f'{outpath}/sue_plan.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
                 , new_color=(255,0,0)
    )
    fs.facemorphing(outpic=f'{outpath}/sue_smile.jpg')
    
    # initialization code, just run once
    fs=FaceSmile(pic=f'{jpgpath}/non.jpg'
                 , model_data=f'{modelpath}/shape_predictor_68_face_landmarks.dat')
    fs.face68pnt()
    fs.img68totxt(outtxt68=f'{txtpath}/non68.txt')
    fs.face2dt()
    fs.imgdt2txt(outtxtdt=f'{txtpath}/non_dt.txt')
    
    fs=FaceSmile(pic=f'{jpgpath}/non.jpg'
                 , pnt68=f'{txtpath}/non68.txt'
                 , dt=f'{txtpath}/non_dt.txt')
    fs.face2dt(boundary=True)
    fs.img68totxt(outtxt68=f'{txtpath}/non68b.txt')
    fs.imgdt2txt(outtxtdt=f'{txtpath}/non_dtb.txt')
    fs.imgdt2jpg(outpic=f'{outpath}/non_dt.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
    #             , new_color=(255,0,0)
    )
    
    fs.faceprep(morphlist=[49, 50, 60, 61, 54, 55, 56, 65]
                  , newpntlist=[(170,660), (200, 650), (204, 700), (193,670)
                                , (330, 642), (385, 652), (338,688), (364, 659)])
    fs.imgdt2jpg(outpic=f'{outpath}/non_plan.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
                 , new_color=(255,0,0)
    )
    fs.facemorphing(outpic=f'{outpath}/non_smile.jpg')
    '''
