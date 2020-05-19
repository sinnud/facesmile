import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

import math
import random

''' the class GeoInt will
1. create rectangle using multiple points
2. create rectangle using triangle
3. create rectangle using multiple triangles
4. check if one point (in rectangle of the triangle) is inside of the triangle
5. compute original coordinate of point in the transformed triangle
6. propotionally expand (shrink with pct negative) rectangle
7. two rectangle intersection is another rectangle
8. collect triangle's edge as set
'''
class GeoInt():
    ''' smallest rectangle with edge parallel to axis
        includes all integer coordinate points '''
    def rect4pnts(self, pntlst=None):
        xset=set()
        yset=set()
        for pnt in pntlst:
            xset.add(pnt[0])
            yset.add(pnt[1])
        return (min(xset), min(yset), max(xset), max(yset))

    ''' smallest rectangle with edge parallel to axis
        includes triangle with 6 coordinates '''
    def rect4triangle(self, t=None):
        pntlst=list()
        pntlst.append((t[0],t[1]))
        pntlst.append((t[2],t[3]))
        pntlst.append((t[4],t[5]))
        return self.rect4pnts(pntlst)

    ''' smallest rectangle with edge parallel to axis
        includes triangle lists '''
    def rect4trilist(self, tl=None):
        xset=set()
        yset=set()
        for t in tl:
            xset.add(t[0])
            yset.add(t[1])
            xset.add(t[2])
            yset.add(t[3])
            xset.add(t[4])
            yset.add(t[5])
        return (min(xset), min(yset), max(xset), max(yset))

    ''' area of rectangle
        https://www.mathopenref.com/coordtrianglearea.html
    '''
    def areaoftrianglefloat(self, p0=None, p1=None, p2=None):
        return abs(  p0[0]*(p1[1]-p2[1])/2
                    +p1[0]*(p2[1]-p0[1])/2
                    +p2[0]*(p0[1]-p1[1])/2
                  )
    def areaoftriangle(self, p0=None, p1=None, p2=None):
        return abs(  p0[0]*(p1[1]-p2[1])
                    +p1[0]*(p2[1]-p0[1])
                    +p2[0]*(p0[1]-p1[1])
                  )

    ''' area coordinate
        http://mae.uta.edu/~lawrence/me5310/course_materials/me5310_notes/7_Triangular_Elements/7-4_Area_Coordinates/7-4_Area_Coordinates.htm
    '''
    def areacoordinate(self, pnt=None, t=None):
        S =self.areaoftriangle(p0=(t[0], t[1]), p1=(t[2], t[3]), p2=(t[4], t[5]))
        SA=self.areaoftriangle(p0=pnt         , p1=(t[2], t[3]), p2=(t[4], t[5]))
        SB=self.areaoftriangle(p0=(t[0], t[1]), p1=pnt         , p2=(t[4], t[5]))
        SC=self.areaoftriangle(p0=(t[0], t[1]), p1=(t[2], t[3]), p2=pnt         )
        return (S, SA, SB, SC)

    ''' if the point in smallest rectangle of the triangle
        is inside of the triangle.
    Compute area of triangle as S
    compute area of the point connected to each edge, as S1, S2, S3
    when S=S1+S2+S3
    '''
    def pnt_in_triangle(self, pnt=None, t=None):
        S =self.areacoordinate(pnt=pnt, t=t)
        # if abs(S[0]-S[1]-S[2]-S[3])<0.0000005:
        # since we compute double area of triangle with int coordinate, it is always integer
        if S[0]==S[1]+S[2]+S[3]:
            return True
        return False
        
    ''' transformation based on area coordinate
        The point in transformed triangle (new) is from
    the original point (ori) in original triangle.
    They have same area coordinate.
    The formula from area coordinate to cartesian coordinate:
        x*A=x1*A1+x2*A2+x3*A3
        y*A=y1*A1+y2*A2+y3*A3
    '''
    def origpnt_area(self, newpnt=None, newt=None, orit=None):
        S =self.areacoordinate(pnt=newpnt, t=newt)
        return ( (orit[0]*S[1]+orit[2]*S[2]+orit[4]*S[3]) / S[0]
               , (orit[1]*S[1]+orit[3]*S[2]+orit[5]*S[3]) / S[0] )
 
    ''' Since image have all coordinate integer,
    if the original point is float, we need to get somehow avarage
    for RGB value based on coordinate
    '''
    def origpntlist(self, newpnt=None, newt=None, orit=None):
        floatpnt=self.origpnt_area(newpnt=newpnt, newt=newt, orit=orit)
        # mystr=f"\nNew point: {newpnt}\n"
        # mystr=f"{mystr}New triangle: {newt}\n"
        # mystr=f"{mystr}Origin triangle: {orit}\n"
        # mystr=f"{mystr}Origin point: {floatpnt}\n\n"
        # logger.info(mystr)
        return ( (math.floor(floatpnt[0]), math.floor(floatpnt[1]))
                 , (math.floor(floatpnt[0]), math.ceil(floatpnt[1]))
                 , (math.ceil(floatpnt[0]), math.floor(floatpnt[1]))
                 , (math.ceil(floatpnt[0]), math.ceil(floatpnt[1])) )
       
    ''' rectangle expand for each direction with given percentage
    Rectangle is defined as left-top to right-bottom
    E.g., from (0, 0, 10, 10) to (-1, -1, 11, 11) if percentage is 10
    '''
    def rectexpand(self, rect=None, pct=None):
        l, t, r, b=rect
        deltaw=int((r-l)*pct/100) # delta width
        deltah=int((b-t)*pct/100) # delta height
        el=l-deltaw
        et=t-deltah
        er=r+deltaw
        eb=b+deltah
        return (el, et, er, eb)
        
    ''' rectangle intersection
    E.g., from (0, 0, 10, 12) intersect to (-1, 1, 11, 11) gives
        (0, 1, 10, 11)
    '''
    def rectintersect(self, rect1=None, rect2=None):
        l1, t1, r1, b1=rect1
        l2, t2, r2, b2=rect2
        return (max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2))

    ''' triangle's three edge set
    '''
    def triangle2edge(self, t=None):
        return set([(t[0], t[1], t[2], t[3]),
                    (t[2], t[3], t[4], t[5]),
                    (t[4], t[5], t[0], t[1])])

if __name__ == '__main__':
    mylog = os.path.realpath(__file__).replace('.py', '.log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)

    logger.info(f'start python code {__file__}.\n')
    gi=GeoInt()
    pntlst=list()
    for x in range(10):
        pntlst.append((random.randint(1,101),random.randint(1,101)))
    logger.info(f"rect {gi.rect4pnts(pntlst=pntlst)} includes {pntlst}")
    triangle=(random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101))
    logger.info(f"rect {gi.rect4triangle(t=triangle)} includes {triangle}")
    trilst=list()
    for x in range(4):
        trilst.append((random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101)))
    logger.info(f"rect {gi.rect4trilist(tl=trilst)} includes {trilst}")
    for x in range(10):
        triangle=(random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101)
              , random.randint(1,101),random.randint(1,101))
        rect=gi.rect4triangle(t=triangle)
        pnt=(rect[0]+(rect[2]-rect[0])*random.randint(1,101)/101.0
            ,rect[1]+(rect[3]-rect[1])*random.randint(1,101)/101.0)
        logger.info(f"{gi.pnt_in_triangle(pnt=pnt,t=triangle)} if {pnt} in {triangle}")
