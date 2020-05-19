import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

from facesmile import FaceSmile
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
    # Assume you have finished initialization, just load existing model
    fs=FaceSmile(pic=f'{jpgpath}/sue.jpg'
                 , pnt68=f'{txtpath}/sue68.txt'
                 , dt=f'{txtpath}/sue_dt.txt')
    '''
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
    fs.face2dt(boundary=True
               , pntdrop=True, morphlist=[49, 50, 60, 61, 54, 55, 56, 65]
               , bigtrianglesplit=True, arearatio=121
    )
    fs.img68pnt2jpg(outpic=f'{outpath}/sue_opt.jpg'
                    , pntcolor=(255, 255, 255) # white
                    , font=None # default
                    , fontsize=0.4
                    , fontcolor=(255, 0, 0)
    )
    fs.imgdt2jpg(outpic=f'{outpath}/sue_dto.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
    #             , new_color=(255,0,0)
    )
    '''
    fs.faceprep(morphlist=[49, 50, 60, 61, 54, 55, 56, 65]
                , newpntlist=[(170,660), (200, 650), (204, 700), (193,670)
                              , (330, 642), (385, 652), (338,688), (364, 659)]
                , boundary=True
                , pntdrop=True, distratio=100
                , bigtrianglesplit=True, arearatio=1600
               )
    fs.imgdt2jpg(outpic=f'{outpath}/sue_plan.jpg', delaunay_color=(255,255,255)
                 , morph_color=(0,255,255)
                 , new_color=(255,0,0)
    )
    fs.facemorphing(outpic=f'{outpath}/sue_new.jpg')
    
