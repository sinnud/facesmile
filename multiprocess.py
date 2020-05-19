import sys
import os
import logzero
import traceback # Python error trace
from logzero import logger

import numpy as np
import cv2
import dlib

import multiprocessing
from multiprocessing import Pool
from itertools import chain

def parallelcode(thread=None, img=None, newimg=None):
    size=img.shape
    logger.info('create value of new image...')
    for i in range(size[0]):
        if i%num_threads != thread:
            continue
        for j in range(size[1]):
            newimg[i][j][0]=img[i][j][0]
            newimg[i][j][1]=img[i][j][1]
            newimg[i][j][2]=img[i][j][2]
    pass

def process(args):
    flist, thread, num_threads, gpbox, dbname, schema, yymm, strtmp = args
    mylog = os.path.realpath(__file__).replace('.py', f'_{thread}.log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)
    info(thread, f"The {thread} process")
    processed=[]
    for idx, df in enumerate(sorted(flist), start=1):
        if idx%num_threads != thread:
            continue
        logger.info(f"The {thread} process deal with {df}.")
        logger.info(f"arguments:{gpbox}, {dbname}, {schema}, {yymm}, {idx}")
        parallelcode(thread=thread, gpbox=gpbox, dbname=dbname, schema=schema
                     , yymm=yymm, idx=idx, df=df
                     , strtmp=strtmp)
        processed.append(df)
    logger.info(f"Finish cleaning temporary table linetmp_{yymm}_{thread}.")
    return ', '.join(processed)

def main():
    logger.info('main: test of multiprocessing')
    logger.info('initialization...')
    img=cv2.imread('sue.jpg')
    size=img.shape
    logger.info('generate empty image with same size...')
    newimg=np.empty(size, dtype=int)
    logger.info('create value of new image...')
    for i in range(size[0]):
        if i%2 == 1:
            continue
        for j in range(size[1]):
            newimg[i][j][0]=img[i][j][0]
            newimg[i][j][1]=img[i][j][1]
            newimg[i][j][2]=img[i][j][2]
    logger.info('output new image...')
    cv2.imwrite('t.jpg', newimg)
    logger.info('main: return')
    return
    num_threads = 1
    # num_threads = multiprocessing.cpu_count()
    logger.info(f"Number of used CPU: '{num_threads}'")
    processed=list()
    pool = Pool(num_threads)
    args = [(flist, i, num_threads, gpbox, dbname, schema, yymm, strtmp) for i in range(num_threads)]
    processed.extend(pool.map(process, args))
    pool.close()
    pool.join()
    processed = set(chain.from_iterable([x.split(',') for x in processed]))
    logger.info(f"finish list element number:{len(processed)}.")

    # collect log files
    for thread in range(num_threads):
        threadlog=os.path.realpath(__file__).replace('.py', f'_{thread}.log')
        with open(threadlog, 'r') as f:  
            data = f.read()
        logger.info(f"Collected {thread} log file content:\n{data}")
        os.remove(threadlog)

if __name__ == '__main__':
    mylog = os.path.realpath(__file__).replace('.py', '.log')
    if os.path.exists(mylog):
        os.remove(mylog)
    logzero.logfile(mylog)

    logger.info(f'start python code {__file__}.\n')
    main()
