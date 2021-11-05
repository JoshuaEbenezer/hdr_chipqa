import time
from colour.models import eotf_PQ_BT2100
import pandas as pd
from utils.hdr_utils import hdr_yuv_read
from utils.csf_utils import blockwise_csf,windows_csf
from joblib import Parallel,delayed
import numpy as np
import cv2
import queue
import glob
import os
import scipy.ndimage
import joblib
import sys
import matplotlib.pyplot as plt
import ChipQA.niqe 
import ChipQA.save_stats
from numba import jit,prange
import argparse

parser = argparse.ArgumentParser(description='Generate ChipQA features from a folder of videos and store them')
parser.add_argument('input_folder',help='Folder containing input videos')
parser.add_argument('results_folder',help='Folder where features are stored')

args = parser.parse_args()

def block_compute_lnl(block,lnl_method):
    block = block.astype(np.float32)
    avg_luminance = np.average(block.flatten()+1)
    if(lnl_method=='nakarushton'):
        block_transform =  block/(block+avg_luminance) #
    elif(lnl_method=='sigmoid'):
        block_transform = 1/(1+(np.exp(-(1e-3*(block-avg_luminance)))))
    elif(lnl_method=='logit'):
        delta = 2 
        block_scaled = -0.99+1.98*(block-np.amin(block))/(1e-3+np.amax(block)-np.amin(block))
        block_transform = np.log((1+(block_scaled)**delta)/(1-(block_scaled)**delta))
        if(delta%2==0):
            block_transform[block<0] = -block_transform[block<0] 
    elif(lnl_method=='exp'):
        delta = 1
        block = -4+(block-np.amin(block))* 8/(1e-3+np.amax(block)-np.amin(block))
        block_transform =  np.exp(np.abs(block)**delta)-1
        block_transform[block<0] = -block_transform[block<0]
    elif(lnl_method=='custom'):
        block = -0.99+(block-np.amin(block))* 1.98/(1e-3+np.amax(block)-np.amin(block))
        block_transform = transform(block,5)


    return block_transform

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def lnl(Y,lnl_method,h_win,w_win):
    h,w = Y.shape
    max_h,max_w = int((h//h_win)*h_win),int((w//w_win)*w_win)
    blocks = blockshaped(Y[:max_h,:max_w],h_win,w_win)
    block_lnl = Parallel(n_jobs=-20)(delayed(block_compute_lnl)(block,lnl_method) \
            for block in blocks)
    Y_lnl = unblockshaped(np.asarray(block_lnl),max_h,max_w)
    return Y_lnl

def logit(Y):
    Y = -0.99+(Y-np.amin(Y))* 1.98/(1e-3+np.amax(Y)-np.amin(Y))
    Y_transform = np.log((1+(Y)**3)/(1-(Y)**3))
    return Y_transform





def niqe_fromfilename(i,filenames,framenos_list,results_folder,ws,hs,use_linear = True, use_gnl=False):
    filename = filenames[i]
    if(os.path.exists(filename)==False):
        return
    name = os.path.basename(filename)
    print(name) 
    w = ws[i]
    h = hs[i]
    framenos = framenos_list[i]
    print(framenos,w,h)
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)):
        print('Feature file exists')
        return
    dis_file_object = open(filename)

    niqe_list = []

    C = 4
    for framenum in range(0,framenos,5): 
#        print(framenum)
        try:
            Y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)
            Y_pq = Y_pq.astype(np.float32)
            Y = Y_pq
        except Exception as e:
            print(e)
            break
        if(use_linear):
            y_pq_normalized = Y.astype(np.float32)/1023.0
            Y = eotf_PQ_BT2100(y_pq_normalized)      
        if(use_gnl):
            Y  = block_compute_lnl(Y,lnl_method='sigmoid')
        i=i+1
        niqe_features = ChipQA.niqe.compute_niqe_features(Y)
        niqe_list.append(niqe_features)
        #except Exception as e:
        #    print(e)
        #    break


    X = np.average(niqe_list,axis=0)
    print(X)
    train_dict = {"features":X}
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,filename_out)
    return


def sts_fromvid(args):
    csv_file = './fall2021_yuv_rw_info.csv'
    csv_df = pd.read_csv(csv_file)
    files = [os.path.join(args.input_folder,f[:-4]+'_upscaled.yuv') for f in csv_df["yuv"]]
    print(files)
    fps = csv_df["fps"]
    framenos_list = csv_df["framenos"]
    ws =[3840]*len(csv_df["w"])
    hs = [2160]*len(csv_df["h"])
    flag = 0
    Parallel(n_jobs=-10)(delayed(niqe_fromfilename)\
            (i,files,framenos_list,args.results_folder,ws,hs,use_linear=False,use_gnl=True)\
            for i in range(len(files)))
#    for i in range(len(files)):
#        niqe_fromfilename(i,files,framenos_list,args.results_folder,ws,hs)
             



    return


def main():
    args = parser.parse_args()
    sts_fromvid(args)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

