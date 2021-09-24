import time
import pandas as pd
from utils.hdr_utils import hdr_yuv_read
from utils.csf_utils import blockwise_csf,windows_csf
from joblib import Parallel,delayed
import numpy as np
import cv2
import queue
import glob
import os
import time
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
C=1
def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
def compute_image_mscn_transform(image, C=1e-3, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='mirror'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean

@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,step,h,w):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((step-1)/2),cx+int((step-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<h else h-1 for j in range(step)])
    else:
        #        print(np.abs(sts_slope))
        y_sts = np.arange(cy-int((step-1)/2),cy+int((step-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<w else w-1 for j in range(step)]) 
    return x_sts,y_sts


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,h,step):
    st_kurtosis = np.zeros((len(theta),))
    data = np.zeros((len(theta),step**2))
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data[index,:] =Y3d_mscn[:,y_sts*h+x_sts].flatten() 
        data_mu4 = np.mean((data[index,:]-np.mean(data[index,:]))**4)
        data_var = np.var(data[index,:])
        st_kurtosis[index] = data_mu4/(data_var**2+1e-4)
    idx = (np.abs(st_kurtosis - 3)).argmin()
    
    data_slice = data[idx,:]
    return data_slice,st_kurtosis[idx]-3


def find_kurtosis_sts(img_buffer,grad_img_buffer,step,cy,cx,rst,rct,theta):

    h, w = img_buffer[step-1].shape[:2]
    Y3d_mscn = np.reshape(img_buffer.copy(),(step,-1))
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(step,-1))
    sts= [find_kurtosis_slice(Y3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]
    sts_grad= [find_kurtosis_slice(gradY3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]

    st_data = [sts[i][0] for i in range(len(sts))]
    st_deviation = [sts[i][1] for i in range(len(sts))]
    st_grad_data = [sts_grad[i][0] for i in range(len(sts_grad))]
    st_grad_dev = [sts_grad[i][1] for i in range(len(sts_grad))]
    return st_data,np.asarray(st_deviation),st_grad_data,np.asarray(st_grad_dev)

def block_compute_lnl(block,lnl_method):
    block = block.astype(np.float32)
    avg_luminance = np.average(block.flatten()+1)
    if(lnl_method=='nakarushton'):
        block_transform =  block/(block+avg_luminance) #
    elif(lnl_method=='sigmoid'):
        block_transform = 1/(1+(np.exp(-(1e-3*(block-avg_luminance)))))
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

def lnl(Y,lnl_method,h_win,w_win,max_h,max_w):
    blocks = blockshaped(Y[:max_h,:max_w],h_win,w_win)
    block_lnl = Parallel(n_jobs=-20)(delayed(block_compute_lnl)(block,lnl_method) \
            for block in blocks)
    Y_lnl = unblockshaped(np.asarray(block_lnl),max_h,max_w)
    return Y_lnl



def sts_fromfilename(i,filenames,framenos_list,results_folder,ws,hs,lnl_method,use_csf=True,use_lnl=True):
    filename = filenames[i]
    name = os.path.basename(filename)
    print(name) 
    w = ws[i]
    h = hs[i]
    framenos = framenos_list[i]
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)):
        return
    ## PARAMETERS for the model
    st_time_length = 5
    t = np.arange(0,st_time_length)
    a=0.5
    # temporal filter
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)

    # SIZE of windows
    h_win,w_win = 45,45
    xx, yy = np.mgrid[h_win//2:h-h_win//2:h_win, w_win//2:w-w_win//2:w_win].reshape(2,-1).astype(int)


    # LUT for coordinate search
    theta = np.arange(0,np.pi,np.pi/6)
    ct = np.cos(theta)
    st = np.sin(theta)
    lower_r = int((st_time_length+1)/2)-1
    higher_r = int((st_time_length+1)/2)
    r = np.arange(-lower_r,higher_r)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)

    # SIZE of frames
    print(h,w)
    if(h>w):
        h_temp = h
        h=w
        w = h_temp
    #percent by which the image is resized
    scale_percent = 0.5
    # dsize
    dsize = (int(scale_percent*h),int(scale_percent*w))

    # opening file object
    dis_file_object = open(filename)
    prevY_pq,_,_ = hdr_yuv_read(dis_file_object,0,h,w)
    prevY_pq = prevY_pq.astype(np.float32)


    # ST chip centers and parameters
    step = st_time_length
    cy, cx = np.mgrid[step:h-step*4:step*4, step:w-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step*4:step*4, step:dsize[1]-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    r1 = len(np.arange(step,h-step*4,step*4)) 
    r2 = len(np.arange(step,w-step*4,step*4)) 
    dr1 = len(np.arange(step,dsize[0]-step*4,step*4)) 
    dr2 = len(np.arange(step,dsize[1]-step*4,step*4)) 

    
    prevY_pq_down = cv2.resize(prevY_pq,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
    if(use_lnl):

        # lnl for the original frame
        max_h,max_w = int((h//h_win)*h_win),int((w//w_win)*w_win)
        prevY_pq = lnl(prevY_pq,lnl_method,h_win,w_win,max_h,max_w)

        # lnl for the downsized frame
        max_h_down,max_w_down = int((dsize[0]//h_win)*h_win),int((dsize[1]//w_win)*w_win)
        prevY_pq_down = lnl(prevY_pq_down,lnl_method,h_win,w_win,max_h_down,max_w_down)

        # point centers
        cy, cx = np.mgrid[step:max_h-step*4:step*4, step:max_w-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
        dcy, dcx = np.mgrid[step:max_h_down-step*4:step*4, step:max_w_down-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
        r1 = len(np.arange(step,max_h-step*4,step*4)) 
        r2 = len(np.arange(step,max_w-step*4,step*4)) 
        dr1 = len(np.arange(step,max_h_down-step*4,step*4)) 
        dr2 = len(np.arange(step,max_w_down-step*4,step*4)) 
    if(use_csf):
        #apply CSF
        prevY_pq = blockwise_csf(prevY_pq)
        prevY_pq_down = blockwise_csf(prevY_pq_down)

    print(prevY_pq.shape,prevY_pq_down.shape)
    img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
    grad_img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
    down_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))
    graddown_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))


    gradient_x = cv2.Sobel(prevY_pq,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(prevY_pq,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

    
    gradient_x_down = cv2.Sobel(prevY_pq_down,ddepth=-1,dx=1,dy=0)
    gradient_y_down = cv2.Sobel(prevY_pq_down,ddepth=-1,dx=0,dy=1)
    gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    
    i = 0

    Y_mscn,_,_ = compute_image_mscn_transform(prevY_pq)
    dY_mscn,_,_ = compute_image_mscn_transform(prevY_pq_down)
    gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag)
    dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down)

    img_buffer[i,:,:] = Y_mscn
    down_img_buffer[i,:,:]= dY_mscn
    grad_img_buffer[i,:,:] =gradY_mscn 
    graddown_img_buffer[i,:,:]=dgradY_mscn 
    i = i+1

    
    head, tail = os.path.split(filename)


    
    spat_list = []
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    j=0
    total_time = 0
    for framenum in range(1,framenos): 
        print(framenum)
        # uncomment for FLOPS
        #high.start_counters([events.PAPI_FP_OPS,])
        
        Y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)
        
        
        Y_pq = Y_pq.astype(np.float32)
        Y_down_pq = cv2.resize(Y_pq,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
        
        
        if(use_lnl):
            Y_pq = lnl(Y_pq,lnl_method,h_win,w_win,max_h,max_w)
            Y_down_pq = lnl(Y_down_pq,lnl_method,h_win,w_win,max_h_down,max_w_down)
        if(use_csf):
            #apply CSF
            Y_pq = blockwise_csf(Y_pq)
            Y_down_pq = blockwise_csf(Y_down_pq)

        gradient_x = cv2.Sobel(Y_pq,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        
        gradient_x_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    


        Y_mscn,Ysigma,_ = compute_image_mscn_transform(Y_pq)
        dY_mscn,dYsigma,_ = compute_image_mscn_transform(Y_down_pq)

        gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag)
        dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down)

        gradient_feats = ChipQA.save_stats.extract_secondord_feats(gradY_mscn)
        gdown_feats = ChipQA.save_stats.extract_secondord_feats(dgradY_mscn)
        gfeats = np.concatenate((gradient_feats,gdown_feats),axis=0)

        
        Ysigma_mscn,_,_= compute_image_mscn_transform(Ysigma)
        dYsigma_mscn,_,_= compute_image_mscn_transform(dYsigma)

        sigma_feats = ChipQA.save_stats.stat_feats(Ysigma_mscn)
        dsigma_feats = ChipQA.save_stats.stat_feats(dYsigma_mscn)


        feats = np.concatenate((gfeats,sigma_feats,dsigma_feats),axis=0)

        feat_sd_list.append(feats)
        spatavg_list.append(feats)

        
        img_buffer[i,:,:] = Y_mscn
        down_img_buffer[i,:,:]= dY_mscn
        grad_img_buffer[i,:,:] =gradY_mscn 
        graddown_img_buffer[i,:,:]=dgradY_mscn 
        i=i+1


        if (i>=st_time_length): 

            Y3d_mscn = spatiotemporal_mscn(img_buffer,avg_window)
            Ydown_3d_mscn = spatiotemporal_mscn(down_img_buffer,avg_window)
            grad3d_mscn = spatiotemporal_mscn(grad_img_buffer,avg_window)
            graddown3d_mscn = spatiotemporal_mscn(graddown_img_buffer,avg_window)
            spat_feats = ChipQA.niqe.compute_niqe_features(Y_pq)

            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []

            sts,st_deviation,sts_grad,sts_grad_deviation = find_kurtosis_sts(Y3d_mscn,grad3d_mscn,step,cy,cx,rst,rct,theta)
            dsts,dsts_deviation,dsts_grad,dsts_grad_deviation = find_kurtosis_sts(Ydown_3d_mscn,graddown3d_mscn,step,dcy,dcx,rst,rct,theta)
            sts_arr = np.reshape(sts,(r1*st_time_length,r2*st_time_length)) 
            sts_grad = np.reshape(sts_grad,(r1*st_time_length,r2*st_time_length))
            print()
            dsts_arr = np.reshape(dsts,(dr1*st_time_length,dr2*st_time_length)) #(int((int(dsize[0]/20)*20-step*4)/4),int((int(dsize[1]/20)*20-step*4)/4)))
            dsts_grad = np.reshape(dsts_grad,(dr1*st_time_length,dr2*st_time_length))#(int((int(dsize[0]/20)*20-step*4)/4),int((int(dsize[1]/20)*20-step*4)/4)))
            feats =  ChipQA.save_stats.brisque(sts_arr)
            grad_feats = ChipQA.save_stats.brisque(sts_grad)
            
            dfeats =  ChipQA.save_stats.brisque(dsts_arr)
            dgrad_feats = ChipQA.save_stats.brisque(dsts_grad)


            allst_feats = np.concatenate((spat_feats,feats,dfeats,grad_feats,dgrad_feats),axis=0)
            X_list.append(allst_feats)


            img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
            grad_img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
            down_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))
            graddown_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))
            i=0
#            x=high.stop_counters()
#        print(x,"is the number of flops")

    X1 = np.average(spatavg_list,axis=0)
    X2 = np.average(sd_list,axis=0)
    X3 = np.average(X_list,axis=0)
    X = np.concatenate((X1,X2,X3),axis=0)
    train_dict = {"features":X}
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,filename_out)
    return


def sts_fromvid(args):
    csv_file = './fall2021_yuv_rw_info.csv'
    csv_df = pd.read_csv(csv_file)
    files = [os.path.join(args.input_folder,f) for f in csv_df["yuv"]]
    fps = csv_df["fps"]
    framenos_list = csv_df["framenos"]
    ws =csv_df["w"]
    hs = csv_df["h"]
    flag = 0
    Parallel(n_jobs=-10)(delayed(sts_fromfilename)\
            (i,files,framenos_list,args.results_folder,ws,hs,lnl_method='nakarushton',use_csf=False,use_lnl=True)\
            for i in range(len(files)))
#    for i in range(len(files)):
#        sts_fromfilename(i,files,framenos_list,args.results_folder,ws,hs,lnl_method='nakarushton',use_csf=False,use_lnl=True)
             



    return


def main():
    args = parser.parse_args()
    sts_fromvid(args)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

