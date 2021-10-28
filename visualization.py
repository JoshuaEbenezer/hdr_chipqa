import time
import pandas as pd
from utils.hdr_utils import hdr_yuv_read
from utils.csf_utils import blockwise_csf,windows_csf
from joblib import Parallel,delayed
import numpy as np
import cv2
import queue
import glob
from colour.models import eotf_PQ_BT2100
import os
import time
import scipy.ndimage
import joblib
from scipy.stats import gmean
import sys
import matplotlib.pyplot as plt
import ChipQA.niqe 
import ChipQA.save_stats
from numba import jit,prange,njit
import argparse
from scipy.stats import kurtosis
from skimage.util.shape import view_as_blocks
C=1

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
def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='reflect'):
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
    mscn = (image - mu_image)/(var_image + C)
    return mscn, image - mu_image, var_image

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='mirror'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean

def block_compute_lnl(block,delta,lnl_transform='exp_inv'):
    block = block.astype(np.float32)
    avg_luminance = np.average(block.flatten()+1)
    if(lnl_transform=='sigmoid'):
        block_transform =  1/(1+np.exp(-delta*(block-avg_luminance)))
    elif(lnl_transform=='nakarushton'):
        block_transform = block/(block+avg_luminance)
    elif(lnl_transform=='exp'):
        block = -4+(block-np.amin(block))* 8/(1e-3+np.amax(block)-np.amin(block))
        block_transform =  np.exp(np.abs(block)**delta)-1
        block_transform[block<0] = -block_transform[block<0]
    elif(lnl_transform=='logit'):
        block = -0.99+(block-np.amin(block))* 1.98/(1e-3+np.amax(block)-np.amin(block))
        block_transform = np.log((1+(block)**delta)/(1-(block)**delta))
        if(delta%2==0):
            block_transform[block<0] = -block_transform[block<0] 
            
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

@njit
def padding(img,pad):
    padded_img = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
    padded_img[pad:-pad,pad:-pad] = img
    return padded_img

@njit(parallel=True)
def AdaptiveLogit(img,s,delta):
    if len(img.shape) == 3:
        raise Exception ("Single channel image only")

    H,W = img.shape
    a = s//2
    padded_img = padding(img,a)

    f_img = np.zeros(padded_img.shape)

    for i in prange(a,H+a+1):
        for j in range(a,W+a+1):
            value = logit_window(padded_img,i,j,s,delta)
            f_img[i,j] = value

    return f_img[a:-a,a:-a] 

@njit
def logit(x,delta):
    if(delta%2==1):
        return np.log((1+(x)**delta)/(1-(x)**delta))
    else:
        if(x<0):
            return -np.log((1+(x)**delta)/(1-(x)**delta))
        else:
            return np.log((1+(x)**delta)/(1-(x)**delta))
@njit
def logit_window(Y,x,y,s,delta):
    block = Y[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
    center_pixel_scaled = -0.99+(block[s//2,s//2]-np.amin(block))* 1.98/(1e-3+np.amax(block)-np.amin(block))
    result = logit(center_pixel_scaled,delta)
            

            
    return result
            
def spatial_mscn(i,filenames,framenum,hs,ws,use_csf=True,linear=True):
    filename = filenames[i] # '/home/josh-admin/Downloads/288p_200kbps_CenterPanorama_upscaled.yuv' #
    name = os.path.basename(filename)
    print(name) 
    framenos = framenos_list[i]



    # SIZE of frames
    h,w = hs[i],ws[i]
    print(h,w)
    if(h>w):
        h_temp = h
        h=w
        w = h_temp
        
    # SIZE of windows
    h_win,w_win = 45,45
    max_h,max_w = int((h//h_win)*h_win),int((w//w_win)*w_win)
    #percent by which the image is resized
    scale_percent = 0.5
    # dsize
    dsize = (int(scale_percent*h),int(scale_percent*w))

    # opening file object
    dis_file_object = open(filename)
    prevY_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)
    count=1
    prevY_pq = prevY_pq.astype(np.float32)
    if(linear==True):
        y_pq_normalized = prevY_pq.astype(np.float32)/1024.0
        prevY_pq = eotf_PQ_BT2100(y_pq_normalized)

#     blocks = blockshaped(prevY_pq[:max_h,:max_w],h_win,w_win)
#     Y = unblockshaped(blocks,max_h,max_w)
    Y_mscn,Y_ms = compute_image_mscn_transform(Y)
    lnl_mscn_list =[]
    lnl_list = []
    lnl_ms_list = []
    
    for delta in [1,2,3]:
#         block_lnl = Parallel(n_jobs=20,verbose=0)(delayed(block_compute_lnl)(block,delta,lnl_transform='logit')\
#                                                               for block in blocks) 
#         Y_lnl = unblockshaped(np.asarray(block_lnl),max_h,max_w)
        Y_lnl= block_compute_lnl(Y,delta,lnl_transform='logit')
#         start = time.time()
#         Y_lnl = AdaptiveLogit(Y,s=45,delta=delta)
#         end = time.time()
#         print(end-start)
        Y_lnl_mscn,Y_lnl_ms = compute_image_mscn_transform(Y_lnl,C=1e-3)
        
        lnl_list.append(Y_lnl)
        lnl_mscn_list.append(Y_lnl_mscn)
        lnl_ms_list.append(Y_lnl_ms)
    return Y, Y_mscn,lnl_list,lnl_mscn_list,lnl_ms_list



input_folder = '/media/josh/nebula_josh/hdr/fall2021_hdr_yuv'
csv_file = './fall2021_yuv_rw_info.csv'
csv_df = pd.read_csv(csv_file)
files = [os.path.join(input_folder,f) for f in csv_df["yuv"]]
fps = csv_df["fps"]
framenos_list = csv_df["framenos"]
ws =csv_df["w"]
hs = csv_df["h"]
flag = 0



delta =[1,2,3]
from scipy.stats import t

ms_means = []
mscn_means = []
std_means = []
for i,f in enumerate(files):
    if('ref' in f):
        base = os.path.splitext(os.path.basename(f))[0]
        print('''\\begin{subfigure}{0.15\\textwidth} 
        \\includegraphics[width=\\linewidth]{images/orig/'''+base+'''_orig.png}
        \\end{subfigure}\\hfil % <-- added
        \\begin{subfigure}{0.15\\textwidth} 
        \\includegraphics[width=\\linewidth]{images/ref_mscn_fixed_dof2/'''+base+'''_hist.png}
        \\end{subfigure}\\hfil % <-- added
        \\begin{subfigure}{0.15\\textwidth} 
        \\includegraphics[width=\\linewidth]{images/ref_mscn_fixed_dof3/'''+base+'''_hist.png}
        \\end{subfigure}\\hfil % <-- added
                \\begin{subfigure}{0.15\\textwidth} 
        \\includegraphics[width=\\linewidth]{images/ref_mscn_fixed_dof4/'''+base+'''_hist.png}
        \\end{subfigure}\\hfil % <-- added
                        \\begin{subfigure}{0.15\\textwidth} 
        \\includegraphics[width=\\linewidth]{images/ref_mscn_fixed_dof5/'''+base+'''_hist.png}
        \\end{subfigure}\\hfil % <-- added
        \\begin{subfigure}{0.15\\textwidth}
        \\includegraphics[width=\\linewidth]{images/ref_mscn_fixed_dof10/'''+base+'''_hist.png}

        \\end{subfigure}\\\\''')
#        dis_file_object = open(f)
#        Y,_,_ = hdr_yuv_read(dis_file_object,10,2160,3840)
#        Y_mscn,Y_ms,local_std = compute_image_mscn_transform(Y)
#
##         Y,Y_mscn,Y_lnl_list,Y_lnl_mscn_list,lnl_ms_list = spatial_mscn(i,files,10,hs,ws,linear=False)
#        Yplot = np.concatenate(Y_mscn).flatten()
#        alpha,sigma = ChipQA.save_stats.estimateggdparam(Yplot-np.mean(Yplot))
#        kurt = kurtosis(Yplot-np.mean(Yplot),fisher=False)
#        x = np.arange(-1.5,1.5,0.001)
#        Y_ggd = ChipQA.save_stats.generate_ggd(x,alpha,sigma)
#        student_t = t.fit(Yplot.flatten(),f0=3)
#        print(student_t)
#        student_t_fitted = t.pdf(x, loc=student_t[1], scale=student_t[2], df=student_t[0])
#        plt.figure()
#        plt.clf()
#        n,bins,_ = plt.hist(Y_mscn.flatten(),range=[-1.5,1.5],bins=2500,label=r'Empirical',density=True)
#        print(len(bins))
#        #         plt.plot(x,Y_ggd,label=r'GGD fit $\alpha=$'+str(alpha)[:4])
#        plt.plot(x,student_t_fitted,label=r'Student t fit $\nu=$'+str(student_t[0])[:5]+' loc= '+str(student_t[1])[:5]+' scale= '+str(student_t[2])[:5])
#        plt.title('MSCN of '+base)
#        plt.legend()
##         plt.imshow(Y,cmap='gray')
#
#        plt.savefig('./images/ref_mscn_fixed_dof3/'+base+'_hist.png')
#



# def get_aggd(aggd_feats):
#     alpha = aggd_feats[0]
#     sigma_l  = np.sqrt(aggd_feats[2])
#     sigma_r  = np.sqrt(aggd_feats[3])
#     x1 = np.arange(-2,0,0.001)
#     x2 = np.arange(0,2,0.001)
#     Y_aggd = ChipQA.save_stats.generate_aggd(x1,x2,alpha,sigma_l,sigma_r)
#     return Y_aggd

# Y_lnl_mscn_list.append(Y_mscn)
# delta =[1,3,5,'orig']

# for index,Y_lnl_mscn in enumerate(Y_lnl_mscn_list):
#     pps1, pps2, pps3, pps4 = ChipQA.save_stats.paired_product(Y_lnl_mscn)
#     aggd_features = ChipQA.save_stats.all_aggd(Y_lnl_mscn)
#     print(aggd_features.shape)
#     first_order = aggd_features[0:4]
#     H_feats = aggd_features[4:8]
#     V_feats = aggd_features[8:12]
#     D1_feats = aggd_features[12:16]
#     D2_feats = aggd_features[16:20]

#     H_aggd = get_aggd(H_feats)
#     V_aggd = get_aggd(V_feats)
#     D1_aggd = get_aggd(D1_feats)
#     D2_aggd = get_aggd(D2_feats)

#     x = np.arange(-2,2,0.001)
    
#     Yplot = pps1.flatten()
#     kurt = kurtosis(Yplot-np.mean(Yplot),fisher=False)

#     plt.figure()
#     plt.hist(Yplot,bins='auto',histtype='step',label=r'$\alpha=$'+str(H_feats[0])[:4]+'\nkurtosis='+str(kurt)[:4]+'\n$\delta=$'+str(delta[index]),density=True)
#     plt.plot(x,H_aggd)
#     plt.ylabel('Empirical distribution')
#     plt.xlabel('MSCN')
#     plt.title('Histogram of nonlinear Horizontal MSCN')
#     plt.legend()
#     plt.savefig('./images/delta_'+str(delta[index])+'_H_aggd.png')
    

# plt.show()
# Yplot = np.concatenate(Y_mscn).flatten()
# alpha,sigma = ChipQA.save_stats.estimateggdparam(Yplot-np.mean(Yplot))
# kurt = kurtosis(Yplot-np.mean(Yplot),fisher=False)
# x = np.arange(-2,2,0.001)
# Y_ggd = ChipQA.save_stats.generate_ggd(x,alpha,sigma)

# plt.figure()
# plt.hist(Yplot-np.mean(Yplot),bins='auto',histtype='step',label=r'$\alpha=$'+str(alpha)[:4]+'\nkurtosis='+str(kurt)[:4],density=True)
# plt.plot(x,Y_ggd)
# plt.ylabel('Empirical distribution')
# plt.xlabel('MSCN')
# plt.title('Histogram of original image MSCNs')
# plt.legend()
# plt.show()
    
# plt.figure()
# plt.clf()
# Yplot = Y.flatten()
# plt.hist(Yplot,bins='auto',histtype='step',label=r'original image',density=True)
# plt.legend()
