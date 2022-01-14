import time
import colour
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
from numba import jit,prange,njit
import argparse

os.nice(1)

parser = argparse.ArgumentParser(description='Generate ChipQA features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')
parser.add_argument('--hdr', dest='hdr', help='Set option if running on HDR YUV',action='store_true')
parser.set_defaults(hdr=False)

args = parser.parse_args()
print(args)
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
    return data_slice


def find_kurtosis_sts(grad_img_buffer,step,cy,cx,rst,rct,theta):

    h, w = grad_img_buffer[step-1].shape[:2]
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(step,-1))
    sts_grad_data = [find_kurtosis_slice(gradY3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]

    return sts_grad_data


def Y_compute_lnl(Y,nl_method='exp',nl_param=1):
    Y = Y.astype(np.float32)

    if(nl_method=='logit'):
        maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
        minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
        delta = nl_param=1
        Y_scaled = -0.99+1.98*(Y-minY)/(1e-3+maxY-minY)
        Y_transform = np.log((1+(Y_scaled)**delta)/(1-(Y_scaled)**delta))
        if(delta%2==0):
            Y_transform[Y<0] = -Y_transform[Y<0] 
    elif(nl_method=='exp'):
        maxY = scipy.ndimage.maximum_filter(Y,size=(31,31))
        minY = scipy.ndimage.minimum_filter(Y,size=(31,31))
        delta = nl_param
        Y = -4+(Y-minY)* 8/(1e-3+maxY-minY)
        Y_transform =  np.exp(np.abs(Y)**delta)-1
        Y_transform[Y<0] = -Y_transform[Y<0]
    elif(nl_method=='sigmoid'):
        avg_luminance = scipy.ndimage.gaussian_filter(Y,sigma=7.0/6.0)
        Y_transform = 1/(1+(np.exp(-(1e-3*(Y-avg_luminance)))))
    return Y_transform




def full_hdr_chipqa_forfile(i,filenames,results_folder,hdr,framenos_list=[]):
    filename = filenames[i]
    if(os.path.exists(filename)==False):
        return
    name = os.path.basename(filename)
    print(name) 
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)):
        print(name, ' has output already')
        return
    if(hdr):
        framenos = framenos_list[i]
    else:
        cap = cv2.VideoCapture(filename)
        framenos = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ## PARAMETERS for the model
    st_time_length = 5
    t = np.arange(0,st_time_length)
    a=0.5
    # temporal filter
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)



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
    h,w = 2160,3840
    if(h>w):
        h_temp = h
        h=w
        w = h_temp
    #percent by which the image is resized
    scale_percent = 0.5
    # dsize
    dsize = (int(scale_percent*h),int(scale_percent*w))

    # opening file object
    if(hdr):
        dis_file_object = open(filename)
        prevY_pq,_,_ = hdr_yuv_read(dis_file_object,0,h,w)
        prevY_pq = prevY_pq.astype(np.float32)
    else:
        ret, bgr = cap.read()
        if(ret==False):
            return
        prevY_pq = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


    # ST chip centers and parameters
    step = st_time_length
    cy, cx = np.mgrid[step:h-step*4:step*4, step:w-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step*4:step*4, step:dsize[1]-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    r1 = len(np.arange(step,h-step*4,step*4)) 
    r2 = len(np.arange(step,w-step*4,step*4)) 
    dr1 = len(np.arange(step,dsize[0]-step*4,step*4)) 
    dr2 = len(np.arange(step,dsize[1]-step*4,step*4)) 

    
    prevY_pq_down = cv2.resize(prevY_pq,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)

    grad_img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
    graddown_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))


    gradient_x = cv2.Sobel(prevY_pq,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(prevY_pq,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

    
    gradient_x_down = cv2.Sobel(prevY_pq_down,ddepth=-1,dx=1,dy=0)
    gradient_y_down = cv2.Sobel(prevY_pq_down,ddepth=-1,dx=0,dy=1)
    gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    
    i = 0

    gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag,C=1)
    dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down,C=1)


    grad_img_buffer[i,:,:] =gradY_mscn 
    graddown_img_buffer[i,:,:]=dgradY_mscn 
    i = i+1

    


    
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []

    framenum = 0
    
    while(True):
        # uncomment for FLOPS
        #high.start_counters([events.PAPI_FP_OPS,])
        
        if(hdr):
            if(framenum==framenos-1):
                break
            else:
                framenum=framenum+1
            try:

                Y_pq,U_pq,V_pq = hdr_yuv_read(dis_file_object,framenum,h,w)
                YUV = np.stack((Y_pq,U_pq,V_pq),axis=2)
                YUV = YUV/1023.0
                rgb_frame = colour.YCbCr_to_RGB(YUV,K = [0.2627,0.0593])
                xyz = colour.RGB_to_XYZ(rgb_frame, [0.3127,0.3290], [0.3127,0.3290],\
                        colour.models.RGB_COLOURSPACE_BT2020.RGB_to_XYZ_matrix,\
                        chromatic_adaptation_transform='CAT02',\
                        cctf_decoding=colour.models.eotf_PQ_BT2100)
                lab = colour.XYZ_to_hdr_CIELab(xyz, illuminant=[ 0.3127, 0.329 ], Y_s=0.2, Y_abs=100, method='Fairchild 2011')
#                Y_pq = Y_pq/1023.0

            except:
                f = open("chipqa_yuv_reading_error.txt", "a")
                f.write(filename+"\n")
                f.close()
                break
        else:
            ret, bgr = cap.read()
            if(ret==False):
                break
            # since this is SDR, the Y is gamma luma, not PQ luma, but is named with the PQ suffix for convenience
            Y_pq = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            Y_pq = Y_pq/255.0

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab = lab.astype(np.float32)
        
        
        Y_pq = Y_pq.astype(np.float32)
        Y_down_pq = cv2.resize(Y_pq,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
        
        
        Y_pq_nl = Y_compute_lnl(Y_pq,nl_method='exp',nl_param=1)
        Y_down_pq_nl =Y_compute_lnl(Y_down_pq,nl_method='exp',nl_param=1)

        Y_mscn_pq_nl,_,_ = compute_image_mscn_transform(Y_pq_nl,1)
        dY_mscn_pq_nl,_,_ = compute_image_mscn_transform(Y_down_pq_nl,1)

        brisque_nl_fullscale = ChipQA.save_stats._extract_subband_feats(Y_mscn_pq_nl)
        brisque_nl_halfscale = ChipQA.save_stats._extract_subband_feats(dY_mscn_pq_nl)
        brisque_nl = np.concatenate((brisque_nl_fullscale,brisque_nl_halfscale),axis=0)

        gradient_x = cv2.Sobel(Y_pq,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        
        gradient_x_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    


        Y_mscn,_,_ = compute_image_mscn_transform(Y_pq,C=1)
        dY_mscn,_,_ = compute_image_mscn_transform(Y_down_pq,C=1)

        gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag,C=1)
        dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down,C=1)

        brisque_fullscale = ChipQA.save_stats._extract_subband_feats(Y_mscn)
        brisque_halfscale = ChipQA.save_stats._extract_subband_feats(dY_mscn)
        brisque = np.concatenate((brisque_fullscale,brisque_halfscale),axis=0)


        chroma = np.sqrt(lab[:,:,1]**2+lab[:,:,2]**2)
        chroma_down = cv2.resize(chroma,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
        chroma_alpha,chroma_sigma = ChipQA.save_stats.estimateggdparam(chroma.flatten())
        dchroma_alpha,dchroma_sigma = ChipQA.save_stats.estimateggdparam(chroma_down.flatten())
        chroma_ggd_feats = np.asarray([chroma_alpha,chroma_sigma,dchroma_alpha,dchroma_sigma])

        chroma_gradient_x = cv2.Sobel(chroma,ddepth=-1,dx=1,dy=0)
        chroma_gradient_y = cv2.Sobel(chroma,ddepth=-1,dx=0,dy=1)
        gradient_mag_chroma = np.sqrt(chroma_gradient_x**2+chroma_gradient_y**2)
        gradient_mag_chroma = gradient_mag_chroma/np.amax(gradient_mag_chroma)
        chroma_grad_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(gradient_mag_chroma,C=1)

        chroma_gradient_x_down= cv2.Sobel(chroma_down,ddepth=-1,dx=1,dy=0)
        chroma_gradient_y_down = cv2.Sobel(chroma_down,ddepth=-1,dx=0,dy=1)
        gradient_mag_chroma_down = np.sqrt(chroma_gradient_x_down**2+chroma_gradient_y_down**2)
        gradient_mag_chroma_down = gradient_mag_chroma_down/np.amax(gradient_mag_chroma_down)
        chroma_grad_mscn_down,_,_ = ChipQA.save_stats.compute_image_mscn_transform(gradient_mag_chroma_down,C=1)


        corr = chroma_grad_mscn*gradY_mscn
        corr_down = chroma_grad_mscn_down*dgradY_mscn
        alpha1, N1, _, _, lsq1, rsq1 = ChipQA.save_stats.aggd_features(corr)
        dalpha1, dN1 ,_, _, dlsq1, drsq1 = ChipQA.save_stats.aggd_features(corr_down)
        colorbleed_features =  np.array([alpha1, N1, lsq1**2, rsq1**2,dalpha1, dN1,dlsq1**2, drsq1**2])

        # [0:36] - BRISQUE
        # [36:72] - BRISQUE after NL
        # [72:76] - Chroma GGD
        # [76:84] - Chroma Colorbleed
        feats = np.concatenate((brisque,brisque_nl,chroma_ggd_feats,colorbleed_features),axis=0)

        feat_sd_list.append(feats)
        spatavg_list.append(feats)

        
        grad_img_buffer[i,:,:] =gradY_mscn 
        graddown_img_buffer[i,:,:]=dgradY_mscn 
        i=i+1


        if (i>=st_time_length): 

            grad3d_mscn = spatiotemporal_mscn(grad_img_buffer,avg_window)
            graddown3d_mscn = spatiotemporal_mscn(graddown_img_buffer,avg_window)

            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []

            sts_grad= find_kurtosis_sts(grad3d_mscn,step,cy,cx,rst,rct,theta)
            dsts_grad = find_kurtosis_sts(graddown3d_mscn,step,dcy,dcx,rst,rct,theta)
            sts_grad = np.reshape(sts_grad,(r1*st_time_length,r2*st_time_length))
            dsts_grad = np.reshape(dsts_grad,(dr1*st_time_length,dr2*st_time_length))
            grad_feats =  ChipQA.save_stats.brisque(sts_grad)
            dgrad_feats = ChipQA.save_stats.brisque(dsts_grad)
            
            allst_feats = np.concatenate((grad_feats,dgrad_feats),axis=0)
            X_list.append(allst_feats)


            grad_img_buffer = np.zeros((st_time_length,prevY_pq.shape[0],prevY_pq.shape[1]))
            graddown_img_buffer =np.zeros((st_time_length,prevY_pq_down.shape[0],prevY_pq_down.shape[1]))
            i=0
#            x=high.stop_counters()
#        print(x,"is the number of flops")

    X1 = np.average(spatavg_list,axis=0)
    # [84:168] - SD feats
    X2 = np.average(sd_list,axis=0)
    # [168:204] - ST Gradient feats
    X3 = np.average(X_list,axis=0)
    X = np.concatenate((X1,X2,X3),axis=0)
    train_dict = {"features":X}
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,filename_out)
    return


def sts_fromvid(args):

    if(args.hdr):
        csv_file = './fall2021_yuv_rw_info.csv'
        csv_df = pd.read_csv(csv_file)
        print(csv_df)
        print([f for f in csv_df["yuv"]])
        files = [os.path.join(args.input_folder,f[:-4]+'_upscaled.yuv') for f in csv_df["yuv"]]
        print(files)
        fps = csv_df["fps"]
        framenos_list = csv_df["framenos"]
    else:
        files = glob.glob(os.path.join(args.input_folder,'*.mp4'))
        print(files)
        framenos_list = []
    
    outfolder = args.results_folder
    if(os.path.exists(outfolder)==False):
        os.mkdir(outfolder)
    Parallel(n_jobs=60,backend='multiprocessing')(delayed(full_hdr_chipqa_forfile)\
            (i,files,outfolder,args.hdr,framenos_list)\
            for i in range(len(files)))
#    for i in range(len(files)):
#        full_hdr_chipqa_forfile(i,files,outfolder,args.hdr,framenos_list)
#    for i in range(len(files)):
#        sts_fromfilename(i,files,framenos_list,args.results_folder,ws,hs,nl_method='exp'='nakarushton',use_csf=False,use_lnl=False)
             



    return


def main():
    args = parser.parse_args()
    sts_fromvid(args)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

