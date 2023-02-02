import time
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
import niqe
import save_stats
from numba import jit,prange
import argparse

parser = argparse.ArgumentParser(description='Generate ChipQA-0 features from a folder of videos and store them')
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
def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
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

    return mscn_buffer


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
def find_sts_polar(cos_theta,sin_theta,cy,cx,r):
    x = np.asarray(r*cos_theta,dtype=np.int32)
    y = np.asarray(r*sin_theta,dtype=np.int32)
    x_sts = cx+x
    y_sts = cy+y
    return x_sts,y_sts
@jit(nopython=True)
def lut_find(theta,fy,fx,rst,rct):
    st_theta = np.pi/2+np.arctan2(fy, fx)
    indices = np.searchsorted(theta,st_theta)
    rsin_theta = rst[:,indices]
    rcos_theta  =rct[:,indices]
    return rcos_theta,rsin_theta 

def lut_find_sts(img_buffer,grad_img_buffer,step,cy,cx,fx,fy,rst,rct,theta):
    h, w = img_buffer[step-1].shape[:2]
    sts =np.zeros((h,w)) 
    grad_sts = np.zeros((h,w)) #grad_img_buffer[step-1]
    start = time.time()
    rcos_theta,rsin_theta = lut_find(theta,fy,fx,rst,rct) #cx[None,:]+rcos_theta,cy[None,:]+rsin_theta
    x_sts,y_sts = cx[None,:]+rcos_theta,cy[None,:]+rsin_theta
    end = time.time()
    print(end-start,' time for LUT')

    start = time.time()
    for i in range(len(cy)):      
        sts[cy[i]-2:cy[i]+3,cx[i]-2:cx[i]+3] = img_buffer[:,y_sts[:,i],x_sts[:,i]]
        grad_sts[cy[i]-2:cy[i]+3,cx[i]-2:cx[i]+3] = grad_img_buffer[:,y_sts[:,i],x_sts[:,i]]
    end = time.time()
    print(end-start,' is the time for array indexing')
    return sts,grad_sts
def find_sts(img_buffer,grad_img_buffer,step,cy,cx,fx,fy):
    h, w = img_buffer[step-1].shape[:2]
    sts_slope =np.tan(np.pi/2+np.arctan2(fy, fx))                  #the direction along the STS plane which lies in the spatial image

    sts =np.zeros((h,w)) 
    grad_sts =np.zeros((h,w)) 

    r = np.arange(-2,3)
    for i in range(len(cy)):      
        x_sts,y_sts = find_sts_locs(sts_slope[i],cy[i],cx[i],step,h,w)
        sts[cy[i]-2:cy[i]+3,cx[i]-2:cx[i]+3] = img_buffer[:,y_sts,x_sts]
        grad_sts[cy[i]-2:cy[i]+3,cx[i]-2:cx[i]+3] = grad_img_buffer[:,y_sts,x_sts]
    return sts,grad_sts

def sts_fromfilename(i,filenames,results_folder):
    filename = filenames[i]
    st_time_length = 5
    name = os.path.basename(filename)
    print(name) 
    cap = cv2.VideoCapture(filename)
    count=1
    ret, prev = cap.read()
    print(ret)
    scale_percent = 0.5
    theta = np.arange(-np.pi/2,3*np.pi/2+0.1,0.3)
    ct = np.cos(theta)
    st = np.sin(theta)
    r = np.arange(-2,3)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)

    prevY = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevY = prevY.astype(np.float32)
    gradient_x = cv2.Sobel(prevY,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(prevY,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

    prev_grad_mscn, _, _ = compute_image_mscn_transform(gradient_mag)
    prev_mscn, _, _ = compute_image_mscn_transform(prevY)
    
    img_buffer =queue.Queue(maxsize=st_time_length)
    img_grad_buffer =queue.Queue(maxsize=st_time_length)
    img_buffer.put(prev_mscn.astype(np.float32)) 
    img_grad_buffer.put(prev_grad_mscn.astype(np.float32))

    step = st_time_length
    h,w = prev.shape[0],prev.shape[1]

    # dsize
    dsize = (int(scale_percent*h),int(scale_percent*w))

    cy, cx = np.mgrid[step:h-step:step, step:w-step:step].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step:step, step:dsize[1]-step:step].reshape(2,-1).astype(int) # these will be the centers of each block

    prevY_down = cv2.resize(prevY,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)

    gradient_x = cv2.Sobel(prevY_down,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(prevY_down,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

    prev_grad_mscn, _, _ = compute_image_mscn_transform(gradient_mag)
    prev_mscn, _, _ = compute_image_mscn_transform(prevY_down)
    
    head, tail = os.path.split(filename)


    
    spat_list = []
    X_list = []
    down_img_buffer =queue.Queue(maxsize=st_time_length)
    down_img_grad_buffer =queue.Queue(maxsize=st_time_length)
    down_img_buffer.put(prev_mscn.astype(np.float32)) 
    down_img_grad_buffer.put(prev_grad_mscn.astype(np.float32))

    
    j=0
    total_time = 0

    while(True):
        # try:
            # 
        j = j+1
#            print('Frame ',j)
        
        
        ret,bgr = cap.read()
        count=count+1
        print(count)
        if(ret==False):
            count=count-1
            break

        whole_start = time.time()
        Y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        Y = Y.astype(np.float32)
        spat_feats = niqe.compute_niqe_features(Y)
        spat_list.append(spat_feats)
        gradient_x = cv2.Sobel(Y,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        Y_grad_mscn,_,_ = compute_image_mscn_transform(gradient_mag)
        Y_mscn,_,_ = compute_image_mscn_transform(Y,C=C)
        Y_down = cv2.resize(Y,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
        
        gradient_x = cv2.Sobel(Y_down,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y_down,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x**2+gradient_y**2)    

        Ydown_grad_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down)
        Ydown_mscn,_,_ = compute_image_mscn_transform(Y_down)

        start = time.time()
        flow =cv2.calcOpticalFlowFarneback(prevY,Y, None, 0.5, 3, 15, 3, 5, 1.2, 0) #cv2.calcOpticalFlowPyrLK(prevgray, gray) 
        down_flow=cv2.calcOpticalFlowFarneback(prevY_down,Y_down, None, 0.5, 3, 15, 3, 5, 1.2, 0) #cv2.calcOpticalFlowPyrLK(prevgray, gray) 
        end = time.time()
    
        print(end-start,' is the time for opt flow')

        prevY = Y
        prevY_down = Y_down
        
        img_buffer.put(Y_mscn.astype(np.float32))
        img_grad_buffer.put(Y_grad_mscn.astype(np.float32))
        
        down_img_buffer.put(Ydown_mscn.astype(np.float32))
        down_img_grad_buffer.put(Ydown_grad_mscn.astype(np.float32))
        if (down_img_buffer.qsize()>=st_time_length): 
            med_flow = cv2.medianBlur(flow,5)
            fy, fx =med_flow[cy,cx].T# flow[:,:,0].flatten(),flow[:,:,1].flatten() #
            start = time.time()
#            _,sts_grad = find_sts(np.array(img_buffer.queue),np.array(img_grad_buffer.queue),st_time_length,cy,cx,fx.astype(np.float32),fy.astype(np.float32))                
            sts,sts_grad = lut_find_sts(np.array(img_buffer.queue),np.array(img_grad_buffer.queue),st_time_length,cy,cx,fx.astype(np.float32),fy.astype(np.float32),rst,rct,theta)                
            end=time.time()
            print(end-start,' is the time for STS')
            feats =  save_stats.brisque(sts)
            grad_feats =  save_stats.brisque(sts_grad)
            

            down_med_flow = cv2.medianBlur(down_flow,5)
            dfy, dfx =down_med_flow[dcy,dcx].T# flow[:,:,0].flatten(),flow[:,:,1].flatten() #
            start = time.time()
            dsts,dsts_grad = lut_find_sts(np.array(down_img_buffer.queue),np.array(down_img_grad_buffer.queue),st_time_length,dcy,dcx,dfx.astype(np.float32),dfy.astype(np.float32),rst,rct,theta)                
#            _,dsts_grad = find_sts(np.array(down_img_buffer.queue),np.array(down_img_grad_buffer.queue),st_time_length,dcy,dcx,dfx.astype(np.float32),dfy.astype(np.float32))                
            end=time.time()
            print(end-start,' is the time for DSTS')
            dfeats =  save_stats.brisque(dsts)
            dgrad_feats =  save_stats.brisque(dsts_grad)

            allfeats = np.concatenate((feats,dfeats,grad_feats,dgrad_feats))
            X_list.append(allfeats)
            img_buffer.get()
            img_grad_buffer.get()
            down_img_buffer.get()
            down_img_grad_buffer.get()
        end = time.time()
#        print(end-whole_start,' is the total time')
    X = np.concatenate((np.average(spat_list,axis=0),np.average(X_list,axis=0)))
    train_dict = {"features":X}
    filename =os.path.join(os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,os.path.join(results_folder,filename))
    return


def sts_fromvid(args):
    filenames = glob.glob(os.path.join(args.input_folder,'*.mp4'))
    filenames = sorted(filenames)
    flag = 0
    Parallel(n_jobs=-10)(delayed(sts_fromfilename)(i,filenames,args.results_folder) for i in range(len(filenames)))
#    sts_fromfilename(144,filenames,args.results_folder)


    return


def main():
    args = parser.parse_args()
    sts_fromvid(args)


if __name__ == '__main__':
    # print(__doc__)
    main()
    

