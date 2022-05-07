from utils.hdr_utils import yuv_read
from utils import colour_utils
import numpy as np
import cv2
import os
import scipy.ndimage
import joblib
import ChipQA.niqe 
import ChipQA.save_stats
from numba import jit
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Generate HDR ChipQA features from a single video')
parser.add_argument('--input_file',help='Input video file')
parser.add_argument('--results_file',help='File where features are stored')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--bit_depth', type=int,choices={8,10,12})
parser.add_argument('--color_space',choices={'BT2020','BT709'})
parser.add_argument('--time_length',type=int,choices={5,10,20,30})
parser.add_argument('--step',type=int,choices={5,10,20,30})

args = parser.parse_args()




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

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='reflect'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean

@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,step,h,width):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((step-1)/2),cx+int((step-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<h else h-1 for j in range(step)])
    else:
        #        print(np.abs(sts_slope))
        y_sts = np.arange(cy-int((step-1)/2),cy+int((step-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<width else width-1 for j in range(step)]) 
    return x_sts,y_sts


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,w):
    st_kurtosis = np.zeros((len(theta),))
    min_kurtosis = 100
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data=Y3d_mscn[:,y_sts*w+x_sts].flatten() 
        data_mu4 = np.mean((data-np.mean(data))**4)
        data_var = np.var(data)
        st_kurtosis = data_mu4/(data_var**2+1e-4)
        if(st_kurtosis<min_kurtosis):
            best_block = Y3d_mscn[:,y_sts*w+x_sts]
            min_kurtosis = st_kurtosis
    
    return best_block 


def find_kurtosis_sts(grad_img_buffer,cy,cx,rst,rct,theta):

    w = grad_img_buffer.shape[2]
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(grad_img_buffer.shape[0],-1))
    sts_grad= [find_kurtosis_slice(gradY3d_mscn,\
        cy[i],cx[i],rst,rct,theta,w) for i in range(len(cy))]

    return sts_grad

def Y_compute_lnl(Y):
    if(len(Y.shape)==2):
        Y = np.expand_dims(Y,axis=2)

    maxY = scipy.ndimage.maximum_filter(Y,size=(17,17,1))
    minY = scipy.ndimage.minimum_filter(Y,size=(17,17,1))
    Y_scaled = -1+(Y-minY)* 2/(1e-3+maxY-minY)
    Y_transform =  np.exp(np.abs(Y_scaled)*4)-1
    Y_transform[Y_scaled<0] = -Y_transform[Y_scaled<0]
    return Y_transform


def unblockshaped(arr, h, width):
    """
    Return an array of shape (h, width) where
    h * width = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, width))



def hdrchipqa_fromvid(filename,filename_out,width,height,framenos,bit_depth,color_space,time_length,step):
    if(os.path.exists(filename)==False):
        print("Input video file does not exist")
        return
    if(os.path.exists(filename_out)):
        print("Output feature file already exists")
        return
    ## PARAMETERS for the model


    lgn_param_dict = {'5':0.5,'10':0.25,'20':0.15,'30':0.1}
    t = np.arange(0,time_length)
    a= lgn_param_dict[str(time_length)]
    # temporal filter
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)

    # LUT for coordinate search
    theta = np.arange(0,np.pi,np.pi/6)
    ct = np.cos(theta)
    st = np.sin(theta)
    lower_r = int((step+1)/2)-1
    higher_r = int((step+1)/2)
    r = np.arange(-lower_r,higher_r)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)

    global_scaling_factor_dict = {'5':0.25,'10':np.sqrt(0.5*0.25),'20':np.sqrt(0.25*0.25),'30':np.sqrt(1/6.0*0.25)}
    global_scaling_factor = global_scaling_factor_dict[str(time_length)]
    #percent by which the image is resized
    scale_percent = 0.5
    # dsize
    full_size = (int(height*global_scaling_factor),int(width*global_scaling_factor))
    dsize = (int(scale_percent*global_scaling_factor*height),int(scale_percent*global_scaling_factor*width))



    # ST chip centers and parameters
    cy, cx = np.mgrid[step:full_size[0]-step:step, step:full_size[1]-step:step].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step:step, step:dsize[1]-step:step].reshape(2,-1).astype(int) # these will be the centers of each block

    

    # declare buffers for ST Chips
    grad_img_buffer = np.zeros((time_length,full_size[0],full_size[1]))
    graddown_img_buffer =np.zeros((time_length,dsize[0],dsize[1]))
    
    i = 0
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    for framenum in range(framenos): 
        
        try:
            # Read YUV frame
            Y_pq,U_pq,V_pq = yuv_read(filename,framenum,height,width,bit_depth)
            YUV = np.stack((Y_pq,U_pq,V_pq),axis=2)
            YUV = cv2.resize(YUV, (full_size[1],full_size[0]),\
                             interpolation = cv2.INTER_AREA)
            Y_pq = YUV[:,:,0]
            if(bit_depth==10):
                YUV_norm = YUV.astype(np.float32)/1023.0
                C = 4
            elif(bit_depth==8):
                YUV_norm = YUV.astype(np.float32)/255.0
                C = 1
            elif(bit_depth==12):
                YUV_norm = YUV.astype(np.float32)/4095.0
                C = 16
            # downsample YUV
            YUV_down_norm = cv2.resize(YUV_norm,(dsize[1],dsize[0]),interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
            break

            
        # convert YUV to RGB
        if(color_space=='BT2020'):
            rgb = colour_utils.YCbCr_to_RGB(YUV_norm,K = [0.2627,0.0593])
            rgb_down = colour_utils.YCbCr_to_RGB(YUV_down_norm,K = [0.2627,0.0593])
        elif(color_space=='BT709'):
            rgb = colour_utils.YCbCr_to_RGB(YUV_norm,K = [0.2126,0.0722])
            rgb_down = colour_utils.YCbCr_to_RGB(YUV_down_norm,K = [0.2126,0.0722])
            
        
        
        # NSS features for Y
        Y_pq = Y_pq.astype(np.float32)
        Y_down_pq = cv2.resize(Y_pq,(dsize[1],dsize[0]),interpolation=cv2.INTER_AREA)
        Y_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(Y_pq,C)
        dY_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(Y_down_pq,C)
        
        brisque_fullscale = ChipQA.save_stats.extract_subband_feats(Y_mscn)
        brisque_halfscale = ChipQA.save_stats.extract_subband_feats(dY_mscn)
        brisque = np.concatenate((brisque_fullscale,brisque_halfscale),axis=0)
        
        
        # NSS features for nonlinear transformed Y
        Y_pq_nl = np.squeeze(Y_compute_lnl(Y_pq))
        Y_down_pq_nl = np.squeeze(Y_compute_lnl(Y_down_pq))
        Y_mscn_pq_nl,_,_ = ChipQA.save_stats.compute_image_mscn_transform(Y_pq_nl,C=0.001)
        dY_mscn_pq_nl,_,_ = ChipQA.save_stats.compute_image_mscn_transform(Y_down_pq_nl,C=0.001)

        brisque_nl_fullscale = ChipQA.save_stats.extract_subband_feats(Y_mscn_pq_nl)
        brisque_nl_halfscale = ChipQA.save_stats.extract_subband_feats(dY_mscn_pq_nl)
        brisque_nl = np.concatenate((brisque_nl_fullscale,brisque_nl_halfscale),axis=0)


        # compute gradient magnitude
        gradient_x = cv2.Sobel(Y_pq,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        
        gradient_x_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(Y_down_pq,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    


        gradY_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(gradient_mag,C)
        dgradY_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(gradient_mag_down,C)
        
        # store gradient mscns in buffer
        grad_img_buffer[i,:,:] =gradY_mscn 
        graddown_img_buffer[i,:,:]=dgradY_mscn 
        i=i+1

        # compute nonlinearity for RGB
        rgb_nl = Y_compute_lnl(rgb)
        rgb_nl_down = Y_compute_lnl(rgb_down)
        rgb_features = np.zeros((3,36))
        rgb_features_nl = np.zeros((3,36))

        # compute NSS features for RGB
        for chroma_index in range(3):
            rgb_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(rgb[:,:,chroma_index],C=0.001)
            rgb_mscn_down,_,_ = ChipQA.save_stats.compute_image_mscn_transform(rgb_down[:,:,chroma_index],C=0.001)
            rgb_fullscale = ChipQA.save_stats.extract_subband_feats(rgb_mscn)
            rgb_halfscale = ChipQA.save_stats.extract_subband_feats(rgb_mscn_down)
            rgb_features[chroma_index,:] = np.concatenate((rgb_fullscale,rgb_halfscale),axis=0) 

            rgb_nl_mscn,_,_ = ChipQA.save_stats.compute_image_mscn_transform(rgb_nl[:,:,chroma_index],C=0.001)
            rgb_nl_mscn_down,_,_ = ChipQA.save_stats.compute_image_mscn_transform(rgb_nl_down[:,:,chroma_index],C=0.001)
            rgb_fullscale_nl = ChipQA.save_stats.extract_subband_feats(rgb_nl_mscn)
            rgb_halfscale_nl = ChipQA.save_stats.extract_subband_feats(rgb_nl_mscn_down)
            rgb_features_nl[chroma_index,:] = np.concatenate((rgb_fullscale_nl,rgb_halfscale_nl),axis=0) 

        feats = np.concatenate((brisque,brisque_nl,rgb_features.flatten(),rgb_features_nl.flatten()),0)

        feat_sd_list.append(feats)
        spatavg_list.append(feats)

        


        # compute ST Gradient chips and rolling standard deviation
        if (i>=time_length): 

            # temporal filtering
            grad3d_mscn = spatiotemporal_mscn(grad_img_buffer,avg_window)
            graddown3d_mscn = spatiotemporal_mscn(graddown_img_buffer,avg_window)

            # compute rolling standard deviation 
            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []

            # ST chips
            sts_grad = find_kurtosis_sts(grad3d_mscn,cy,cx,rst,rct,theta)
            dsts_grad = find_kurtosis_sts(graddown3d_mscn,dcy,dcx,rst,rct,theta)
            secondord_grad_feats = np.average([ChipQA.save_stats.extract_secondord_feats(chip)\
                 for chip in sts_grad],axis=0)
            secondord_dgrad_feats = np.average([ChipQA.save_stats.extract_secondord_feats(chip)\
                for chip in dsts_grad],axis=0)
            firstord_grad_feats = ChipQA.save_stats.estimateggdparam(np.asarray(sts_grad).flatten())
            firstord_dgrad_feats = ChipQA.save_stats.estimateggdparam(np.asarray(dsts_grad).flatten())
            grad_feats = np.concatenate((firstord_grad_feats,secondord_grad_feats),axis=0)
            dgrad_feats = np.concatenate((firstord_dgrad_feats,secondord_dgrad_feats),axis=0)


            allst_feats = np.concatenate((grad_feats,dgrad_feats),axis=0)
            X_list.append(allst_feats)


            # refresh buffer
            grad_img_buffer = np.zeros((time_length,full_size[0],full_size[1]))
            graddown_img_buffer =np.zeros((time_length,dsize[0],dsize[1]))
            i=0

    # average features and save to file
    X1 = np.average(spatavg_list,axis=0)
    X2 = np.average(sd_list,axis=0)
    X3 = np.average(X_list,axis=0)
    X = np.concatenate((X1,X2,X3),axis=0)
    train_dict = {"features":X}
    joblib.dump(train_dict,filename_out)
    return




def main():
    args = parser.parse_args()
    vid_stream = open(args.input_file,'r')
    vid_stream.seek(0, os.SEEK_END)
    vid_filesize = vid_stream.tell()

    if(args.bit_depth==10 or args.bit_depth==12):
        multiplier = 3
    elif(args.bit_depth==8):
        multiplier=1.5
    vid_T = int(vid_filesize/(args.height*args.width*multiplier))
    hdrchipqa_fromvid(args.input_file,args.results_file,args.width,args.height,vid_T,args.bit_depth,\
        args.color_space,args.time_length,args.step)


if __name__ == '__main__':
    main()
    

