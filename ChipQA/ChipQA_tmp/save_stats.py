import numpy as np
import cv2
import glob
import os
import skvideo.io
from scipy.special import gamma
from matplotlib import pyplot as plt 
import skvideo.utils
import pandas as pd
import math
from joblib import dump
import scipy
from joblib import load
from scipy.fftpack import fft, dct
import scipy.io as sio
from scipy.stats import norm,lognorm,skew,kurtosis
from skvideo.measure import videobliinds,videobliinds_features,niqe
from skimage.util.shape import view_as_windows




win = np.array(skvideo.utils.gen_gauss_window(3, 7.0/6.0))


gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)
def generate_ggd(x,alphaparam,sigma):
    betaparam = sigma*np.sqrt(gamma(1.0/alphaparam)/gamma(3.0/alphaparam))    
    y = alphaparam/(2*betaparam*gamma(1.0/alphaparam))*np.exp(-(np.abs(x)/betaparam)**alphaparam)
    return y
def stat_feats(chroma_mscn):
    alpha,sigma = estimateggdparam(chroma_mscn)
    skewness = skew(chroma_mscn.flatten())
    kurt =kurtosis(chroma_mscn.flatten())
    return alpha,sigma,skewness,kurt

def extract_secondord_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([
            alpha1, N1, lsq1**2, rsq1**2,  # (V)
            alpha2, N2, lsq2**2, rsq2**2,  # (H)
            alpha3, N3, lsq3**2, rsq3**2,  # (D1)
            alpha4, N4, lsq4**2, rsq4**2])  # (D2)
def _extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, sigma = estimateggdparam(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([
            alpha_m, sigma,
            alpha1, N1, lsq1**2, rsq1**2,  # (V)
            alpha2, N2, lsq2**2, rsq2**2,  # (H)
            alpha3, N3, lsq3**2, rsq3**2,  # (D1)
            alpha4, N4, lsq4**2, rsq4**2,  # (D2)
    ])

def estimateggdparam(vec):
    gam = np.asarray([x / 1000.0 for x in range(200, 10000, 1)])
    r_gam = (gamma(1.0/gam)*gamma(3.0/gam))/((gamma(2.0/gam))**2)
    # print(np.mean(vec))
    sigma_sq = np.mean(vec**2) #-(np.mean(vec))**2
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq/(E**2+1e-6)
    array_position =(np.abs(rho - r_gam)).argmin()
    alphaparam = gam[array_position]
    return alphaparam,sigma


def all_aggd(y):

    falpha1,fN1,fbl1,fbr1,flsq1,frsq1 = aggd_features(y.copy())
    pps1, pps2, pps3, pps4 = paired_product(y)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([
            falpha1, fN1, flsq1**2,frsq1**2,
            alpha1, N1, lsq1**2, rsq1**2,  # (V)
            alpha2, N2, lsq2**2, rsq2**2,  # (H)
            alpha3, N3, lsq3**2, rsq3**2,  # (D1)
            alpha4, N4, lsq4**2, rsq4**2,  # (D2)
    ])
def brisque(y_mscn):
    # half_scale = cv2.resize(y, dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    feats_full = _extract_subband_feats(y_mscn)
    # feats_half = _extract_subband_feats(half_scale)
    return feats_full#np.concatenate((feats_full,feats_half))

def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
      gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
      gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
      r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
      r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

# def ggd_features(imdata):
#     nr_gam = 1/prec_gammas
#     sigma_sq = np.var(imdata)
#     E = np.mean(np.abs(imdata))
#     rho = sigma_sq/E**2
#     pos = np.argmin(np.abs(nr_gam - rho));
#     return gamma_range[pos], sigma_sq

def sigma_map(image):
    im = image.astype(np.float32)
    mu = cv2.GaussianBlur(im,(7,7),7.0/6.0,7.0/6.0)
    mu_sq = mu*mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(im**2,(7,7),7.0/6.0,7.0/6.0)-mu_sq))
    return sigma
def dog(image):
    image = image.astype(np.float32)
    gauss1 = cv2.GaussianBlur(image,(7,7),7.0/6.0,7.0/6.0)
    gauss2 = cv2.GaussianBlur(image,(7,7),7.0*1.5/6.0,7.0*1.5/6.0)
    dog = gauss1-gauss2
    return dog

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)
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

def generate_aggd(x1,x2,alpha,sigma_l,sigma_r):
    beta_l = sigma_l*np.sqrt(gamma(1/alpha)/gamma(3/alpha))
    beta_r= sigma_r*np.sqrt(gamma(1/alpha)/gamma(3/alpha))
    f1 = alpha/((beta_l+beta_r)*gamma(1/alpha))*np.exp(-(-x1/beta_l)**alpha)
    f2 = alpha/((beta_l+beta_r)*gamma(1/alpha))*np.exp(-(x2/beta_r)**alpha)
    f = np.concatenate((f1,f2),axis=0)
    return f
def chroma_feats(lab):
    # lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab)
    a = lab[:,:,1]
    b = lab[:,:,2]


    chroma = np.sqrt(a**2+b**2)
    chroma_mscn,sigma_map,_ = compute_image_mscn_transform(chroma)
    sigma_mscn,_,_ =compute_image_mscn_transform(sigma_map)

    alpha,sigma,skewness,kurt= stat_feats(chroma_mscn)
    salpha,ssigma,sskewness,skurt= stat_feats(sigma_mscn)

    
    half_scale = cv2.resize(chroma.astype(np.uint8), dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)
    half_chroma_mscn,half_sigma_map,_ = compute_image_mscn_transform(half_scale)
    half_sigma_mscn,_,_ = compute_image_mscn_transform(half_sigma_map)

    halpha,hsigma,hskewness,hkurt= stat_feats(half_chroma_mscn)
    hsalpha,hssigma,hsskewness,hskurt= stat_feats(half_sigma_mscn)
    first_order_feats = np.asarray([alpha,sigma,skewness,kurt,halpha,hsigma,hskewness,hkurt,salpha,ssigma,sskewness,skurt,hsalpha,hssigma,hsskewness,hskurt]) 
    return first_order_feats 
def estimate_log_deri_ggd(image):
    log_im = np.log(image+0.5)
    log_feats = []
    shifts= [(0,1),(1,0),(1,1),(1,-1)]
    for i in range(len(shifts)):
        rolled = np.roll(log_im, shift=shifts[i],axis=(0,1))
        log_deri = log_im - rolled
        alpha,sigma = estimateggdparam(log_deri)
        log_feats.append(np.asarray([alpha,sigma]))
    D5 = log_im + np.roll(log_im,shift=(1,1),axis=(0,1))-np.roll(log_im,shift=(0,1),axis=(0,1))-np.roll(log_im,shift=(1,0),axis=(0,1))
    D6 = np.roll(log_im,shift=(-1,0),axis=(0,1))+np.roll(log_im,shift=(1,0),axis=(0,1))-np.roll(log_im,shift=(0,-1),axis=(0,1))-np.roll(log_im,shift=(0,1),axis=(0,1))
    D7 = np.roll(log_im,shift=(-1,-1),axis=(0,1))+np.roll(log_im,shift=(1,1),axis=(0,1))-np.roll(log_im,shift=(-1,1),axis=(0,1))-np.roll(log_im,shift=(1,-1),axis=(0,1))
    alpha,sigma = estimateggdparam(D6)
    log_feats.append(np.asarray([alpha,sigma]))
    alpha,sigma = estimateggdparam(D7)
    log_feats.append(np.asarray([alpha,sigma]))
    alpha,sigma = estimateggdparam(D5)
    log_feats.append(np.asarray([alpha,sigma]))
    
    log_feats = np.asarray(log_feats)
    log_feats = np.reshape(log_feats,(14,))
    return log_feats
def estimate_extralogderis(image):
    log_im = np.log(image+0.5)
    log_feats =[]
    D6 = np.roll(log_im,shift=(-1,0),axis=(0,1))+np.roll(log_im,shift=(1,0),axis=(0,1))-np.roll(log_im,shift=(0,-1),axis=(0,1))-np.roll(log_im,shift=(0,1),axis=(0,1))
    D7 = np.roll(log_im,shift=(-1,-1),axis=(0,1))+np.roll(log_im,shift=(1,1),axis=(0,1))-np.roll(log_im,shift=(-1,1),axis=(0,1))-np.roll(log_im,shift=(1,-1),axis=(0,1))
    alpha,sigma = estimateggdparam(D6)
    log_feats.append(np.asarray([alpha,sigma]))
    alpha,sigma = estimateggdparam(D7)
    log_feats.append(np.asarray([alpha,sigma]))

    log_feats = np.asarray(log_feats)
    log_feats = np.reshape(log_feats,(4,))
    return log_feats
def chroma_gradients(lab):
    # lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab)
    # a = lab[:,:,1]
    # b = lab[:,:,2]
    chroma_grad_feats = []  
    gradient_x = cv2.Sobel(lab,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(lab,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    
    return [gradient_mag[:,:,0],gradient_mag[:,:,1],gradient_mag[:,:,2]]

def chroma_gradient_feats(lab):
    gradient_mag = chroma_gradients(lab)
    for i in range(3):
        gradient_mscn,_,_ = compute_image_mscn_transform(gradient_mag[i])
        alpha,sigma = estimateggdparam(gradient_mscn)
        # log_ggd_params = estimate_log_ggd(gradient_mag[:,:,i])
        grad_sigma = strided_variance(gradient_mag[:,:,i],5)
        grad_sigma_mean = np.mean(grad_sigma.flatten())
        grad_sigma_var = np.std(grad_sigma.flatten())
        dispersion = grad_sigma_var/grad_sigma_mean
        log_sigma_params = np.asarray([alpha,sigma,dispersion,grad_sigma_mean])
        # chroma_grad_feats.append(np.asarray([sigma_alpha,sigma_var]))        
        chroma_grad_feats.append(log_sigma_params)
    chroma_grad_feats= np.asarray(chroma_grad_feats)
    chroma_grad_feats = np.reshape(chroma_grad_feats,(12,))
    return chroma_grad_feats
def colorfulness(image):
    rg = image[:,:,2]-image[:,:,1]
    yb = 0.5*(image[:,:,2]+image[:,:,1])-image[:,:,0]
    mu = np.sqrt(np.mean(rg.flatten())**2+np.mean(yb.flatten())**2)
    sigma = np.sqrt(np.std(rg.flatten())**2+np.std(yb.flatten())**2)
    c = sigma+0.3*mu
    return c

def main():
    dataset = 'vqc'
    if(dataset=='konvid'):
        folder = '/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/konvid/konvid_sts_mscn_down_videos'
        results_folder =  '/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/konvid/konvid_sts_mscn_down_features'
        csv_file = "/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/konvid/KoNViD_1k_metadata/KoNViD_1k_mos.csv"


    elif(dataset=='vqc'):
        folder = '/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/VQC/vqc_sts_medianof'
        results_folder =  '/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/VQC/vqc_sts_medianof_feats'
    elif(dataset=='vqa'):
        folder = '/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/VQA/sts_mscn_down'
