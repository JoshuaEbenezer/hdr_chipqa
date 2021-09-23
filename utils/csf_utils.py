import numpy as np
import cv2
from joblib import dump,Parallel,delayed
from skimage.util.shape import view_as_blocks,view_as_windows
from skimage.util import apply_parallel

TV_HEIGHT = 32.6
VIEWING_DIST = 1.5*TV_HEIGHT
theta = TV_HEIGHT/2160*180/np.pi*1/VIEWING_DIST
f_max = 1/theta
h_win = 45 
w_win = 45 
X =h_win/2160*1/1.5
def gen_gauss_window(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., int(l))
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)
gauss_window = gen_gauss_window(h_win,h_win/6)


def csf_barten_frequency(f,L,X=40):
    '''
    Implements simplified CSF from Peter Barten,"Formula for the contrast sensitivity of the human eye", Electronic Imaging 2004
    '''
    
    num = 5200*np.exp(-0.0016*(f**2)*((1+100/L)**0.08))
    denom1 = 1+144/X**2+0.64*f**2
    denom21 = 63/L**0.83
    denom22 = 1/(1e-4+1-np.exp(-0.02*(f**2)))
    denom2 =denom21+denom22 
    csf_freq = num/(np.sqrt(denom1*denom2))
    return csf_freq
def round_down(num,divisor):
    return num-(num%divisor)

def blockwise_csf(y,adaptation='gaussian',h_win=h_win,w_win=w_win):
    '''
    Divides frame into non-overlapping blocks and applies Barten's CSF
    '''

    max_h,max_w = round_down(y.shape[0],h_win),round_down(y.shape[1],w_win)
    y_crop = y[:max_h,:max_w]
    blocks_list = view_as_blocks(y_crop,(h_win,w_win))
    blocks =np.reshape(blocks_list,(-1,blocks_list.shape[2],blocks_list.shape[3]))
    block_csf = Parallel(n_jobs=-5,verbose=0)(delayed(csf_filter_block)(block=block,use_index=False,adaptation=adaptation,overlap=False) for block in blocks)
    block_csf = np.reshape(block_csf,(max_h,max_w))

    return block_csf

def windows_csf(y,use_views=False,adaptation='bilateral',h_win=h_win,w_win=w_win):
    '''
    Divides frame into overlapping blocks and applies Barten's CSF
    '''

 #TODO find out if using views is faster
    if(use_views): 
        max_h,max_w = round_down(y.shape[0],h_win),round_down(y.shape[1],w_win)
        y_crop = y[:max_h,:max_w]
        #window_csf = apply_parallel(csf_filter_block,y_crop,(h_win,w_win),extra_keywords={"window":True},dtype=np.float32)
        windows_list = view_as_windows(y_crop,(h_win,w_win))
        out_h,out_w = windows_list.shape[0],windows_list.shape[1]
        windows =np.reshape(windows_list,(-1,windows_list.shape[2],windows_list.shape[3]))
        window_csf = Parallel(n_jobs=-5,verbose=0)(delayed(csf_filter_block)(window,overlap=True) for window in windows)
    else:
        h_indices = np.arange(h_win//2,y.shape[0]-h_win//2)
        w_indices = np.arange(w_win//2,y.shape[1]-w_win//2)
        xx,yy = np.meshgrid(h_indices,w_indices)
        xx = xx.flatten()
        yy = yy.flatten()

        window_csf = Parallel(n_jobs=-5,verbose=0)(delayed(csf_filter_block)(y=y,adaptation=adaptation,index=pixel_index,xx_list=xx,yy_list=yy,h_win=h_win,w_win=w_win,use_index=True,overlap=True) for pixel_index in range(len(xx)))
        out_h,out_w = len(h_indices),len(w_indices)
    window_csf = np.reshape(window_csf,(out_h,out_w))
    return window_csf



def csf_filter_block(y=None,block=None,adaptation='gaussian',index=None,xx_list=None,yy_list=None,h_win=None,w_win=None,use_index=True,overlap=True,pq=True,gauss_window=gauss_window,X=X):
    if(use_index):
        xx,yy = xx_list[index],yy_list[index]
        block = y[xx-h_win//2:xx+h_win//2+1, yy-w_win//2:yy+w_win//2+1]
    block = block.astype(np.float32)
    if(adaptation=='gaussian'):
        gauss_filtered = gauss_window*block
        avg_luminance = np.sum(gauss_filtered)
    elif(adaptation=='bilateral'):
        bilateral_filtered = cv2.bilateralFilter(src=block,d=-1,sigmaColor=20,sigmaSpace=20)
        avg_luminance = np.average(bilateral_filtered)


    h,w = block.shape
    u_min = -(h >> 1)
    u_max = (h >> 1) + 1 if h & 1 else (h >> 1)
    v_min = -(w >> 1)
    v_max = (w >> 1) + 1 if w & 1 else (w >> 1)

    u, v = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max), indexing='ij')
    fx, fy = u*f_max/h, v*f_max/w

    csf_mat = csf_barten_frequency(np.abs(fx),avg_luminance,X) * csf_barten_frequency(np.abs(fy),avg_luminance,X)
    if(pq==True):
        csf_mat = csf_mat/np.max(csf_mat)

    block_csf_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(block)) * csf_mat)))
    if(overlap):
       return np.reshape(block_csf_filtered[h//2,w//2],(-1,1))  
    else:
        return block_csf_filtered


def csf_frequency(f):
    return (0.31 + 0.69*f) * np.exp(-0.29*f)


def csf_spat_filter(d2h, k=21):
    assert isinstance(k, int) and k > 0 and (k & 1), 'The length of the filter must be an odd positive integer'
    del_theta = 180 / (d2h * 1080 * np.pi)
    t = np.arange(-(k >> 1), (k >> 1) + 1) * del_theta
    assert len(t) == k, 'Filter is of the wrong size'

    a = 0.31
    b = 0.69
    c = 0.29
    f = 2*((a*c + b)*c**2 + (a*c - b)*4*np.pi**2 * t**2) / (c**2 + 4*np.pi**2 * t**2)**2   # Inverse Fourier Transform of CSF.

    return f*del_theta


# Ref "Most apparent distortion: full-reference image quality assessment and the role of strategy", E. C. Larson and D. M. Chandler
def csf_mannos_daly(f, theta):
    f_theta = f / (0.15*np.cos(theta) + 0.85)
    lamda = 0.228
    f_max = 4  # TODO: Find out how f_max is obtained from lamda
    if isinstance(f, np.ndarray):
        return np.where(f >= f_max, 2.6*(0.0192 + lamda*f_theta)*np.exp(-np.power(lamda*f_theta, 1.1)), 0.981)
    else:
        return 2.6*(0.0192 + lamda*f_theta)*np.exp(-np.power(lamda*f_theta, 1.1)) if f >= f_max else 0.981


def detection_threshold(a, k, f0, g, level_frequency):
    return a * np.power(10, k*np.log10(f0 * g / level_frequency)**2)


def csf_adm(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    '''
    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))
    orientation_factors = [1.0/0.85, 1.0, 1.0, 1/(0.85-0.15)]
    return csf_frequency(level_frequency * orientation_factors[subband])


def csf_cdf97_watson(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: A. Watson, G. Yang, et al. "Visibility of Wavelet Quantization Noise"
    '''
    # Detection threshold model parameters
    a = 0.495
    k = 0.466
    f0 = 0.401
    gs = [1.501, 1, 1, 0.534]

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    # Basis function amplitudes
    amplitudes = np.array([[0.621710, 0.672340, 0.672340, 0.727090],
                           [0.345370, 0.413170, 0.413170, 0.494280],
                           [0.180040, 0.227270, 0.227270, 0.286880],
                           [0.091401, 0.117920, 0.117920, 0.152140],
                           [0.045943, 0.059758, 0.059758, 0.077727],
                           [0.023013, 0.030018, 0.030018, 0.039156]])

    return 0.5 * amplitudes[level, subband] / detection_threshold(a, k, f0, gs[subband], level_frequency)


def csf_dwt_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 2.818
    k = 0.783
    f0 = 0.578
    gs = [1.5, 1, 1, 0.534]  # g0, i.e. for approximation subband, is not provided. Using value from Watson (ref: csf_cdf97_watson). Do not recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    # Basis function amplitudes
    # amplitudes = np.array([[0.621710, 0.672340, 0.672340, 0.727090],
    #                        [0.345370, 0.413170, 0.413170, 0.494280],
    #                        [0.180040, 0.227270, 0.227270, 0.286880],
    #                        [0.091401, 0.117920, 0.117920, 0.152140],
    #                        [0.045943, 0.059758, 0.059758, 0.077727],
    #                        [0.023013, 0.030018, 0.030018, 0.039156]])

    # return 2.0 * detection_threshold(a, k, f0, gs[subband], level_frequency) / amplitudes[level, subband]
    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_dtcwt_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 3.107
    k = 1.025
    f0 = 0.755
    gs = [1.3, 1, 1, 0.814]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_curvelet_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 1.083
    k = 0.790
    f0 = 0.509
    gs = [1, 1, 1, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_steerable_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 2.617
    k = 0.960
    f0 = 0.487
    gs = [1, 1, 1, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def ahc_weight(level, subband, n_levels, binarized=True):
    '''
    Weighting function used in Adaptive High Frequency Clipping
    If binarized is True, weight is compared to a threshold and 0-1 outputs are returned.
    Ref: K. Gu, G. Zhai, et al. "Adaptive High Frequency Clipping for Improved Image Quality Assessment"
    '''
    # Weighting function parameters
    a = 10
    k = 10
    t = 2
    d0 = 512
    gs = [1, 2, 2, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.
    thresh = 1.0

    # Distance to height ratio of the display
    d2h = 3.0

    weight = gs[subband] * np.power(k, t*(n_levels - (level+1))) / np.power(a,  d2h/d0)  # Paper says "d/d0" but I think they meant to use d2h
    if binarized:
        weight = float(weight >= thresh)

    return weight


csf_dict = {'frequency': csf_frequency,
            'spat_filter': csf_spat_filter,
            'adm': csf_adm,
            'mannos_daly': csf_mannos_daly,
            'cdf97_watson': csf_cdf97_watson,
            'dwt_hill': csf_dwt_hill,
            'dtcwt_hill': csf_dtcwt_hill,
            'curvelet_hill': csf_curvelet_hill,
            'steerable_hill': csf_steerable_hill,
            'ahc': ahc_weight}
