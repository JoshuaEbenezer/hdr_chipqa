import numpy as np
from scipy.ndimage import convolve1d
from .csf_utils import csf_dict, csf_frequency, csf_mannos_daly, csf_spat_filter
from pywt import wavedec2, waverec2


def filter_pyr(pyr, csf_funct):
    n_levels = len(pyr) - 1
    filt_pyr = []
    filt_pyr.append(pyr[0])  # Do not filter approx subband. TODO: Figure out how to best filter approx subband.
    for level in range(n_levels):
        filt_level = []
        for subband in range(3):
            if csf_funct.__name__ != 'ahc_weight':
                filt_level.append(pyr[level+1][subband] * csf_funct(n_levels-level-1, subband+1))  # No approximation coefficient. Only H, V, D.
            else:
                filt_level.append(pyr[level+1][subband] * csf_funct(n_levels-level-1, subband+1, n_levels))  # No approximation coefficient. Only H, V, D.
        filt_pyr.append(tuple(filt_level))
    # for i in range(1, n_levels+1):
    #     print(filt_pyr[i][0].shape, filt_pyr[i][1].shape, filt_pyr[i][2].shape)
    return filt_pyr


def filter_img(img, filter_key, wavelet=None, **kwargs):
    if filter_key in ['frequency', 'frequency_rad', 'mannos_daly']:
        d2h = 3.0
        pic_height = 1080
        f_max = np.pi*pic_height*d2h/180
        h, w = img.shape
        u_min = -(h >> 1)
        u_max = (h >> 1) + 1 if h & 1 else (h >> 1)
        v_min = -(w >> 1)
        v_max = (w >> 1) + 1 if w & 1 else (w >> 1)

        u, v = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max), indexing='ij')
        fx, fy = u*f_max/h, v*f_max/w

        if filter_key == 'frequency':
            csf_mat = csf_frequency(np.abs(fx)) * csf_frequency(np.abs(fy))  # Separable filtering
        elif filter_key == 'frequency_rad':
            f_mat = np.sqrt(fx**2 + fy**2)
            csf_mat = csf_frequency(f_mat)
        elif filter_key == 'mannos_daly':
            f_mat = np.sqrt(fx**2 + fy**2)
            theta_mat = np.arctan2(v, u)
            csf_mat = csf_mannos_daly(f_mat, theta_mat)

        img_filtered = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * csf_mat))

    elif filter_key in ['adm', 'cdf97_watson', 'ahc']:
        n_levels = 4
        pyr = wavedec2(img, wavelet, 'reflect', n_levels)
        # for i in range(1, n_levels+1):
        #     print(pyr[i][0].shape, pyr[i][1].shape, pyr[i][2].shape)
        csf_funct = csf_dict[filter_key]

        pyr_filtered = filter_pyr(pyr, csf_funct)
        img_filtered = waverec2(pyr_filtered, wavelet, 'reflect')

    elif filter_key == 'spat_filter':
        d2h = 3.0
        filt = csf_spat_filter(d2h, k=kwargs.get('k', 21))
        img_filtered = convolve1d(img, filt, axis=0)
        img_filtered = convolve1d(img_filtered, filt, axis=1)

    return np.real(img_filtered)
