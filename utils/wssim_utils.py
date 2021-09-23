import numpy as np
from pywt import wavedec2


def wssim(img_ref, img_dist, wavelet='db2'):
    n_levels = 4
    K = 1e-4

    pyr_ref = wavedec2(img_ref, wavelet, 'reflect', n_levels)
    pyr_dist = wavedec2(img_dist, wavelet, 'reflect', n_levels)

    # Flatten into lists of subbands
    pyr_ref = pyr_ref[:1] + [item for sublist in pyr_ref[1:] for item in sublist]
    pyr_dist = pyr_dist[:1] + [item for sublist in pyr_dist[1:] for item in sublist]

    n_subbands = len(pyr_ref)

    wssim_ret = 0
    for subband_ref, subband_dist in zip(pyr_ref, pyr_dist):
        wssim_ret += (2*np.abs(np.sum(subband_ref*subband_dist)) + K) / (np.sum(subband_ref**2 + subband_dist**2) + K)
    wssim_ret /= n_subbands

    return wssim_ret
