import numpy as np
from .vif import vif_spatial
from .partial_dwpt import partial_waveletpacketdec2


def vif_dwt(img_ref, img_dist, wavelet='db2', levels=1, k=3, stride=None, sigma_nsq=5, beta=0.94):
    if stride is None:
        stride = k
    pyr_ref = partial_waveletpacketdec2(img_ref, wavelet=wavelet, level=levels)
    pyr_dist = partial_waveletpacketdec2(img_dist, wavelet=wavelet, level=levels)

    approx_ref = pyr_ref[0]
    approx_dist = pyr_dist[0]
    vif_approx = vif_spatial(approx_ref, approx_dist, k=k, sigma_nsq=sigma_nsq, stride=stride)

    subband_factors = [0.45, 0.45, 0.10]
    edge_ref = np.zeros(pyr_ref[1][0].shape)  # Infer size from a subband edge map
    edge_dist = np.zeros(pyr_dist[1][0].shape)  # Infer size from a subband edge map

    for i in range(levels):
        level_edge_ref = np.zeros_like(edge_ref)
        level_edge_dist = np.zeros_like(edge_dist)

        for subband, subband_edge_map in enumerate(pyr_ref[i+1]):
            level_edge_ref += subband_factors[subband] * subband_edge_map**2
        edge_ref += np.sqrt(level_edge_ref)

        for subband, subband_edge_map in enumerate(pyr_dist[i+1]):
            level_edge_dist += subband_factors[subband] * subband_edge_map**2
        edge_dist += np.sqrt(level_edge_dist)

    vif_edge = vif_spatial(edge_ref, edge_dist, k=k, sigma_nsq=sigma_nsq, stride=stride)

    return vif_approx * beta + vif_edge * (1 - beta)
