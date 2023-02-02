from os.path import dirname
from os.path import join

import numpy as np
import scipy.fftpack
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.stats

import cv2
from .save_stats import aggd_features,paired_product,compute_image_mscn_transform


def _extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0]), np.array([
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, br3,  # (D1)
            alpha4, N4, bl4, br4,  # (D2)
    ])

def extract_on_patches(img, blocksizerow, blocksizecol):
    h, w = img.shape
    blocksizerow = np.int(blocksizerow)
    blocksizecol = np.int(blocksizecol)
    patches = []
    for j in range(0, np.int(h-blocksizerow+1), np.int(blocksizerow)):
        for i in range(0, np.int(w-blocksizecol+1), np.int(blocksizecol)):
            patch = img[j:j+blocksizerow, i:i+blocksizecol]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        mscn_features, pp_features = _extract_subband_feats(p)
        patch_features.append(np.hstack((mscn_features, pp_features)))
    patch_features = np.array(patch_features)

    return patch_features

def computequality(img, blocksizerow, blocksizecol, mu_prisparam, cov_prisparam):
    # img = img[:, :, 0]
    h, w = img.shape

    if (h < blocksizerow) or (w < blocksizecol):
        print("Input frame is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % blocksizerow)
    woffset = (w % blocksizecol)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)

    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100) 

    img2 = cv2.resize(img, (height,width), interpolation=cv2.INTER_CUBIC)

    mscn1, var, mu = compute_image_mscn_transform(img, extend_mode='nearest')
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2, extend_mode='nearest')
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, blocksizerow, blocksizecol)
    feats_lvl2 = extract_on_patches(mscn2, blocksizerow/2, blocksizecol/2)

    # stack the scale features
    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    mu_distparam = np.mean(feats, axis=0)
    cov_distparam = np.cov(feats.T)

    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam)/2)

    xd = mu_prisparam - mu_distparam 
    quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]

    return np.hstack((mu_distparam, [quality]))


def compute_niqe_features(frames):
    blocksizerow = 96
    blocksizecol = 96

    M, N = frames.shape
    assert ((M >= blocksizerow*2) & (N >= blocksizecol*2)), "Video too small for NIQE extraction"

    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path, 'frames_modelparameters.mat'))
    mu_prisparam = params['mu_prisparam']
    cov_prisparam = params['cov_prisparam']

    # niqe_features = np.zeros((frames.shape[0]-10, 37))
    # idx = 0
    # for i in range(5, frames.shape[0]-5):
    #   niqe_features[idx] = computequality(frames[i], blocksizerow, blocksizecol, mu_prisparam, cov_prisparam)
    #   idx += 1

    niqe_features = computequality(frames, blocksizerow, blocksizecol, mu_prisparam, cov_prisparam)
    return niqe_features

