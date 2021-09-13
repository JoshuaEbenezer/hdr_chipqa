import numpy as np


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def dnt(x, k, stride=1):
    kh = kw = k

    k_norm = k**2

    x_pad = np.pad(x, int((kh - stride)/2), mode='reflect')
    int_1_x = integral_image(x_pad)
    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride]) / k_norm
    x_pad_cent = np.pad(x - mu_x, int((kh - stride)/2), mode='reflect')
    int_2_x = integral_image(x_pad_cent**2)
    var_x = np.clip((int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride]) / k_norm, 0, None)
    std_x = np.sqrt(var_x)

    return (x - mu_x) / (std_x + 1e-4)


def moments(x, y, k, stride, padding=None):
    kh = kw = k

    k_norm = k**2

    if padding is None:
        x_pad = x
        y_pad = y
    else:
        x_pad = np.pad(x, int((kh - stride)/2), mode=padding)
        y_pad = np.pad(y, int((kh - stride)/2), mode=padding)

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride]) / k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride]) / k_norm

    int_2_x = integral_image(x_pad**2)
    int_2_y = integral_image(y_pad**2)

    int_xy = integral_image(x_pad*y_pad)

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride]) / k_norm - mu_x**2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride]) / k_norm - mu_y**2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride]) / k_norm - mu_x*mu_y

    # Correcting negative values of variance.
    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    # If either variance was negative, it has been set to zero.
    # So, the correponding covariance should also be zero.
    cov_xy[mask_x + mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def ssim(img_ref, img_dist, k=11, max_val=1, K1=0.01, K2=0.03, no_lum=False, full=False, padding=None, stride=1):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')
    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride, padding=padding)

    C1 = (max_val*K1)**2
    C2 = (max_val*K2)**2

    if not no_lum:
        l = (2*mu_x*mu_y + C1)/(mu_x**2 + mu_y**2 + C1)
    cs = (2*cov_xy + C2)/(var_x + var_y + C2)

    ssim_map = cs
    if not no_lum:
        ssim_map *= l

    if (full):
        return (np.mean(ssim_map), ssim_map)
    else:
        return np.mean(ssim_map)


def ms_ssim(img_ref, img_dist, k=11, max_val=1, K1=0.01, K2=0.03, full=False, padding=None, stride=1):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    scores = np.ones((n_levels,))
    for i in range(n_levels-1):
        if np.min(x.shape) <= k:
            break
        scores[i] = ssim(x, y, k, max_val, K1, K2, no_lum=True, padding=padding, stride=stride)
        x = x[:(x.shape[0]//2)*2, :(x.shape[1]//2)*2]
        y = y[:(y.shape[0]//2)*2, :(y.shape[1]//2)*2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2])/4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2])/4

    if np.min(x.shape) > k:
        scores[-1] = ssim(x, y, k, max_val, K1, K2, no_lum=False, padding=padding, stride=stride)
    msssim = np.prod(np.power(scores, weights))
    if full:
        return msssim, scores
    else:
        return msssim


def vif_spatial(img_ref, img_dist, k=11, max_val=1, sigma_nsq=0.1, padding=None):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, 1)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4)
    return vif_val
