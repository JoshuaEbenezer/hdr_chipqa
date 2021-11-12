import numpy as np
import scipy.ndimage

Y = 100*np.random.rand(8,8)
print(Y.shape)
avg_luminance = scipy.ndimage.gaussian_filter(Y,sigma=7.0/6.0,mode='reflect')
print(avg_luminance.shape)
print(avg_luminance)
print(Y)
