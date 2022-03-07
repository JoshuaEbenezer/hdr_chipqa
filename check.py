import numpy as np
from joblib import load
import os
import glob
import scipy.ndimage

feat_files = glob.glob(os.path.join('../hdr_colorbleed/features/livestream_rgb_C1sdrC1e-3hdr/','1Runner_1_p1*.z'))

for f in feat_files:
    print(f)
    X = load(f)
    print(X['features'][0])
