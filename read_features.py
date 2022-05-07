import numpy as np
import os
import glob
from joblib import load

files = glob.glob('./10_20/*')
for f in files:
    X = load(f)
    print(X['features'].shape)

