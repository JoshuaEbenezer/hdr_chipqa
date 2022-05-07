import numpy as np
import os
import glob
import pandas as pd

result_files = glob.glob('./ythfr_results/*')
max_srcc = 0

for f in result_files:
    X = pd.read_csv(f)
    srcc_list = X['srcc'].tolist()
    med_srcc = np.median(srcc_list)
    if(med_srcc>max_srcc):
        max_srcc = med_srcc
        max_f = f
    print(f,med_srcc)
print(max_f,max_srcc)
