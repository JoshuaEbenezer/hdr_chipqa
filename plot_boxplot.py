import numpy as np
from joblib import Parallel,delayed,load,dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load
from scipy.stats import pearsonr,spearmanr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import glob
import os
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


scores_df = pd.read_csv('/home/josh-admin/hdr/fall21_score_analysis/sureal_dark_mos_and_dmos.csv')
video_names = scores_df['video']
scores = scores_df['dark_mos']
scores_df['content']=[ i.split('_')[2] for i in scores_df['video'] ]
print(len(scores_df['content'].unique()))
srocc_list = []
test_zips = []

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

test_zip_files = glob.glob('./preds/*.z')
X = []
tzf_names = []
for tzf in test_zip_files:
    test_zips = load(tzf)
    tzf_names.append(os.path.splitext(os.path.basename(tzf))[0].split('_')[0].upper())

    scores = []
    names =[]
    preds =[]
    srocc_list = []
    for v in test_zips:
        current_scores = []
        current_preds  = []
        for l in v:
            names.append(l[0])
            scores.append(l[1])
            preds.append(l[2])

            current_scores.append(l[1])
            current_preds.append(l[2])

        preds_srocc = spearmanr(current_preds,current_scores)
        srocc_list.append(preds_srocc[0])
    X.append(srocc_list)
print(tzf_names)
X = np.asarray(X)
X = pd.DataFrame(X.T,columns=tzf_names)
    



import matplotlib
matplotlib.rcParams.update({'font.size':15})

plt.figure(figsize=(12, 6), dpi=80)

plt.clf()

meds = X.median()
print(meds)
meds.sort_values(ascending=False, inplace=True)
X = X[meds.index]
X.boxplot()
print(len(tzf_names))
#plt.xticks(np.arange(len(tzf_names))+1, tzf_names)


plt.ylabel('SRCC')
plt.savefig('./images/boxplot.png')

