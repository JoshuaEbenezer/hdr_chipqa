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


scores_df = pd.read_csv('/home/josh/hdr/fall21_score_analysis/sureal_dark_mos_and_dmos.csv')
video_names = scores_df['video']
scores = scores_df['dark_mos']
scores_df['content']=[ i.split('_')[2] for i in scores_df['video'] ]
print(len(scores_df['content'].unique()))
srocc_list = []
test_zips = []

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

def fit(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

    try:
        [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                              all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)
        preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    except:
        preds_fitted = all_preds

    return preds_fitted

test_zip_files = glob.glob('./preds/*.z')
X = []
lcc_arr = []
rmse_arr = []
tzf_names = []
for tzf in test_zip_files:
    test_zips = load(tzf)
    tzf_names.append(os.path.splitext(os.path.basename(tzf))[0].split('_')[0].upper())

    scores = []
    names =[]
    preds =[]
    srocc_list = []
    lcc_list = []
    rmse_list = []
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
        preds_fitted= fit(current_preds,current_scores)
        preds_lcc = pearsonr(preds_fitted,current_scores)
        preds_rmse = np.sqrt(np.mean((preds_fitted-current_scores)**2))
        srocc_list.append(preds_srocc[0])
        lcc_list.append(preds_lcc[0])
        rmse_list.append(preds_rmse)
    X.append(srocc_list)
    lcc_arr.append(lcc_list)
    rmse_arr.append(rmse_list)
print(tzf_names)
X = np.asarray(X)
lcc_arr = np.asarray(lcc_arr)
rmse_arr = np.asarray(rmse_arr)

X = pd.DataFrame(X.T,columns=tzf_names)
lcc_arr = pd.DataFrame(lcc_arr.T,columns=tzf_names)
rmse_arr = pd.DataFrame(rmse_arr.T,columns=tzf_names)
    

import matplotlib
matplotlib.rcParams.update({'font.size':15})

plt.figure(figsize=(12, 6), dpi=80)

plt.clf()

meds = X.median()
srcc_std = X.std()
print(meds)
print(srcc_std)
lcc_meds = lcc_arr.median()
lcc_std = lcc_arr.std()
rmse_meds = rmse_arr.median()
rmse_std = rmse_arr.std()
print(lcc_meds)
print(lcc_std)
print(rmse_meds)
print(rmse_std)

meds.sort_values(ascending=False, inplace=True)
X = X[meds.index]
X.boxplot()
print(len(tzf_names))
#plt.xticks(np.arange(len(tzf_names))+1, tzf_names)


plt.ylabel('SRCC')
plt.savefig('./images/boxplot.png')

