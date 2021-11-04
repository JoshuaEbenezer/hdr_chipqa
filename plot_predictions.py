import numpy as np
from joblib import Parallel,delayed
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



scores_df = pd.read_csv('../fall21_hdr_score_analysis/fall21_mos_and_dmos_rawavg.csv')
video_names = scores_df['video']
scores = scores_df['dark_mos']

print(len(scores_df['content'].unique()))
srocc_list = []
test_zips = []

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

def single_run_svr(r):
    train,test = train_test_split(scores_df['content'].unique(),test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    test_features = []
    train_scores = []
    test_scores = []
    train_vids = []
    test_vids = []
    feature_folder= './features/fall21_hdr_chipqa_pq_upscaled_features'
    feature_folder2= './features/fall21_hdr_chipqa_global_logit_upscaled'

    for i,vid in enumerate(video_names):
        featfile_name = vid+'_upscaled.z'
        score = scores[i]
        feature1 = np.asarray(load(os.path.join(feature_folder,featfile_name))['features'],dtype=np.float32)
        feature2 = np.asarray(load(os.path.join(feature_folder2,featfile_name))['features'],dtype=np.float32)
        feature = feature1
        feature = np.nan_to_num(feature)
#        if(np.isnan(feature).any()):
#            print(vid)
        if(scores_df.loc[i]['content'] in train):
            train_features.append(feature)
            train_scores.append(score)
            train_indices.append(i)
            train_vids.append(vid)
            
        else:
            test_features.append(feature)
            test_scores.append(score)
            test_vids.append(vid)
    train_features = np.asarray(train_features)
    naninds =np.argwhere(np.isnan(train_features)) 
    nanshape = naninds.shape
#    for nanind in range(nanshape[0]):
#        actind = train_indices[nanind]
#        print(scores_df.loc[actind]['video'])
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    grid_svr = GridSearchCV(SVR(),param_grid = {"gamma":np.logspace(-8,1,10),"C":np.logspace(1,10,10,base=2)},cv=5)
    grid_svr.fit(X_train, train_scores)
    preds =grid_svr.predict(X_test)
    test_zip = list(zip(test_vids,test_scores,preds))
    return test_zip
#    test_zips.append(test_zip)
#
#
#    srocc_test = spearmanr(preds,test_scores)
#    print(srocc_test)
#    srocc_list.append(srocc_test[0])

test_zips = Parallel(n_jobs=-1)(delayed(single_run_svr)(r) for r in range(100))

scores = []
names =[]
preds =[]
print(np.median(srocc_list))
for v in test_zips:
    print(v)
    for l in v:
        names.append(l[0])
        scores.append(l[1])
        preds.append(l[2])
print(names)
print(scores)
nscores= []
npreds = []
nset = set(names)
print(len(names))
print(len(nset))
print(nset)
for n in nset:
    indices = find(names,n)
    nscores.append(np.mean([scores[i] for i in indices]))
    npreds.append(np.mean([preds[i] for i in indices]))
print(nscores,npreds)



plt.scatter(nscores,npreds)
plt.xlabel('MOS')
plt.ylabel('Prediction')
plt.savefig('./images/scatter_plots/fall21_hdr_chipqa_pq_upscaled_features.png')