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


def single_run_svr(r):
    train,test = train_test_split(scores_df['content'].unique(),test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    test_features = []
    train_scores = []
    test_scores = []
    train_vids = []
    test_vids = []
    feature_folder= './features/fall21_hdr_brisque_nl4'
    feature_folder2= './features/fall21_hdr_full_hdrchipqa'

    for i,vid in enumerate(video_names):
        featfile_name = vid+'_upscaled.z'
        score = scores[i]

        feat_file = load(os.path.join(feature_folder,featfile_name))
        feat_file2 = load(os.path.join(feature_folder2,featfile_name))
            
        feature1 = np.asarray(feat_file['features'],dtype=np.float32)
        feature2 = np.asarray(feat_file2['features'],dtype=np.float32)

        feature = np.concatenate((feature1,feature2[0:36],feature2[168:],feature2[72:84]),axis=0)
#         feature1 = np.asarray(load(os.path.join(feature_folder,featfile_name))['features'],dtype=np.float32)
#         feature = feature1
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
    #naninds =np.argwhere(np.isnan(train_features))
    #nanshape = naninds.shape
#    for nanind in range(nanshape[0]):
#        actind = train_indices[nanind]
#        print(scores_df.loc[actind]['video'])
#     scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()

    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    grid_svr = GridSearchCV(SVR(kernel='linear'),param_grid = {"C":np.logspace(-7,2,10,base=2)},cv=5)
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
dump(test_zips,'./preds/hdrchipqa_feats.z')
test_zips = load('./preds/hdrchipqa_feats.z')
def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

scores = []
names =[]
preds =[]
# print(np.median(srocc_list))
for v in test_zips:
#     print(v)
    for l in v:
        names.append(l[0])
        scores.append(l[1])
        preds.append(l[2])
# print(names)
# print(scores)
nscores= []
npreds = []
nset = set(names)
print(len(names))
print(len(nset))
# print(nset)
for n in nset:
    indices = find(names,n)
    nscores.append(np.mean([scores[i] for i in indices]))
    npreds.append(np.mean([preds[i] for i in indices]))
# print(nscores,npreds)

print(len(nscores),len(npreds))

def fit(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

#     try:
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)
    preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
# #     except:
#         preds_fitted = all_preds

    return preds_fitted,b0, b1, b2, b3, b4

preds_fitted,b0, b1, b2, b3, b4 = fit(npreds,nscores)
x = np.arange(np.amin(nscores),np.amax(nscores))
y = b0 * (0.5 - 1.0/(1 + np.exp(b1*(x - b2))) + b3 * x+ b4)

import matplotlib
matplotlib.rcParams.update({'font.size':15})
plt.figure()
plt.clf()
plt.scatter(nscores,npreds,label='predictions')
plt.plot(x,y,color='#ff7f0e',linewidth=3,label='fit')
plt.xlabel('MOS')
plt.ylabel('Prediction')
plt.savefig('./images/hdrchipqa_predictions.png')

