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


def trainval_split(trainval_content,r):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []

    feature_folder= './features/fall21_hdr_brisque_nl4'
    feature_folder2= '../hdr_brisque/features/brisque_pq_upscaled_features'
    feature_folder3= './features/fall21_hdr_full_hdrchipqa'

    train_names = []
    val_names = [] 
    for i,vid in enumerate(video_names):
#        if("Jockey" in vid or "Football" in vid):
#            continue
#        else:
#        featfile_name = vid +'.z'
        featfile_name = vid+'_upscaled.z'
        score = scores[i]
        feat_file = load(os.path.join(feature_folder,featfile_name))
        feat_file2 = load(os.path.join(feature_folder2,featfile_name))
        feat_file3 = load(os.path.join(feature_folder3,featfile_name))
            
        feature1 = np.asarray(feat_file['features'],dtype=np.float32)
        feature2 = np.asarray(feat_file2['features'],dtype=np.float32)
        feature3 = np.asarray(feat_file3['features'],dtype=np.float32)

#        feature = np.concatenate((feature1,feature3[0:36],feature3[168:],feature3[72:84]),axis=0)
        feature = feature3[0:36]
#        print(feature.shape)
        feature = np.nan_to_num(feature)
#        if(np.isnan(feature).any()):
#            print(vid)
        if(scores_df.loc[i]['content'] in train):
            train_features.append(feature)
            train_scores.append(score)
            train_indices.append(i)
            train_names.append(scores_df.loc[i]['video'])
            
        elif(scores_df.loc[i]['content'] in val):
            val_features.append(feature)
            val_scores.append(score)
            val_names.append(scores_df.loc[i]['video'])
#    print('Train set')
#    print(len(train_names))
#    print('Validation set')
#    print(len(val_names))
    return np.asarray(train_features),train_scores,np.asarray(val_features),val_scores,train,val_names

def single_split(trainval_content,cv_index,C):

    train_features,train_scores,val_features,val_scores,_,_ = trainval_split(trainval_content,cv_index)
    clf = SVR(kernel='linear',C=C)
#    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
#    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(val_features)
    clf.fit(X_train,train_scores)
    return clf.score(X_test,val_scores)
def grid_search(C_list,trainval_content):
    best_score = -100
    best_C = C_list[0]
    for C in C_list:
        cv_score = Parallel(n_jobs=-1)(delayed(single_split)(trainval_content,cv_index,C) for cv_index in range(5))
        avg_cv_score = np.average(cv_score)
        if(avg_cv_score>best_score):
            best_score = avg_cv_score
            best_C = C
    return best_C

def train_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    print(test_names)
    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=trainval_content)
#    scaler = MinMaxScaler(feature_range=(-1,1))  
    scaler = StandardScaler()
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    best_svr =SVR(kernel='linear',C=best_C) 
    best_svr.fit(X_train,train_scores)
    preds = best_svr.predict(X_test)


    test_zip = list(zip(test_names,test_scores,preds))
    return test_zip
#    test_zips.append(test_zip)
#
#
#    srocc_test = spearmanr(preds,test_scores)
#    print(srocc_test)
#    srocc_list.append(srocc_test[0])

test_zips = Parallel(n_jobs=-1)(delayed(train_test)(r) for r in range(100))
dump(test_zips,'./preds/brisque_preds.z')
test_zips = load('./preds/brisque_preds.z')
print(test_zips)
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
x = np.arange(np.amin(npreds),np.amax(npreds))
y = b0 * (0.5 - 1.0/(1 + np.exp(b1*(x - b2))) + b3 * x+ b4)

import matplotlib
matplotlib.rcParams.update({'font.size':15})
plt.figure()
plt.clf()
plt.scatter(npreds,nscores,label='predictions')
plt.plot(x,y,color='#ff7f0e',linewidth=3,label='fit')
plt.ylabel('MOS')
plt.xlabel('Prediction')
plt.savefig('./images/brisque_predictions.png',bbox_inches='tight')

