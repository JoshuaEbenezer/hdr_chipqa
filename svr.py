import numpy as np                                                                                                                                                                                                          
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import PredefinedSplit,KFold
import glob
import os
from matplotlib import pyplot as plt 
import pandas as pd
import math
import scipy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from joblib import dump, load
from scipy.stats.mstats import gmean

from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load,Parallel,delayed
from sklearn.svm import SVR
from scipy.io import savemat
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob


def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    print(np.max(all_preds),np.min(all_preds))
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

    try:
        [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                              all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)
        preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    except:
        preds_fitted = all_preds
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean(preds_fitted-all_dmos)**2)
#    print('SROCC:')
#    print(preds_srocc[0])
#    print('LCC:')
#    print(preds_lcc[0])
#    print('RMSE:')
#    print(preds_rmse)
#    print(len(all_preds),' videos were read')
    return preds_srocc[0],preds_lcc[0],preds_rmse



scores_df = pd.read_csv('/home/josh/hdr/fall21_score_analysis/fall21_mos_and_dmos_rawavg.csv')
video_names = scores_df['video']
scores = scores_df['dark_mos']
print(len(scores_df['content'].unique()))
srocc_list = []

def trainval_split(trainval_content,r):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []
#    feature_folder= "/home/ubuntu/bitstream_mode3_p1204_3/features/p1204_etri_features"

    feature_folder= './features/brisque_pq_upscaled_local_exp_delta2'
#    feature_folder= './features/brisque_pq_upscaled_global_logit1_features'
    feature_folder= './features/brisque_pq_upscaled_local_exp_delta2'
#    feature_folder= './features/brisque_pq_upscaled_global_logit1_features'
    feature_folder= './features/brisque_linear_logit_csf_mscn_features'
    feature_folder2= './features/brisque_pq_upscaled_features'
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
        #print(feat_file)
        feat_file2 = load(os.path.join(feature_folder2,featfile_name))
            
        feature1 = np.asarray(feat_file['features'],dtype=np.float32)
        feature2 = np.asarray(feat_file2['features'],dtype=np.float32)
        if(np.sum(np.isnan(feature1))):
            feature1 = np.zeros_like(feature2)

#        feature = feature1
        try:
            feat_file = load(os.path.join(feature_folder,featfile_name))
            feat_file2 = load(os.path.join(feature_folder2,featfile_name))
            score = scores[i]
        except:
            continue
            
        feature1 = np.asarray(feat_file['features'],dtype=np.float32)
        feature2 = np.asarray(feat_file2['features'],dtype=np.float32)
        feature = np.concatenate((feature1,feature2),axis=0)
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
    return np.asarray(train_features),train_scores,np.asarray(val_features),val_scores,train

def single_split(trainval_content,cv_index,C):

    train_features,train_scores,val_features,val_scores,_ = trainval_split(trainval_content,cv_index)
    clf = svm.SVR(kernel='linear',C=C)
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
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=trainval_content)
#    scaler = MinMaxScaler(feature_range=(-1,1))  
    scaler = StandardScaler()
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    best_svr =SVR(kernel='linear',C=best_C) 
    best_svr.fit(X_train,train_scores)
    preds = best_svr.predict(X_test)
    srocc,lcc,rmse = results(preds,test_scores)
    return srocc,lcc,rmse
def only_train(r):
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
#    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features)
    grid_svr = GridSearchCV(svm.SVR(kernel='linear'),param_grid = {"C":np.logspace(-7,2,10,base=2)},cv=5)
    grid_svr.fit(X_train, all_scores)
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    dump(grid_svr,"./stgreed_lbmfr_fitted_scaler.z")
    dump(scaler,"./stgreed_lbmfr_trained_svr.z")
    return

def only_test(r):
    outfolder = './chipqa_etri_preds_from_apv'
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
#    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    scaler = load('/home/ubuntu/ChipQA_files/zfiles/vbliinds_on_apv_scaler.z')
    X_train = scaler.fit_transform(all_features)
    grid_svr = load('/home/ubuntu/ChipQA_files/zfiles/vbliinds_on_apv_svr.z')
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    predfname = 'preds_'+str(r)+'.mat'
    out = {'pred':preds,'y':test_scores}
    savemat(os.path.join(outfolder,predfname),out)
    srocc_val = np.nan_to_num(srocc_test[0])
    print(srocc_val)
    return
#    best_C,best_gamma = grid_search(np.logspace(-7,2,10),np.logspace(1,10,10,base=2),trainval_content)
#
#    scaler = StandardScaler()
#    scaler.fit(train_features)
#    X_train = scaler.transform(train_features)
#    X_test = test_features #scaler.transform(test_features)
#    best_svr =SVR(gamma=best_gamma,C=best_C) 
#    best_svr.fit(X_train,train_scores)
#    preds = X_test #best_svr.predict(X_test)
#    srocc_test = spearmanr(preds,test_scores)
#    predfname = 'preds_'+str(r)+'.mat'
#    out = {'pred':preds,'y':test_scores}
#    savemat(os.path.join(outfolder,predfname),out)
#    srocc_val = np.nan_to_num(srocc_test[0])
#    print(srocc_val)
    return #srocc_val

#only_train(0)
#only_test(0)
#srocc_list = train_test(0) 
#print(srocc_list)
srocc_list = Parallel(n_jobs=-1,verbose=0)(delayed(train_test)(i) for i in range(1000))
##srocc_list = np.nan_to_num(srocc_list)
print("median srocc is")
med_srcc = np.median([s[0] for s in srocc_list])
print(med_srcc)
print("median lcc is")
med_lcc = np.median([s[1] for s in srocc_list])
print(med_lcc)
print("median rmse is")
med_rmse = np.median([s[2] for s in srocc_list])
print(med_rmse)
print("std of srocc is")
print(np.std([s[0] for s in srocc_list]))
print("std of lcc is")
print(np.std([s[1] for s in srocc_list]))
print("std of rmse is")
print(np.std([s[2] for s in srocc_list]))
print(med_srcc, med_lcc, med_rmse)
##
