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

import random

random.seed(21)

def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

    try:
        f = lambda x, a, b, c, s : (a-b) / (1 + np.exp(-((x - c) / s))) + b
        init_val = np.array([np.max(all_dmos), np.min(all_dmos), np.mean(all_preds) , np.std(all_preds)/4])
        [[a, b, c, s], _] = curve_fit(f, all_preds, all_dmos, p0=init_val, maxfev=20000)
        preds_fitted = (a-b) / (1 + np.exp(-((all_preds - c) / s))) + b
    except:
        preds_fitted = all_preds
    preds_fitted = np.nan_to_num(preds_fitted)
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean((preds_fitted-all_dmos)**2))
    print('SROCC:')
    print(preds_srocc[0])
    print('LCC:')
    print(preds_lcc[0])
    print('RMSE:')
    print(preds_rmse)
    return np.nan_to_num(preds_srocc[0]),np.nan_to_num(preds_lcc[0]),np.nan_to_num(preds_rmse)



scores_df = pd.read_csv('./ETRI_metadata.csv')
scores_df.reset_index(drop=True, inplace=True)
print(len(scores_df))
video_names = scores_df['video']
scores = list(scores_df['mos'])
print(scores_df['content'])
print(len(scores_df['content'].unique()))
srocc_list = []

def trainval_split(trainval_content,r,feature_folder):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []

    train_names = []
    val_names = [] 

    for i,vid in enumerate(video_names):
        featfile_name = vid+'.z'
        score = scores[i]
        feature_folder1= '../../hdr_chipqa/features/etri_fullhdrchipqa'
        feature_folder2= './etri_hdrchipqa_rgbc1e-2/'
        feature_folder3= './etri_samespace_difftemp_chips/5'
        feat_file1= load(os.path.join(feature_folder1,featfile_name))
        feat_file2 = load(os.path.join(feature_folder2,featfile_name))
        feature1 = np.asarray(feat_file1['features'],dtype=np.float32)
        feature2 = np.asarray(feat_file2['features'],dtype=np.float32)
        feature = np.concatenate((feature1[0:72],feature1[84:156],feature1[168:],feature2),0)
        feat_file= load(os.path.join(feature_folder3,featfile_name))
        feature3 = np.asarray(feat_file['features'],dtype=np.float32)
        feature = np.concatenate((feature,feature3[-36:]),0)


#        exclude = np.concatenate((np.arange(72,288),np.arange(360,576)))
#        feature = np.delete(feature,exclude)
#        for folder in glob.glob('./etri_multiple_length_chips/*'):
#            feat_file = load(os.path.join(folder,featfile_name))
#            extra_feature = np.asarray(feat_file['features'],dtype=np.float32)
#            feature = np.concatenate((feature,extra_feature[-36:]),0)
        feature = np.nan_to_num(feature)
        if(scores_df.loc[i]['content'] in train):
            train_features.append(feature)
            train_scores.append(score)
            train_indices.append(i)
            train_names.append(scores_df.loc[i]['video'])
            
        elif(scores_df.loc[i]['content'] in val):
            val_features.append(feature)
            val_scores.append(score)
            val_names.append(scores_df.loc[i]['video'])
    return np.asarray(train_features),train_scores,np.asarray(val_features),val_scores,train,val_names

def single_split(trainval_content,cv_index,C,feature_folder,kernel='linear',gamma='auto'):

    train_features,train_scores,val_features,val_scores,_,_ = trainval_split(trainval_content,cv_index,feature_folder)
    if(kernel=='linear'):
        clf = svm.SVR(kernel='linear',C=C)
    elif(kernel=='rbf'):
        clf = svm.SVR(kernel='rbf',gamma=gamma,C=C)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(val_features)
    clf.fit(X_train,train_scores)
    return clf.score(X_test,val_scores)
def grid_search_linear(C_list,trainval_content,feature_folder):
    best_score = -100
    best_C = C_list[0]
    for C in C_list:
        cv_score = Parallel(n_jobs=-1)(delayed(single_split)(trainval_content,cv_index,C,feature_folder) for cv_index in range(5))
        avg_cv_score = np.average(cv_score)
        if(avg_cv_score>best_score):
            best_score = avg_cv_score
            best_C = C
    return best_C
def grid_search_rbf(gamma_list,C_list,trainval_content,feature_folder):
    best_score = -100
    best_C = C_list[0]
    best_gamma = gamma_list[0]
    for gamma in gamma_list:
        for C in C_list:
            cv_score = Parallel(n_jobs=-1)(delayed(single_split)(trainval_content,cv_index,C,feature_folder,'rbf',gamma) for cv_index in range(5))
            avg_cv_score = np.average(cv_score)
            print(avg_cv_score)
            if(avg_cv_score>best_score):
                best_score = avg_cv_score
                best_C = C
                best_gamma = gamma
    return best_C,best_gamma

def train_test(r,feature_folder):
    kernel = 'linear'
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = \
        trainval_split(scores_df['content'].unique(),r,feature_folder)
    if(kernel=='linear'):
        best_C= grid_search_linear(np.logspace(1,10,10,base=2),trainval_content,feature_folder)
        best_svr = SVR(kernel=kernel,C=best_C)
    elif(kernel=='rbf'):
        best_C,best_gamma= grid_search_rbf(np.logspace(-7,2,10),np.logspace(1,10,10,base=2),trainval_content,feature_folder)
        best_svr =SVR(gamma=best_gamma,C=best_C) 


    scaler = MinMaxScaler(feature_range=(-1,1))  
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    best_svr.fit(X_train,train_scores)
    preds = best_svr.predict(X_test)
    srocc,lcc,rmse = results(preds,test_scores)
    return srocc,lcc,rmse,test_names
def only_train(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    feature_dict = {"feature":all_features,"name":video_names,"score":all_scores}
    dump(feature_dict,'hdrchipqa_etri_features.z')
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X_train = scaler.fit_transform(all_features)
    grid_svr = GridSearchCV(svm.SVR(),param_grid = {"gamma":np.logspace(-8,1,10),"C":np.logspace(1,10,10,base=2)},cv=5)
    grid_svr.fit(X_train, all_scores)
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
#    dump(grid_svr,"./stgreed_lbmfr_fitted_scaler.z")
#    dump(scaler,"./stgreed_lbmfr_trained_svr.z")
    return

def only_test(r):
    outfolder = './chipqa_etri_preds_from_apv'
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
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
feature_folders1 = glob.glob('./etri_multiple_length_chips/*')
f = '../../hdr_chipqa/features/etri_fullhdrchipqa'
#for f in feature_folders1:
base = os.path.splitext(os.path.basename(f))[0]
out_csv = 'results/'+base+'_linear_variabletime_etri_variablechipqa_srcc_lcc_rmse_list.csv'
if(os.path.exists(out_csv)):
    print('output exists')
else:
    srocc_list = Parallel(n_jobs=-1,verbose=0)(delayed(train_test)(i,f) for i in range(100))
    srcc_csv = pd.DataFrame(srocc_list,columns=['srcc','lcc','rmse','names'])
    srcc_csv.to_csv(out_csv)
    srocc_list = np.nan_to_num(srocc_list)
    print("median srocc is")
    print(np.median([s[0] for s in srocc_list]))
    print("median lcc is")
    print(np.median([s[1] for s in srocc_list]))
    print("median rmse is")
    print(np.median([s[2] for s in srocc_list]))
    print("std of srocc is")
    print(np.std([s[0] for s in srocc_list]))
    print("std of lcc is")
    print(np.std([s[1] for s in srocc_list]))
    print("std of rmse is")
    print(np.std([s[2] for s in srocc_list]))
