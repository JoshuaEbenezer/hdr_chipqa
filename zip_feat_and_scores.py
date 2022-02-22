import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import os
import glob
from joblib import load,dump

import argparse

parser = argparse.ArgumentParser(description='Zip ChipQA features from a folder with scores and names from a csv file into a single dictionary and save it')
parser.add_argument('input_folder',help='Folder containing input features generated by chipqa.py (format: name_of_video.z)')
parser.add_argument('csv_file',help='CSV file containing video name under column "video" and score under column "MOS"')
parser.add_argument('results_file',help='File where features and scores are stored. Must end with .z extension')

args = parser.parse_args()
dataset='apv'
names = []
score_list =[]
if(dataset=='apv'):
    csv_file =args.csv_file 
    score_csv = pd.read_csv(csv_file)

folder =args.input_folder 
folder2 = '../hdr_colorbleed/features/livestream_rgb_C1'
folder3 = '../hdr_colorbleed/features/livestream_rgb_nl'

filenames = glob.glob(os.path.join(folder,'*.z'))
filenames2 = glob.glob(os.path.join(folder2,'*.z'))
filenames3 = glob.glob(os.path.join(folder3,'*.z'))
feats = []
scores = []
ns = []
for i,file in enumerate(sorted(filenames)):
    fname = os.path.basename(file)
    print(fname)
    bname = os.path.splitext(fname)[0]
    yuv_fname = os.path.splitext(fname)[0]+'.mp4'

    if(dataset=='apv'):
         name = os.path.splitext(fname)[0]+'.yuv'
         score = score_csv[score_csv['video'] ==name].MOS.iloc[0]
    X = load(file)
    X2 = load(filenames2[i])
    X3 = load(filenames3[i])
    x1 = X['features']
    x2 = X2['features']
    x3 = X3['features']
    print(fname,filenames2[i],filenames3[i])
#    print(x2)
    print(x2.shape)
    print(x3.shape)
#    x = np.concatenate((x1[0:72],x1[120:156],x1[168:],x2,x3[0:36],x3[72:108],x3[108+36:108+72],x3[108+72:216],x3[216+36:216+72],x3[216+108:216+108+36]),0)
#    x =np.concatenate((x1[0:72],x1[120:120+36],x1[168:],x3[36:72],x3[108:108+36],x3[108+72:216],x3[216+36:216+72],x3[216+108:216+108+36],x3[216+180:216+180+36]),0)
#   x =np.concatenate((x1[0:84],x1[120:120+36],x1[168:],x2),0)
#    x =np.concatenate((x1,x2),0)
#    print(x)
    x = x2
    print(x.shape)
    y = score
#
#
#    print(x2)
#    x =np.concatenate((x,x2),0)


    feats.append(x)
    scores.append(y)
    ns.append(fname)
feats = np.asarray(feats)
print(feats.shape)
X = {'features':feats,'score':scores,'name':ns}
dump(X,args.results_file,compress=('lzma',3))
print('dump done')
print()

