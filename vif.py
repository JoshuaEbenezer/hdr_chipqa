from skimage.util.shape import view_as_blocks
import csv
from skimage import filters
import matplotlib.pyplot as plt
from utils.vif.vif_utils import vif
from joblib import dump,Parallel,delayed
from colour.models import eotf_PQ_BT2100
from scipy.stats import gmean
import time
from scipy.ndimage import gaussian_filter
from utils.hdr_utils import hdr_yuv_read
from utils.csf_utils import csf_barten_frequency,csf_filter_block,blockwise_csf,windows_csf
import numpy as np
import glob
import pandas as pd
import os


csv_file = '../../yuv_rw_info.csv'
csv_df = pd.read_csv(csv_file)
files = csv_df["yuv"]
ref_files = glob.glob('../../../hdr_yuv/reference*')
fps = csv_df["fps"]
framenos_list = csv_df["framenos"]
ws =csv_df["w"]
hs = csv_df["h"]







def vif_refall_wrapper(ind,files):
    ref_f = files[ind]
    content = os.path.basename(ref_f).split('_')[1]
    print(content)
    dis_filenames = glob.glob("../../../hdr_yuv/4k*"+content)
    print(dis_filenames)
    Parallel(n_jobs=-1,verbose=1)(delayed(vif_video_wrapper)(ref_f,dis_f) for dis_f in dis_filenames)

def vif_video_wrapper(ref_f,dis_f):
    basename = os.path.basename(dis_f)
    dis_index = csv_df.index[csv_df['yuv'] == basename].tolist()[0]
    h =hs[dis_index]
    w = ws[dis_index]
    framenos = framenos_list[dis_index]
    vif_image_wrapper(ref_f,dis_f,framenos,h,w)

def vif_image_wrapper(ref_f,dis_f,framenos,h,w,adaptation='bilateral',use_adaptive_csf=True,use_non_overlapping_blocks=True,use_views=False):
    ref_file_object = open(ref_f)
    dis_file_object = open(dis_f)
    randlist = np.arange(framenos) # np.random.randint(0,framenos,10)

    score_df = pd.DataFrame([])
    dis_name = os.path.splitext(os.path.basename(dis_f))[0]
    output_csv = os.path.join('./features/local_csf_nonoverlapping_45x45_Xfromh_bilateral_adaptation_vif_features',dis_name+'.csv')
    if(os.path.exists(output_csv)==True):
        return
    with open(output_csv,'a') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['framenum','vif','nums','denoms'])
        for framenum in range(framenos):
            ref_y_pq,_,_ = hdr_yuv_read(ref_file_object,framenum,h,w)
            dis_y_pq,_,_ = hdr_yuv_read(dis_file_object,framenum,h,w)

            if(use_adaptive_csf==True):
                # apply CSF here
                if(use_non_overlapping_blocks==True): # apply CSF on non-overlapping blocks of the image
                    csf_filtered_ref_y_pq = blockwise_csf(ref_y_pq,adaptation=adaptation)
                    csf_filtered_dis_y_pq = blockwise_csf(dis_y_pq,adaptation=adaptation)
                else: # sliding window; returns filtered value at center of each sliding window
                    csf_filtered_ref_y_pq = windows_csf(ref_y_pq,use_views=use_views)
                    csf_filtered_dis_y_pq = windows_csf(dis_y_pq,use_views=use_views)

                # standard VIF but without CSF
                vif_val = vif(csf_filtered_ref_y_pq,csf_filtered_dis_y_pq,use_csf=False)
            else:
                # standard VIF 
                vif_val = vif(csf_filtered_ref_y_pq,csf_filtered_dis_y_pq,use_csf=True)
            # standard VIF
            row = [framenum,vif_val[0],vif_val[1],vif_val[2]]
            writer.writerow(row)


Parallel(n_jobs=5,verbose=1)(delayed(vif_refall_wrapper)(i,ref_files) for i in range(len(ref_files)))
#for i in range(len(ref_files)):
#    vif_refall_wrapper(i,ref_files)
