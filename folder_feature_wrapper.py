from joblib import load,Parallel,delayed
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Generate HDR BRISQUE features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')

args = parser.parse_args()


orig_files = glob.glob(os.path.join(args.input_folder,'*.yuv'))
files = []
for vname in orig_files:
    results_folder = args.results_folder
    name = os.path.basename(vname)
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)==True):
        continue    
    files.append(vname)


cmd_list= [] 
for i in range(len(files)):
    vname = files[i]
    results_folder = args.results_folder
    name = os.path.basename(vname)
    os.makedirs(results_folder,exist_ok=True)
    results_file = os.path.join(results_folder,os.path.splitext(os.path.basename(vname))[0]+'.z')
    cmd = "python3 hdr_chipqa.py --input_file {vname} --results_file {results_file} --bit_depth 10 --color_space BT709 --width 3840 --height 2160".format(vname=vname,results_file=results_file)
    cmd_list.append(cmd)

def call_cmd(cmd):
    os.system(cmd)



#for cmd in cmd_list:
#    call_cmd(cmd)
#
Parallel(n_jobs=20)(delayed(call_cmd)(cmd) for cmd in cmd_list)
