from mpi4py import MPI
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Generate HDR BRISQUE features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')
parser.add_argument('--step',type=int,choices={5,10,20,30})
parser.add_argument('--database',type=str,choices={'YTHFR','ETRI'})

args = parser.parse_args()
database = args.database
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

orig_files = glob.glob(os.path.join(args.input_folder,'*.yuv'))
files = []
for vname in orig_files:
    results_folder = args.results_folder
    name = os.path.basename(vname)
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)==True):
        continue    
    files.append(vname)


for i in range(rank, len(files), size):
    vname = files[i]
    results_folder = os.path.join(args.results_folder,str(args.step))
    name = os.path.basename(vname)
    os.makedirs(results_folder,exist_ok=True)
    content = name.split('_')[0]
    if(database=='YTHFR'):
        fourkcontent = ['1Runner','3Runners','Flips','Hurdles','LongJump']
        if(content in fourkcontent):
            multiplier = 3
            bit_depth = 10
        else:
            multiplier=1.5
            bit_depth = 8
    elif(database=='ETRI'):
        multiplier = 3
        bit_depth = 10    
    results_file = os.path.join(results_folder,os.path.splitext(os.path.basename(vname))[0]+'.z')
    cmd = "python3 /work/08176/jebeneze/ls6/code/variable_length_chipqa/same_spatial_different_temporal_chipqa.py --input_file {vname} --results_file {results_file} --bit_depth {bit_depth} --color_space BT709\
         --width 3840 --height 2160  --step {step}".format(vname=vname,results_file=results_file,\
             bit_depth=bit_depth,step=args.step)
    print(cmd)
    os.system(cmd)
