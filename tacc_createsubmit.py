import os
import glob

num_scripts = len(glob.glob('./etri_scripts/*'))
f = open("etri_submit.sh", "a")
for num in range(num_scripts):

    f.write("sbatch etri_scripts/job_{num}.script\n".format(num=num))
f.close()
