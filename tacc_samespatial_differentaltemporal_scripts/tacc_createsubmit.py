import os
import glob

num_scripts = len(glob.glob('./samespace_difftemp_ythfr_scripts/*'))
f = open("ythfr_submit.sh", "a")
for num in range(num_scripts):

    f.write("sbatch samespace_difftemp_ythfr_scripts/job_{num}.script\n".format(num=num))
f.close()
