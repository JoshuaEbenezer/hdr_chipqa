counter = 0

quarter_list = ['first_quarter','second_quarter','third_quarter','fourth_quarter']

param_list = [[5,30],[5,10],[10,20],[20,10],[30,5],[30,10]]
for data_counter in range(4):
    quarter = quarter_list[data_counter]
    for param_counter in param_list:
        p1 = param_counter[0]
        p2 = param_counter[1]

        f = open(f"./etri_scripts/job_{counter}.script", "w")

        string = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -o /work/08176/jebeneze/ls6/code/variable_length_chipqa/eval/etri_eval{counter}.log
#SBATCH -J /work/08176/jebeneze/ls6/code/variable_length_chipqa/eval/etri_eval{counter}
#SBATCH -p normal
#SBATCH -t 03:15:00
#SBATCH --mail-user=joshuaebenezer@utexas.edu
#SBATCH --mail-type=all
source env/bin/activate

conda activate hdr_chipqa

module load python3

ibrun -n 10  python3 tacc_obtain.py --input_folder /scratch/08176/jebeneze/ETRI/ETRI_LIVE_Database/dist_vids/{quarter}  --results_folder /work/08176/jebeneze/ls6/code/variable_length_chipqa/etri_multiple_length_chips --time_length {p1} --step {p2} --database ETRI
            """
        f.write(string)
        f.close()
        counter = counter+1
