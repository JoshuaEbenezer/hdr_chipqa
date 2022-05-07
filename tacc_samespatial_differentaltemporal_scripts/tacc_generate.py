counter = 0

quarter_list = ['first_quarter','second_quarter','third_quarter','fourth_quarter']

param_list = [5,11,21,31]
for data_counter in range(4):
    quarter = quarter_list[data_counter]
    for p1 in param_list:

        f = open(f"./samespace_difftemp_ythfr_scripts/job_{counter}.script", "w")

        string = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -o /work/08176/jebeneze/ls6/code/variable_length_chipqa/eval/samespace_difftemp_ythfr_eval{counter}.log
#SBATCH -J /work/08176/jebeneze/ls6/code/variable_length_chipqa/eval/samespace_difftemp_ythfr_eval{counter}
#SBATCH -p normal
#SBATCH -t 03:15:00
#SBATCH --mail-user=joshuaebenezer@utexas.edu
#SBATCH --mail-type=all
source env/bin/activate

conda activate hdr_chipqa

module load python3

ibrun -n 10  python3 tacc_obtain.py --input_folder /scratch/08176/jebeneze/YTHFR/{quarter}  --results_folder /work/08176/jebeneze/ls6/code/variable_length_chipqa/ythfr_samespace_difftemp_chips  --database YTHFR --step {p1}
            """
        f.write(string)
        f.close()
        counter = counter+1
