counter = 0

quarter_list = ['first_quarter','second_quarter','third_quarter','fourth_quarter']

for counter in range(4):
    quarter = quarter_list[counter]

    f = open(f"./scripts/job_{counter}.script", "w")

    string = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -o eval{counter}.log
#SBATCH -J eval{counter}
#SBATCH -p normal
#SBATCH -t 02:15:00
#SBATCH --mail-user=joshuaebenezer@utexas.edu
#SBATCH --mail-type=all
source env/bin/activate

conda activate hdr_chipqa

module load python3

ibrun -n 10  python3 tacc_obtain.py --input_folder /scratch/08176/jebeneze/spring22/{quarter}  --results_folder /work/08176/jebeneze/ls6/code/production_hdr_chipqa/spring22_features/  
            """
    f.write(string)
    f.close()
