#!/bin/sh
#SBATCH --partition=standard --time=04:00:00
#SBATCH --output=batch_output/output.txt%j
#SBATCH --error=batch_output/error.txt%j
#SBATCH -c 32
module load python
module load anaconda
python truth_bluff.py -seed 200 -n 4 -k 4 -i default &
python truth_bluff.py -seed 200 -n 4 -k 5 -i default &
python truth_bluff.py -seed 200 -n 4 -k 6 -i default