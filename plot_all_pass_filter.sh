#!/bin/bash
#SBATCH --job-name=plot_all_pass_filter
#SBATCH --output=sbatch_outputs/plot_all_pass_filter_%j.out
#SBATCH --error=sbatch_outputs/plot_all_pass_filter_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

module purge
module load conda
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh

conda activate plasticityvenv

python -u plot_all_pass_filter.py --semifine /project2/dsong_945/BenR/prescreening/FilterFiles/Jan2010-Nonstationarity_Learning_DUPLICATE/1089/semiFine --base /project2/dsong_945/BenR/prescreening/data/Jan2010-Nonstationarity_Learning_DUPLICATE --out /project2/dsong_945/BenR/prescreening/1089_good_pairs_ccg.png

conda deactivate