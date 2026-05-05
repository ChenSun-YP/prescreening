#!/bin/bash
#SBATCH --job-name=preprocess_test
#SBATCH --output=sbatch_outputs/preprocess_test_%j.out
#SBATCH --error=sbatch_outputs/preprocess_test_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

module purge
module load conda
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh

conda activate plasticityvenv

python -u plot_tsv_correlograms.py /project2/dsong_945/BenR/prescreening/Control_1089_20_pair_20260504.tsv --out /project2/dsong_945/BenR/prescreening/Control_1089_20_pair_20260504.png

conda deactivate