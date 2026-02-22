#!/bin/bash
#SBATCH --job-name=preprocess_test
#SBATCH --output=sbatch_outputs/preprocess_test_%j.out
#SBATCH --error=sbatch_outputs/preprocess_test_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

module purge
module load conda
# Ensure Conda is initialized
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh

conda activate plasticityvenv

python -u preprpoecessing_pipeline_siso_function.py


conda deactivate