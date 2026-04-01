#!/bin/bash

## this is a job submission script for the initial analysis pipeline in case thereare too many sessions to run on your local
## do run the stage 1 on local simply run : python preprpoecessing_pipeline_siso_function.py


#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB
#SBATCH --time=36:00:00
#SBATCH --partition=main
##SBATCH --gres=gpu:1  # Uncomment if GPU is needed
#SBATCH --account=dsong_945
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=csun8109@usc.edu # change to your own email
#SBATCH --output=log/log_%x-%j.out
#SBATCH --error=log/log_%x-%j.error

# Load modules and activate environment
module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load openssl

# Activate  your own virtual environment, change the path to your own virtual environment
VENV_PATH="/home1/csun8109/myenv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    # FIX: Ensure venv's Python is in PATH
    export PATH="$VENV_PATH/bin:$PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH" >&2
    exit 1
fi

# Verify activation worked
PYTHON_EXE=$(which python)
if [[ "$PYTHON_EXE" != "$VENV_PATH/bin/python" ]]; then
    echo "WARNING: Using wrong Python: $PYTHON_EXE" >&2
    echo "Expected: $VENV_PATH/bin/python" >&2
    PYTHON_EXE="$VENV_PATH/bin/python"
fi

# Debug output
echo "Using Python: $PYTHON_EXE"
$PYTHON_EXE --version
pip list | head -10

# Run the preprocessing pipeline
python preprpoecessing_pipeline_siso_function.py

