#!/bin/bash
## this is a job submission script for the stationary fit pipeline in case thereare too many sessions to run on your local
## do run the stage 2 on local simply run : python glm_fit_cv_one_neuron.py --data_file <data_file> --ranking_file <ranking_file> --save_dir <save_dir> --rank_range <rank_range> --alpha_k <alpha_k> --alpha_h <alpha_h> --num_folds <num_folds> --L <L> --max_tau <max_tau>
## the data_file and ranking_file are the output of the stage 1, the save_dir is the directory to save the results, the rank_range is the range of ranks to fit, the alpha_k and alpha_h are the regularization parameters, the num_folds is the number of folds for cross-validation, the L is the length of the time series, the max_tau is the maximum lag for the correlogram

# put the dir under dsong945/project2/your_name/

# Fixed parameters 
PARTITION="main"  # Adjust based on your cluster;
SCRIPT="glm_fit_cv_one_neuron.py" 
ACCOUNT="dsong_945"
EMAIL="csun8109@usc.edu" # change to your email

# Create log directory if it doesn't exist
mkdir -p log

# Define parameter sets as arrays
# Each set contains: DATA_FILE, RANKING_FILE, START_RANK, END_RANK, STEP, L, ALPHA_K, ALPHA_H, SAVE_DIR_BASE, MAX_TAU
PARAM_SETS=(
  # Set 1

  "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_fine.csv 1 800 35 5 0.7 0.7 DNMS_data/single_pairs_analysis1150b033 100"

  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 35 4 0.7 0.7 DNMS_data/single_pairs_analysis1150b032 100"
  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 35 4 0.81 0.88 DNMS_data/single_pairs_analysis1150b032 200"
  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 35 5 0.81 0.88 DNMS_data/single_pairs_analysis1150b032 200"

  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 30 4 0.7 0.7 DNMS_data/single_pairs_analysis1150b032 100"
  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 30 4 0.7 0.7 DNMS_data/single_pairs_analysis1150b032 100"
  # "DNMS_data/selected_neurons1150b032.pkl DNMS_data/analysis/selected_neurons1150b032/pair_rankings_fine.csv 1 800 30 4 0.7 0.7 DNMS_data/single_pairs_analysis1150b032 100"
  
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_fine.csv 1 800 30 5 0.7 0.7 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b034.pkl DNMS_data/analysis/selected_neurons1150b034/pair_rankings_fine.csv 1 800 30 5 0.7 0.7 DNMS_data/single_pairs_analysis1150b034 100"
  # "DNMS_data/selected_neurons1150b035.pkl DNMS_data/analysis/selected_neurons1150b035/pair_rankings_fine.csv 1 800 30 5 0.7 0.7 DNMS_data/single_pairs_analysis1150b035 100"



  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_semifine.csv 1 30 30 8 0.7 0.7 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b034.pkl DNMS_data/analysis/selected_neurons1150b034/pair_rankings_semifine.csv 1 800 30 8 0.7 0.7 DNMS_data/single_pairs_analysis1150b034 100"
  # "DNMS_data/selected_neurons1150b035.pkl DNMS_data/analysis/selected_neurons1150b035/pair_rankings_semifine.csv 1 800 30 8 0.7 0.7 DNMS_data/single_pairs_analysis1150b035 100"

  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.6 0.6 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.5 0.5 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.4 0.4 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.4 0.4 DNMS_data/single_pairs_analysis1150b033 100"

  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.7 0.7 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.6 0.6 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.5 0.5 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.4 0.4 DNMS_data/single_pairs_analysis1150b033 100"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.4 0.4 DNMS_data/single_pairs_analysis1150b033 100"

  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.7 0.95 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.7 0.92 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.8 0.92 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 6 0.9 0.92 DNMS_data/single_pairs_analysis1150b033 500"

  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.7 0.95 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.7 0.92 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.8 0.92 DNMS_data/single_pairs_analysis1150b033 500"
  # "DNMS_data/selected_neurons1150b033.pkl DNMS_data/analysis/selected_neurons1150b033/pair_rankings_ultra-fine.csv 1 30 40 5 0.9 0.92 DNMS_data/single_pairs_analysis1150b033 500"
)

# Loop over parameter sets
for SET in "${PARAM_SETS[@]}"; do
  # Split the set into individual parameters
  read -r DATA_FILE RANKING_FILE START_RANK END_RANK STEP L ALPHA_K ALPHA_H SAVE_DIR_BASE MAX_TAU <<< "$SET"
  # Loop over rank ranges for the current set
  for ((start=$START_RANK; start<=$END_RANK; start+=$STEP)); do
    end=$((start + STEP - 1))
    if [ $end -gt $END_RANK ]; then
      end=$END_RANK
    fi
    
    # Define rank range string (e.g., "750-780")
    RANK_RANGE="${start}-${end}"
    # Replace dots with underscores in ALPHA_K and ALPHA_H to avoid dots in file names
    ALPHA_K_SAFE=${ALPHA_K//./_}
    ALPHA_H_SAFE=${ALPHA_H//./_}
    # Construct save directory by appending parameters to the base path
    SAVE_DIR="${SAVE_DIR_BASE}_${MAX_TAU}_L${L}_k${ALPHA_K_SAFE}_h${ALPHA_H_SAFE}"
    # Create a unique job name with set identifier
    JOB_NAME="glm_rank_${start}_${end}_L${L}_k${ALPHA_K_SAFE}_h${ALPHA_H_SAFE}_$(date +%Y%m%d)"
    echo "Submitting job for set with L=$L, ALPHA_K=$ALPHA_K, ALPHA_H=$ALPHA_H, MAX_TAU=$MAX_TAU, rank range: $RANK_RANGE"
    echo "Save directory: $SAVE_DIR"
    
    # Create save directory if it doesn't exist
    mkdir -p "$SAVE_DIR"
    sleep 10
    
    # Submit the job with sbatch
    sbatch --job-name="$JOB_NAME" <<EOT
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB
#SBATCH --time=36:00:00
#SBATCH --partition=$PARTITION
##SBATCH --gres=gpu:1  # Uncomment if GPU is needed
#SBATCH --account=$ACCOUNT
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$EMAIL
#SBATCH --output=log/log_%x-%j.out
#SBATCH --error=log/log_%x-%j.error

# Load modules and activate environment
module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load openssl

# 

# Activate virtual environment
VENV_PATH="/home1/csun8109/myenv"
if [ -f "\$VENV_PATH/bin/activate" ]; then
    source "\$VENV_PATH/bin/activate"
    # FIX: Ensure venv's Python is in PATH
    export PATH="\$VENV_PATH/bin:\$PATH"
else
    echo "ERROR: Virtual environment not found at \$VENV_PATH" >&2
    exit 1
fi

# Verify activation worked
PYTHON_EXE=\$(which python)
if [[ "\$PYTHON_EXE" != "\$VENV_PATH/bin/python" ]]; then
    echo "WARNING: Using wrong Python: \$PYTHON_EXE" >&2
    echo "Expected: \$VENV_PATH/bin/python" >&2
    PYTHON_EXE="\$VENV_PATH/bin/python"
fi

# Debug output
echo "Using Python: \$PYTHON_EXE"
\$PYTHON_EXE --version
pip list | head -10




# Run the Python script with arguments
python $SCRIPT \\
    --data_file "$DATA_FILE" \\
    --ranking_file "$RANKING_FILE" \\
    --save_dir "$SAVE_DIR" \\
    --rank_range "$RANK_RANGE" \\
    --alpha_k "$ALPHA_K" \\
    --alpha_h "$ALPHA_H" \\
    --num_folds 5 \\
    --L "$L" \\
    --max_tau "$MAX_TAU"
EOT
  done
done