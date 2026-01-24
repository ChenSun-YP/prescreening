## Prescreening

Spike-train prescreening utilities for SISO datasets. This pipeline performs initial analysis on spike train data and prepares ranked neuron pairs for downstream GLM fitting.
### Installation


For a virtual environment start with 
`conda create -n prescreening python=3.10` then activate it with `conda activate prescreening` 

then install the requirements with `pip install -r requirements.txt`

### Stage 1: Initial Analysis

The Stage 1 pipeline (`preprpoecessing_pipeline_siso_function.py`) performs the following:
- Loads `.pkl` spike train files and filters neurons by spike count.
- Generates QC plots for silent periods, spike raster, auto- and cross-correlograms.
- Computes cross-correlation matrices at multiple time resolutions.
- Ranks all neuron pairs by bump score for each config.
- Saves all figures, correlation data, rankings, and summary files.
#### Configuration

##### Adding/Editing Configs

1. Add a new JSON config to `analysis_pipeline/`.
2. Include its path in the `default_configs` list in `preprpoecessing_pipeline_siso_function.py`'s `main()`.

Example (`analysis_pipeline/config_dnms.json`):

```json
{
  "paths": {
    "file_dir": "DNMS_data/",
    "analysis_dir": "DNMS_data/analysis",
    "pkl_file": "*200s.pkl"
  },
  "processing": {
    "sample_rate": 1000,
    "min_spikes": 40,
    "n_top": 40,
    "recompute_cc": true
  },
  "plotting": {
    "plot_all": true,
    "raster_bin_size": 5
  },
  "cc_configs": [
    [20, 250, "Fine"],
    [40, 400, "Coarse"]
  ]
}
```

- **paths.file_dir**:   where to look for `.pkl` files  
- **paths.analysis_dir**:  output folder for results  
- **paths.pkl_file**: wildcard for filenames (e.g., `"*200s.pkl"`). Use wildcard like `*200s.pkl` to do a short test set. run full lenghth set by using `*.pkl` is goning to take hours.
- **processing.sample_rate**: in Hz  
- **processing.min_spikes**: neurons with fewer are excluded  
- **processing.n_top**: number of top/bottom pairs to analyze per label  
- **processing.recompute_cc**: set false to reuse previous cross-correlation files  
- **plotting.plot_all**: enables all plots for QC (set false for fast runs)  
- **plotting.raster_bin_size**: in ms  
- **cc_configs**: list of `[bin_size_ms, max_lag_ms, label]`; defines the cross-correlation analyses  

To run Stage 1 locally:
```bash
python preprpoecessing_pipeline_siso_function.py
```
To run on a SLURM cluster:
```bash
chmod +x stage1_run_inital_analysis.sh
./stage1_run_inital_analysis.sh
```

Stage 1 output:  
- `analysis_dir/<pkl_stem>/summary.txt`: Summary statistics  
- `analysis_dir/<pkl_stem>/pair_rankings_<label>.csv`: Pair rankings by bump score (for each resolution)  
- `analysis_dir/<pkl_stem>/crosscorrs_edge_mean_<flag>_<label>.pkl`: Cross-correlation data  
- `analysis_dir/<pkl_stem>/autocorrs_edge_mean_<flag>_<label>.pkl`: Auto-correlation data  
- `analysis_dir/<pkl_stem>/correlation_matrix_edge_mean_<flag>_<label>.png`: Correlation matrix plot  
- `analysis_dir/<pkl_stem>/top_cc_plots_<label>.png`: Top/bottom pair correlogram plots  
- `analysis_dir/<pkl_stem>/bump_score_distribution_<label>.png`: Bump score histograms  
- `analysis_dir/<pkl_stem>/spike_raster_binned.png`: Raster plot  
- `analysis_dir/<pkl_stem>/all_neurons_silent_periods.png`: Silent period analysis  

**Note:** Stage 2 takes a rankings file and fits models to pairs within specified rank limits. Higher rank = more structured cross-correlation.


### Stage 2: GLM Fitting

The Stage 2 pipeline (`stage2_stationary_fit_exp_data.sh`) performs the following:
- Fits GLM models to the ranked neuron pairs produced in Stage 1.
- Performs cross-validation across multiple parameter configurations.
- Can be used on HPC cluster to submits batch jobs for parallel processing.

To run Stage 2 locally, use:
For batch
```bash
python glm_fit_cv_one_neuron.py --data_file <data_file> --ranking_file <ranking_file> --save_dir <save_dir> --rank_range <rank_range> --alpha_k <alpha_k> --alpha_h <alpha_h> --num_folds <num_folds> --L <L> --max_tau <max_tau>
```
For single pair
```bash
python glm_fit_cv_one_neuron.py \
    --data_file "data/selected_neurons1150b034.pkl" \
    --neuron_pairs "n005_L_CA3_wire_2_cell_1:n025_L_CA3_wire_7_cell_1" \
    --f_SISO True \
    --save_dir "data/selected_neurons1150b034" \
    --alpha_k 0.7 \
    --alpha_h 0.7 \
    --L 5 \
    --chunk_size 1000 \
    --max_tau 200 \
    --num_folds 1
```
To run on a SLURM cluster, config the bash script then:
```bash
chmod +x stage2_stationary_fit_exp_data.sh
./stage2_stationary_fit_exp_data.sh
```


**Note:** Before running on SLURM, update the following in both scripts:
- `#SBATCH --mail-user`: Change to your email address
- `VENV_PATH`: Change to your virtual environment path
- `#SBATCH --account`: dsong945
- `#SBATCH --partition`: choose from `main,debug,gpu,epyc-64`


<!--
add filters here at very end

access csv or pkl files to do so

>


<!-- ### Output Structure

For each `.pkl` file processed, outputs are saved in `analysis_dir/<pkl_stem>/`:

**Summary Files:**
- `summary.txt`: Processing summary with statistics

**Rankings:**
- `pair_rankings_<label>.csv`: Ranked neuron pairs by bump score for each resolution

**Correlation Data:**
- `crosscorrs_edge_mean_<flag>_<label>.pkl`: Cross-correlation data
- `autocorrs_edge_mean_<flag>_<label>.pkl`: Auto-correlation data

**Figures:**
- `correlation_matrix_edge_mean_<flag>_<label>.png`: Correlation matrices
- `top_cc_plots_<label>.png`: Top/bottom pair cross-correlograms
- `bump_score_distribution_<label>.png`: Bump score histograms
- `spike_raster_binned.png`: Raster plot (if `plot_all=true`)
- `all_neurons_silent_periods.png`: Silent period analysis (if `plot_all=true`)

### Tips

- **Binary spike trains**: Automatically converted to spike indices
- **Caching**: Set `recompute_cc=false` to reuse cached correlations and speed up re-runs
- **Performance**: Disable `plot_all` for faster processing when QC plots aren't needed
- **Multiple resolutions**: The pipeline processes all configurations in `cc_configs` automatically

### Files

- `preprpoecessing_pipeline_siso_function.py`: Main preprocessing pipeline
- `stage1_run_inital_analysis.sh`: SLURM script for Stage 1
- `stage2_stationary_fit_exp_data.sh`: SLURM batch submission script for Stage 2
- `utils.py`: Utility functions for plotting and correlation computation
- `plot_all_plots_for_siso_cc.py`: Additional plotting utilities
- `analysis_pipeline/config_dnms.json`: Example configuration file  -->
