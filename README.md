## Prescreening

Spike-train prescreening utilities for SISO datasets. This pipeline performs initial analysis on spike train data and prepares ranked neuron pairs for downstream GLM fitting.

### Overview

The pipeline consists of two main stages:

1. **Stage 1: Initial Analysis** (`preprpoecessing_pipeline_siso_function.py`)
   - Load `.pkl` spike trains, filter by spike count
   - Plot silent periods, raster plots, auto/cross-correlograms
   - Compute cross-correlation matrices at multiple resolutions
   - Rank neuron pairs by bump scores
   - Save figures, correlation pickles, rankings, and summaries

2. **Stage 2: GLM Fitting** (`stage2_stationary_fit_exp_data.sh`)
   - Fit GLM models to ranked neuron pairs from Stage 1
   - Performs cross-validation across multiple parameter sets
   - Submits batch jobs for parallel processing

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

#### Local Execution (Stage 1)

For small datasets, run Stage 1 locally:

```bash
python preprpoecessing_pipeline_siso_function.py
```

The script will automatically use the default config file: `analysis_pipeline/config_dnms.json`

#### SLURM Cluster Execution

For large datasets or batch processing, use the provided SLURM scripts:

**Stage 1 - Initial Analysis:**
```bash
sbatch stage1_run_inital_analysis.sh
```

**Stage 2 - GLM Fitting:**
```bash
bash stage2_stationary_fit_exp_data.sh
```

**Note:** Before running on SLURM, update the following in both scripts:
- `#SBATCH --mail-user`: Change to your email address
- `VENV_PATH`: Change to your virtual environment path
- `#SBATCH --account`: Change to your SLURM account (if different)
- `#SBATCH --partition`: Adjust based on your cluster

### Configuration

#### Adding New Config Files

1. Create a new JSON config file under `analysis_pipeline/`
2. Add the config path to the `default_configs` list in the `main()` function of `preprpoecessing_pipeline_siso_function.py`

#### Key Config Fields

See `analysis_pipeline/config_dnms.json` for a complete example:

**Paths:**
- `paths.file_dir`: Directory containing `.pkl` files
- `paths.analysis_dir`: Output directory for analysis results
- `paths.pkl_file`: Wildcard pattern for `.pkl` files (e.g., `"*200s.pkl"`)

**Processing:**
- `processing.sample_rate`: Sampling rate of spike trains
- `processing.min_spikes`: Minimum spike count to filter neurons
- `processing.n_top`: Number of top/bottom pairs to analyze
- `processing.recompute_cc`: Whether to recompute cross-correlations (set `false` to reuse cached)

**Plotting:**
- `plotting.plot_all`: Generate all QC plots (set `false` for speed)
- `plotting.raster_bin_size`: Bin size for raster plots (ms)

**Cross-Correlation Configs:**
- `cc_configs`: List of `[bin_size_ms, max_lag_ms, "Label"]` configurations
  - Example: `[20, 250, "Fine"]` computes correlations with 20ms bins up to 250ms lag

### Output Structure

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
- `analysis_pipeline/config_dnms.json`: Example configuration file 
