## Prescreening

Spike-train prescreening utilities for SISO datasets. Included one example config file `analysis_pipeline/config_dnms.json` for DNMS datasets. and corresponding example data data/selected_neurons1150_b034.pkl.

- Entry: `preprpoecessing_pipeline_siso_function.py`
- Plots/utils: `plot_all_plots_for_siso_cc.py`

### What it does
- Load `.pkl` spike trains, filter by spike count.
- Plot silent periods, raster, auto/cross-correlograms.
- Compute cross-correlation matrices at multiple resolutions; rank pairs.
- Save figures, correlation pickles, rankings, summaries.

### Install
```bash
pip install -r prescreening/requirements.txt
```

### Run 
```bash
python prescreening/preprpoecessing_pipeline_siso_function.py
```
Add other config under `analysis_pipeline/`. And use them by adding the path to the config file to the `default_configs` list in the `main` function.


### Key config fields (see sample JSON):
- `paths.file_dir`, `paths.analysis_dir`, `paths.pkl_file`
- `processing.sample_rate`, `processing.min_spikes`, `processing.n_top`, `processing.recompute_cc`
- `plotting.plot_all`, `plotting.raster_bin_size`
- `cc_configs`: `[bin_size_ms, max_lag_ms, "Label"]` list

### Outputs per `.pkl`
- Summary: `analysis_dir/<pkl_stem>/summary.txt`
- Rankings: `pair_rankings_<label>.csv`
- Correlation data: `crosscorrs_edge_mean_<flag>_<label>.pkl`, `autocorrs_edge_mean_<flag>_<label>.pkl`
- Figures: correlation matrices, top/bottom pairs, bump score histograms, QC plots if enabled.

Tips: binary spike trains are auto-converted to indices; set `recompute_cc=false` to reuse cached correlations; disable `plot_all` for speed. 
