import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import logging
import traceback
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from utils.create_dataset import create_dataset
from utils.time_rescaling_ks import time_rescaling_ks
from model.laguerre import ParameterizedLaguerreBasis_multineuron
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import gc
import scipy.signal
import psutil

# Try to import tqdm for progress bars, fallback to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        """Simple progress indicator if tqdm is not available"""
        def __init__(self, iterable, desc=None, total=None, leave=True, **kwargs):
            self.iterable = iterable
            self.desc = desc or "Progress"
            if total is None and hasattr(iterable, '__len__'):
                self.total = len(iterable)
            else:
                self.total = total
            self.leave = leave
            self.current = 0
        
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.total and (self.current % max(1, self.total // 20) == 0 or self.current == self.total):
                    pct = self.current * 100 // self.total if self.total else 0
                    print(f"\r{self.desc}: {pct}% ({self.current}/{self.total})", end='', flush=True)
                yield item
            if self.leave:
                print(f"\r{self.desc}: 100% ({self.total}/{self.total})", flush=True)
            else:
                print("", flush=True)  # New line

SET_USE_BIC_LLF(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default font size for matplotlib plots
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'figure.figsize': (10, 8)
})

# # For better readability in dense plots
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 0.3



# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='GLM fitting for neuron pairs from a ranking CSV with kernel overlay')
    parser.add_argument('--data_file', type=str, default='DNMS_data/selected_neurons.pkl',
                        help='Filename of the spike data pickle file')
    parser.add_argument('--ranking_file', type=str, default='DNMS_data/analysis/selected_neurons/pair_rankings_ultra-fine.csv',
                        help='CSV file with neuron pair rankings (Neuron1, Neuron2, BumpScore, Rank)')
    parser.add_argument('--neuron_pairs', type=str, default=None,
                        help='Comma-separated list of neuron pairs in format neuron1:neuron2, e.g., "n001_X:n001_Y,n002_X:n002_Y"')
    parser.add_argument('--save_dir', type=str, default='DNMS_data/single_pairs_analysis',
                        help='Directory to save results')
    parser.add_argument('--max_tau', type=int, default=100,
                        help='Maximum time lag for basis functions (in ms)')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Sample rate of the spike data')
    parser.add_argument('--num_folds', type=int, default=4,
                        help='Number of cross-validation folds')
    parser.add_argument('--L', type=int, default=4,
                        help='Number of basis functions')
    parser.add_argument('--alpha_k', type=float, default=0.7,
                        help='Alpha parameter for feedforward basis')
    parser.add_argument('--alpha_h', type=float, default=0.95,
                        help='Alpha parameter for feedback basis')
    parser.add_argument('--rank_range', type=str, default='31-33',
                        help='Range of ranks to analyze (e.g., 1-10, 5-20)')
    parser.add_argument('--f_SISO', type=bool, default=True,
                        help='do the glm fit using output neuron as input neuron too')
    parser.add_argument('--chunk_size', type=int, default=10000,  # Lowered default from 50000
                        help='Chunk size for design matrix processing (lower to reduce memory usage)')
    parser.add_argument('--only_use_trial_data', action='store_true',
                        default=False,
                        help='Only use trial data for the neurons')
    return parser.parse_args()

# Global variables that will be set in __main__ or by external scripts
cmd_args = None
SAVE_DIR = None
max_tau = None
sample_rate = None
NUMBER_OF_FOLDS = None
L = None
alpha_k = None
alpha_h = None
RANKING_FILE = None
RANK_RANGE = None
f_SISO = None
neurons = None
neuron_pairs = None

# Helper function to get memory usage
def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_status(prefix=""):
    """Print current memory usage"""
    mem_mb = get_memory_usage_mb()
    print(f"{prefix}Memory usage: {mem_mb:.1f} MB", flush=True)
    return mem_mb

# Pre-process neuron data function
def preprocess_neuron_data(input_neuron, output_neuron, neurons, sample_rate):
    print(f'Pre-processing: Input: {input_neuron}, Output: {output_neuron}')
    input_neuron_data = np.array([sample/sample_rate for sample in neurons[input_neuron]])
    output_neuron_data = np.array([sample/sample_rate for sample in neurons[output_neuron]])

    if np.sum(input_neuron_data == 0) > 100:
        print('input_neuron_data is a spike train')
    else:
        print('input_neuron_data is a spike time, converting to spike train')
        if np.all(np.array(input_neuron_data, dtype=int) == input_neuron_data):
            print('input_neuron_data is in ms')
        else:
            print('input_neuron_data is in seconds')
            input_neuron_data = input_neuron_data * 1000
        data_spike_train = np.zeros(int(np.max(input_neuron_data))+5)
        spike_indices = np.round(input_neuron_data).astype(int)
        data_spike_train[spike_indices] = 1
        input_neuron_data = data_spike_train

    if np.sum(output_neuron_data == 0) > 100:
        print('output_neuron_data is a spike train')
    else:
        print('output_neuron_data is a spike time, converting to spike train')
        if np.all(np.array(output_neuron_data, dtype=int) == output_neuron_data):
            print('output_neuron_data is in ms')
        else:
            print('output_neuron_data is in seconds')
            output_neuron_data = output_neuron_data * 1000
        data_spike_train = np.zeros(int(np.max(output_neuron_data))+5)
        spike_indices = np.round(output_neuron_data).astype(int)
        data_spike_train[spike_indices] = 1
        output_neuron_data = data_spike_train

    min_length = min(len(input_neuron_data), len(output_neuron_data))
    input_neuron_data = input_neuron_data[:min_length]
    output_neuron_data = output_neuron_data[:min_length]
    input_neuron_data = np.where(input_neuron_data > 1, 1, input_neuron_data)
    output_neuron_data = np.where(output_neuron_data > 1, 1, output_neuron_data)
    if np.any(input_neuron_data < 0) or np.any(input_neuron_data > 1):
        raise ValueError('input_neuron_data is not binary')
    if np.any(output_neuron_data < 0) or np.any(output_neuron_data > 1):
        raise ValueError('output_neuron_data is not binary')

    x_full = torch.from_numpy(input_neuron_data).float().reshape(1, 1, -1)
    y_full = torch.from_numpy(output_neuron_data).float().reshape(1, 1, -1)
    full_set, _, _, _ = create_dataset(x_full, y_full)
    return (full_set[0][0], full_set[0][1], x_full, y_full)

def fit_glm_to_stdp(data, target, L, alpha_k, alpha_h, input_neuron, output_neuron, rank, chunk_size=10000, min_spike_factor=1.5, max_tau=100, folds=5, save_dir='.', sigma=None, n_resample=10):
    '''
    Fits a Generalized Linear Model (GLM) to Spike-Timing-Dependent Plasticity (STDP) data.
    Handles both standard and sigma (offset) cases, with proper cross-validation splitting of sigma.
    '''
    logger = logging.getLogger('glm_fitting')
    print(f"\n{'='*60}")
    print(f"Starting GLM fit: {input_neuron} -> {output_neuron} (Rank {rank})")
    print(f"{'='*60}")
    print_memory_status("Initial ")
    
    # ff_basis = ParameterizedLaguerreBasis(num_basis=L, max_tau=max_tau, initial_alpha=alpha_k, name='ff_basis')().T
    # fb_basis = ParameterizedLaguerreBasis(num_basis=L, max_tau=max_tau, initial_alpha=alpha_h, name='fb_basis')().T
    with torch.no_grad():
        print("Creating basis functions...", flush=True)
        ff_basis= ParameterizedLaguerreBasis_multineuron( [L], [max_tau], alpha_k, name='ff_basis' )().T.squeeze(-1)
        fb_basis= ParameterizedLaguerreBasis_multineuron( [L], [max_tau], alpha_h, name='fb_basis' )().T.squeeze(-1) # tau,L
        print(f"Basis shapes: FF={ff_basis.shape}, FB={fb_basis.shape}")
        ff_basis_batched = ff_basis.unsqueeze(0).permute(2, 0, 1).flip(dims=[-1])
        fb_basis_batched = fb_basis.unsqueeze(0).permute(2, 0, 1).flip(dims=[-1])
        x_full = data[0].unsqueeze(0).unsqueeze(0)
        y_full = data[1].unsqueeze(0).unsqueeze(0)
        y_target_full = target.flatten().cpu().numpy()
        
        T_full = max(x_full.shape[-1], y_full.shape[-1])
        tmax = T_full
        print(f"Data length: {T_full:,} time points")
        
        eps = 1e-15
        p_full = np.mean(y_target_full)
        p_full = np.clip(p_full, eps, 1 - eps)
        baseline_ce_full = -np.mean(y_target_full * np.log(p_full) + (1 - y_target_full) * np.log(1 - p_full))
        
        all_metrics = []
        ff_coeffs_list = []
        fb_coeffs_list = []
        k0_list = []
        valid_folds = 0
        
        # Skip cross-validation if folds == 1
        if folds == 1:
            print(f"No cross-validation (folds=1): Fitting directly on full dataset ({T_full:,} samples)")
        else:
            fold_size = tmax // folds
            folds_idx = [(i * fold_size, min((i + 1) * fold_size, tmax)) for i in range(folds)]
            print(f"Cross-validation: {folds} folds, ~{fold_size:,} samples per fold")

        for fold_idx, (val_start, val_end) in enumerate(tqdm(folds_idx if folds > 1 else [(0, 0)], desc=f"CV Folds ({input_neuron}->{output_neuron})" if folds > 1 else f"Fitting ({input_neuron}->{output_neuron})", total=folds if folds > 1 else 1)):
            if folds == 1:
                # No cross-validation: use full dataset for both train and val
                train_indices = torch.arange(0, tmax)
                val_indices = torch.arange(0, tmax)  # Same as train for metrics
                train_x = x_full
                train_y = y_full
                val_x = x_full
                val_y = y_full
                train_target = torch.from_numpy(y_target_full).float()
                val_target = torch.from_numpy(y_target_full).float()
            else:
                # Cross-validation: split train/val
                train_indices = torch.cat([torch.arange(0, val_start), torch.arange(val_end, tmax)])
                val_indices = torch.arange(val_start, val_end)
                train_x = x_full[:, :, train_indices]
                train_y = y_full[:, :, train_indices]
                val_x = x_full[:, :, val_indices]
                val_y = y_full[:, :, val_indices]
                train_target = torch.from_numpy(y_target_full[train_indices]).float()
                val_target = torch.from_numpy(y_target_full[val_indices]).float()
            
            y_val = val_target.cpu().numpy()
            n_spikes_val = int(np.sum(y_val))
            p_val = np.mean(y_val)
            p_val = np.clip(p_val, eps, 1 - eps)
            baseline_ce_val = -np.mean(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)) if n_spikes_val > 0 else float('inf')
            
            if folds > 1 and (n_spikes_val == 0 or baseline_ce_val > min_spike_factor * baseline_ce_full):
                logger.info(f"Skipping fold {fold_idx + 1}: {n_spikes_val} spikes")
                continue
            
            valid_folds += 1
            T = train_x.shape[-1]
            T_val = val_x.shape[-1]
            num_basis = ff_basis_batched.shape[0]
            padding = ff_basis_batched.shape[-1] - 1
            temp_prefix = f"{input_neuron}_{output_neuron}_rank{rank}"
            
            if sigma is not None:
                X_file = os.path.join(save_dir, f'X_temp_{temp_prefix}_fold{fold_idx}.npy')
                X_val_file = os.path.join(save_dir, f'X_val_temp_{temp_prefix}_fold{fold_idx}.npy')
                X = np.memmap(X_file, dtype='float32', mode='w+', shape=(T, 2 * num_basis))
                X_val = np.memmap(X_val_file, dtype='float32', mode='w+', shape=(T_val, 2 * num_basis))
                def process_chunks_to_design(input_x, input_y, basis_ff, basis_fb, total_length, chunk_size, design_matrix, desc="Processing"):
                    step_size = chunk_size
                    num_chunks = (total_length + step_size - 1) // step_size
                    for i in tqdm(range(num_chunks), desc=desc, leave=False, total=num_chunks):
                        start = i * step_size
                        end = min(start + step_size + padding, total_length)
                        chunk_x = input_x[:, :, start:end]
                        chunk_y = input_y[:, :, start:end]
                        if chunk_x.shape[-1] < padding + 1:
                            chunk_x = F.pad(chunk_x, (padding + 1 - chunk_x.shape[-1], 0), mode='constant', value=0)
                            chunk_y = F.pad(chunk_y, (padding + 1 - chunk_y.shape[-1], 0), mode='constant', value=0)
                        ff_conv = F.conv1d(chunk_x, basis_ff, padding=padding)[:, :, :-padding + 1]
                        fb_conv = F.conv1d(chunk_y, basis_fb, padding=padding)[:, :, :-padding + 1]
                        valid_length = total_length - start if i == num_chunks - 1 else step_size
                        ff_conv_chunk = ff_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                        fb_conv_chunk = fb_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                        design_matrix[start:start + valid_length, :num_basis] = ff_conv_chunk
                        design_matrix[start:start + valid_length, num_basis:] = fb_conv_chunk
                        del chunk_x, chunk_y, ff_conv, fb_conv, ff_conv_chunk, fb_conv_chunk
                        torch.cuda.empty_cache()
                        # Periodic memory cleanup for long sequences
                        if (i + 1) % 10 == 0:
                            gc.collect()
                print(f"  Fold {fold_idx + 1}: Building design matrix (train: {T:,} samples)...", flush=True)
                process_chunks_to_design(train_x, train_y, ff_basis_batched, fb_basis_batched, T, chunk_size, X, desc=f"  Train chunks")
                print(f"  Fold {fold_idx + 1}: Building design matrix (val: {T_val:,} samples)...", flush=True)
                process_chunks_to_design(val_x, val_y, ff_basis_batched, fb_basis_batched, T_val, chunk_size, X_val, desc=f"  Val chunks")
                y_train = train_target.cpu().numpy()
                y_val = val_target.cpu().numpy()
                # Split sigma for train/val
                sigma_np = np.asarray(sigma)
                sigma_train = sigma_np[train_indices]
                sigma_val = sigma_np[val_indices]
                # When folds=1, sigma_train is the full sigma
                if folds == 1:
                    sigma_full_single = sigma_train
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                print(f"  VIF: {[f'{v:.2f}' for v in vif]}")  # VIF > 5-10 indicates issues
                print(f"  Fold {fold_idx + 1}: Fitting GLM (train: {len(y_train):,} samples)...", flush=True)
                glm_model = sm.GLM(y_train, X, family=sm.families.Binomial(link=sm.families.links.Probit()), offset=sigma_train)
                model_fit = glm_model.fit(maxiter=50, tol=1e-9)
                # Get iteration count safely
                iterations = 'N/A'
                try:
                    if hasattr(model_fit, 'fit_history') and model_fit.fit_history:
                        iterations = model_fit.fit_history.get('iteration', 'N/A')
                    elif hasattr(model_fit, 'mle_retvals') and model_fit.mle_retvals:
                        iterations = model_fit.mle_retvals.get('iterations', 'N/A')
                except (AttributeError, TypeError):
                    pass
                print(f"  Fold {fold_idx + 1}: GLM fit complete (iterations: {iterations})", flush=True)
                conf_int = model_fit.conf_int()
                y_pred = model_fit.fittedvalues
                y_pred_val = model_fit.predict(X_val, offset=sigma_val)
                n_basis = ff_basis.shape[1]
                params = model_fit.params
                ff_coeffs = params[:n_basis]
                fb_coeffs = params[n_basis:]
                k0 = None
            else:
                X_file = os.path.join(save_dir, f'X_temp_{temp_prefix}_fold{fold_idx}.npy')
                X_val_file = os.path.join(save_dir, f'X_val_temp_{temp_prefix}_fold{fold_idx}.npy')
                X = np.memmap(X_file, dtype='float32', mode='w+', shape=(T, 2 * num_basis + 1))
                X_val = np.memmap(X_val_file, dtype='float32', mode='w+', shape=(T_val, 2 * num_basis + 1))
                X[:, 0] = 1 # this is the offset and the fitted c0
                X_val[:, 0] = 1
                def process_chunks_to_design(input_x, input_y, basis_ff, basis_fb, total_length, chunk_size, design_matrix, desc="Processing"):
                    step_size = chunk_size
                    num_chunks = (total_length + step_size - 1) // step_size
                    for i in tqdm(range(num_chunks), desc=desc, leave=False, total=num_chunks):
                        start = i * step_size
                        end = min(start + step_size + padding, total_length)
                        chunk_x = input_x[:, :, start:end]
                        chunk_y = input_y[:, :, start:end]
                        if chunk_x.shape[-1] < padding + 1:
                            chunk_x = F.pad(chunk_x, (padding + 1 - chunk_x.shape[-1], 0), mode='constant', value=0)
                            chunk_y = F.pad(chunk_y, (padding + 1 - chunk_y.shape[-1], 0), mode='constant', value=0)
                        ff_conv = F.conv1d(chunk_x, basis_ff, padding=padding)[:, :, :-padding + 1]
                        fb_conv = F.conv1d(chunk_y, basis_fb, padding=padding)[:, :, :-padding + 1]
                        valid_length = total_length - start if i == num_chunks - 1 else step_size
                        ff_conv_chunk = ff_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                        fb_conv_chunk = fb_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                        design_matrix[start:start + valid_length, 1:num_basis + 1] = ff_conv_chunk
                        design_matrix[start:start + valid_length, num_basis + 1:] = fb_conv_chunk
                        del chunk_x, chunk_y, ff_conv, fb_conv, ff_conv_chunk, fb_conv_chunk
                        torch.cuda.empty_cache()
                        # Periodic memory cleanup for long sequences
                        if (i + 1) % 10 == 0:
                            gc.collect()
                print(f"  Fold {fold_idx + 1}: Building design matrix (train: {T:,} samples)...", flush=True)
                process_chunks_to_design(train_x, train_y, ff_basis_batched, fb_basis_batched, T, chunk_size, X, desc=f"  Train chunks")
                print(f"  Fold {fold_idx + 1}: Building design matrix (val: {T_val:,} samples)...", flush=True)
                process_chunks_to_design(val_x, val_y, ff_basis_batched, fb_basis_batched, T_val, chunk_size, X_val, desc=f"  Val chunks")
                y_train = train_target.cpu().numpy()
                y_val = val_target.cpu().numpy()
                # When folds=1, no sigma offset
                if folds == 1:
                    sigma_full_single = None
                # from statsmodels.stats.outliers_influence import variance_inflation_factor
                # vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                # print(vif)  # VIF > 5-10 indicates issues

                print(f"  Fold {fold_idx + 1}: Fitting GLM (train: {len(y_train):,} samples)...", flush=True)
                glm_model = sm.GLM(y_train, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
                model_fit = glm_model.fit(maxiter=100, tol=1e-9)
                # Get iteration count safely
                iterations = 'N/A'
                try:
                    if hasattr(model_fit, 'fit_history') and model_fit.fit_history:
                        iterations = model_fit.fit_history.get('iteration', 'N/A')
                    elif hasattr(model_fit, 'mle_retvals') and model_fit.mle_retvals:
                        iterations = model_fit.mle_retvals.get('iterations', 'N/A')
                except (AttributeError, TypeError):
                    pass
                print(f"  Fold {fold_idx + 1}: GLM fit complete (iterations: {iterations})", flush=True)




                # # Fit GLM with IRLS
                # model_fit_irls = glm_model.fit(method='IRLS', maxiter=500, tol=1e-9)
                # # Fit GLM with BFGS, capturing optimization result
                # print('irls')
                # model_fit_irls = glm_model.fit(method='IRLS', maxiter=500, tol=1e-9)
                # model_fit = glm_model.fit(method='bfgs', maxiter=500, tol=1e-9)



                conf_int = model_fit.conf_int()
                y_pred = model_fit.fittedvalues
                y_pred_val = model_fit.predict(X_val)
                n_basis = ff_basis.shape[1]
                params = model_fit.params
                ff_coeffs = params[1:n_basis + 1]
                fb_coeffs = params[n_basis + 1:]
                k0 = params[0]
         



            metrics = compute_metrics(y_train, y_pred, y_val, y_pred_val, model_fit, X, X_val, L, alpha_k, alpha_h, ff_basis, fb_basis, sigma)
            all_metrics.append(metrics)
            k0_list.append(k0)
            ff_coeffs_list.append(ff_coeffs)
            fb_coeffs_list.append(fb_coeffs)
            
            # If folds == 1, save model and design matrix for later use (before cleanup)
            if folds == 1:
                model_full_single = model_fit
                conf_int_single = conf_int
                X_full_single = X
                X_val_full_single = X_val
                X_file_single = X_file
                X_val_file_single = X_val_file
            
            X.flush()
            X_val.flush()
            if folds > 1:  # Only delete if doing CV (will reuse if folds==1)
                del X, X_val, model_fit, glm_model
                # Clean up temporary files immediately to save disk space
                try:
                    os.remove(X_file)
                    os.remove(X_val_file)
                except:
                    pass
            gc.collect()
            print(f"  Fold {fold_idx + 1}: Complete. Memory: {get_memory_usage_mb():.1f} MB", flush=True)
        if valid_folds == 0:
            logger.warning("No folds had sufficient spikes or stable CE.")
            return None, None, [], None, None, ff_basis, fb_basis
        
        # If folds == 1, we already fit on the full dataset, so reuse that model
        if folds == 1:
            # Use the model from the single "fold" as the full model
            model_full = model_full_single
            conf_int = conf_int_single
            # For metrics, use the same predictions for both train and val
            Pb_full = model_full.predict(X_full_single, offset=sigma_full_single if sigma is not None else None)
            y_full_np = y_target_full
            ks_full = time_rescaling_ks(Pb_full, y_full_np)
            all_metrics_with_ks = {
                'fold_metrics': all_metrics,
                'KS_full': ks_full
            }
            # Clean up
            del X_full_single, X_val_full_single, model_full_single, glm_model
            try:
                os.remove(X_file_single)
                os.remove(X_val_file_single)
            except:
                pass
            avg_ff_coeffs = ff_coeffs_list[0] if ff_coeffs_list else None
            avg_fb_coeffs = fb_coeffs_list[0] if fb_coeffs_list else None
            k0_avg = k0_list[0] if k0_list else None
            del ff_basis_batched, fb_basis_batched, x_full, y_full, y_target_full
            del ff_coeffs_list, fb_coeffs_list, k0_list
            torch.cuda.empty_cache()
            gc.collect()
            print(f"✓ GLM fit complete: {input_neuron} -> {output_neuron} (Rank {rank})")
            print(f"  No cross-validation (folds=1), Final memory: {get_memory_usage_mb():.1f} MB")
            print(f"{'='*60}\n")
            return model_full, conf_int, all_metrics_with_ks, avg_ff_coeffs, avg_fb_coeffs, k0_avg, ff_basis, fb_basis
        
        print(f"\nFull dataset fit: Building design matrix ({T_full:,} samples)...", flush=True)
        print_memory_status("Before full fit ")
        # Full fit
        if sigma is not None:
            X_full_file = os.path.join(save_dir, f'X_full_temp_{temp_prefix}.npy')
            X_full = np.memmap(X_full_file, dtype='float32', mode='w+', shape=(T_full, 2 * num_basis))
            def process_chunks_to_design(input_x, input_y, basis_ff, basis_fb, total_length, chunk_size, design_matrix, desc="Processing"):
                step_size = chunk_size
                num_chunks = (total_length + step_size - 1) // step_size
                for i in tqdm(range(num_chunks), desc=desc, leave=False, total=num_chunks):
                    start = i * step_size
                    end = min(start + step_size + padding, total_length)
                    chunk_x = input_x[:, :, start:end]
                    chunk_y = input_y[:, :, start:end]
                    if chunk_x.shape[-1] < padding + 1:
                        chunk_x = F.pad(chunk_x, (padding + 1 - chunk_x.shape[-1], 0), mode='constant', value=0)
                        chunk_y = F.pad(chunk_y, (padding + 1 - chunk_y.shape[-1], 0), mode='constant', value=0)
                    ff_conv = F.conv1d(chunk_x, basis_ff, padding=padding)[:, :, :-padding + 1]
                    fb_conv = F.conv1d(chunk_y, basis_fb, padding=padding)[:, :, :-padding + 1]
                    valid_length = total_length - start if i == num_chunks - 1 else step_size
                    ff_conv_chunk = ff_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                    fb_conv_chunk = fb_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                    design_matrix[start:start + valid_length, :num_basis] = ff_conv_chunk
                    design_matrix[start:start + valid_length, num_basis:] = fb_conv_chunk
                    del chunk_x, chunk_y, ff_conv, fb_conv, ff_conv_chunk, fb_conv_chunk
                    torch.cuda.empty_cache()
                    # Periodic memory cleanup for long sequences
                    if (i + 1) % 10 == 0:
                        gc.collect()
            process_chunks_to_design(x_full, y_full, ff_basis_batched, fb_basis_batched, T_full, chunk_size, X_full, desc="Full design matrix")
            sigma_np = np.asarray(sigma)
            sigma_full = sigma_np[:T_full]
            print("Fitting full GLM model...", flush=True)
            glm_full = sm.GLM(y_target_full, X_full, family=sm.families.Binomial(link=sm.families.links.Probit()), offset=sigma_full)
            model_full = glm_full.fit(maxiter=100, tol=1e-9)
            # Get iteration count safely
            iterations = 'N/A'
            try:
                if hasattr(model_full, 'fit_history') and model_full.fit_history:
                    iterations = model_full.fit_history.get('iteration', 'N/A')
                elif hasattr(model_full, 'mle_retvals') and model_full.mle_retvals:
                    iterations = model_full.mle_retvals.get('iterations', 'N/A')
            except (AttributeError, TypeError):
                pass
            print(f"Full GLM fit complete (iterations: {iterations})", flush=True)
        else:
            X_full_file = os.path.join(save_dir, f'X_full_temp_{temp_prefix}.npy')
            X_full = np.memmap(X_full_file, dtype='float32', mode='w+', shape=(T_full, 2 * num_basis + 1))
            X_full[:, 0] = 1
            def process_chunks_to_design(input_x, input_y, basis_ff, basis_fb, total_length, chunk_size, design_matrix, desc="Processing"):
                step_size = chunk_size
                num_chunks = (total_length + step_size - 1) // step_size
                for i in tqdm(range(num_chunks), desc=desc, leave=False, total=num_chunks):
                    start = i * step_size
                    end = min(start + step_size + padding, total_length)
                    chunk_x = input_x[:, :, start:end]
                    chunk_y = input_y[:, :, start:end]
                    if chunk_x.shape[-1] < padding + 1:
                        chunk_x = F.pad(chunk_x, (padding + 1 - chunk_x.shape[-1], 0), mode='constant', value=0)
                        chunk_y = F.pad(chunk_y, (padding + 1 - chunk_y.shape[-1], 0), mode='constant', value=0)
                    ff_conv = F.conv1d(chunk_x, basis_ff, padding=padding)[:, :, :-padding + 1]
                    fb_conv = F.conv1d(chunk_y, basis_fb, padding=padding)[:, :, :-padding + 1]
                    valid_length = total_length - start if i == num_chunks - 1 else step_size
                    ff_conv_chunk = ff_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                    fb_conv_chunk = fb_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
                    design_matrix[start:start + valid_length, 1:num_basis + 1] = ff_conv_chunk
                    design_matrix[start:start + valid_length, num_basis + 1:] = fb_conv_chunk
                    del chunk_x, chunk_y, ff_conv, fb_conv, ff_conv_chunk, fb_conv_chunk
                    torch.cuda.empty_cache()
                    # Periodic memory cleanup for long sequences
                    if (i + 1) % 10 == 0:
                        gc.collect()
            process_chunks_to_design(x_full, y_full, ff_basis_batched, fb_basis_batched, T_full, chunk_size, X_full, desc="Full design matrix")
            print("Fitting full GLM model...", flush=True)
            glm_full = sm.GLM(y_target_full, X_full, family=sm.families.Binomial(link=sm.families.links.Probit()))
            model_full = glm_full.fit(maxiter=100, tol=1e-9)
            # Get iteration count safely
            iterations = 'N/A'
            try:
                if hasattr(model_full, 'fit_history') and model_full.fit_history:
                    iterations = model_full.fit_history.get('iteration', 'N/A')
                elif hasattr(model_full, 'mle_retvals') and model_full.mle_retvals:
                    iterations = model_full.mle_retvals.get('iterations', 'N/A')
            except (AttributeError, TypeError):
                pass
            print(f"Full GLM fit complete (iterations: {iterations})", flush=True)
        print("Computing predictions and KS test...", flush=True)
        Pb_full = model_full.predict(X_full, offset=sigma_full if sigma is not None else None)
        y_full_np = y_full.flatten().cpu().numpy()
        ks_full = time_rescaling_ks(Pb_full, y_full_np)
        all_metrics_with_ks = {
            'fold_metrics': all_metrics,
            'KS_full': ks_full
        }
        X_full.flush()
        del X_full, Pb_full, y_full_np
        try:
            os.remove(X_full_file)
        except:
            pass
        avg_ff_coeffs = np.mean(ff_coeffs_list, axis=0)
        avg_fb_coeffs = np.mean(fb_coeffs_list, axis=0)
        k0_avg = None if sigma is not None else np.mean([k for k in k0_list if k is not None])
        del ff_basis_batched, fb_basis_batched, x_full, y_full, y_target_full
        del ff_coeffs_list, fb_coeffs_list, k0_list
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✓ GLM fit complete: {input_neuron} -> {output_neuron} (Rank {rank})")
        print(f"  Valid folds: {valid_folds}/{folds}, Final memory: {get_memory_usage_mb():.1f} MB")
        print(f"{'='*60}\n")
        return model_full, conf_int, all_metrics_with_ks, avg_ff_coeffs, avg_fb_coeffs, k0_avg, ff_basis, fb_basis

def compute_metrics(y_true, y_pred, y_val_true, y_pred_val, model_fit, X, X_val, L, alpha_k, alpha_h, ff_basis, fb_basis, sigma):
    metrics = {}
    
    # Extract coefficients based on whether sigma is provided
    params = model_fit.params
    n_basis = ff_basis.shape[1]
    
    if sigma is not None:
        # When sigma is provided, no k0 in parameters
        metrics['k0'] = None
        c_ff = params[:n_basis]
        c_fb = params[n_basis:]
    else:
        # Without sigma, k0 is the first parameter
        metrics['k0'] = params[0]
        c_ff = params[1:n_basis + 1]
        c_fb = params[n_basis + 1:]

    metrics['deviance_reduction'] = (model_fit.null_deviance - model_fit.deviance) / model_fit.null_deviance * 100

    for dataset, y_t, y_p in [('train', y_true, y_pred), ('val', y_val_true, y_pred_val)]:
        rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
        y_range = np.max(y_t) - np.min(y_t) if np.max(y_t) != np.min(y_t) else 1
        y_mean = np.mean(y_t) or 1
        metrics[f'rmse_{dataset}'] = rmse
        metrics[f'nrmse_range_{dataset}'] = rmse / y_range
        metrics[f'nrmse_mean_{dataset}'] = rmse / y_mean

        eps = 1e-15
        y_p_clipped = np.clip(y_p, eps, 1 - eps)
        ce_loss = -np.mean(y_t * np.log(y_p_clipped) + (1 - y_t) * np.log(1 - y_p_clipped))
        metrics[f'cross_entropy_{dataset}'] = ce_loss
        
        p = np.mean(y_t)
        p = np.clip(p, eps, 1 - eps)
        baseline_ce = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        metrics[f'cross_entropy_{dataset}_normalized'] = ce_loss / baseline_ce if baseline_ce > 0 else ce_loss
        metrics[f'firing_rate_{dataset}'] = p

    truth_firing_rate = np.mean(y_true)
    y_pred_binary = (y_pred >= truth_firing_rate).astype(int)
    y_val_pred_binary = (y_pred_val >= truth_firing_rate).astype(int)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_binary)
    metrics['confusion_matrix_val'] = confusion_matrix(y_val_true, y_val_pred_binary)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc})

    Pb = model_fit.predict(X)
    Pb_val = model_fit.predict(X_val)
    ks_train = time_rescaling_ks(Pb, y_true)  #KS_score, Zs, b, b95, Tau
    ks_val = time_rescaling_ks(Pb_val, y_val_true)  #KS_score, Zs, b, b95, Tau

    # feedforward kernel
    ff_kernel = np.dot(ff_basis, c_ff)
    # feedback kernel
    fb_kernel = np.dot(fb_basis, c_fb)
    
    n_train = len(ks_train[4])
    if ks_val[0] is not None:
        n_val = len(ks_val[4])
        metrics.update({
            'KS_score': ks_train[0],
            'KS_score_normalized': ks_train[0] / np.sqrt(n_train),
            'KS':ks_train,
            'KS_score_val': ks_val[0],
            'KS_score_val_normalized': ks_val[0] / np.sqrt(n_val),
            'KS_val':ks_val,
            'n_train': n_train,
            'n_val': n_val,
            'Zs': ks_train[1], 'b': ks_train[2], 'b95': ks_train[3], 'Tau': ks_train[4],
            'Zs_val': ks_val[1], 'b_val': ks_val[2], 'b95_val': ks_val[3], 'Tau_val': ks_val[4],
            'log_likelihood': model_fit.llf,
            'null_log_likelihood': model_fit.llnull,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'c_ff': c_ff,
            'c_fb': c_fb,
            'ff_kernel': ff_kernel,
            'fb_kernel': fb_kernel,
            'L': L,
            'alpha_k': alpha_k,
            'alpha_h': alpha_h
        })
    else:
        metrics.update({
            'KS_score': ks_train[0],
            'KS_score_normalized': ks_train[0] / np.sqrt(n_train),
            'KS':ks_train,
            'KS_score_val': 0,
            'KS_score_val_normalized': 0,
            'KS_val':ks_val,
            'n_train': n_train,
            'n_val': 0,
            'Zs': ks_train[1], 'b': ks_train[2], 'b95': ks_train[3], 'Tau': ks_train[4],
            'Zs_val': ks_val[1], 'b_val': ks_val[2], 'b95_val': ks_val[3], 'Tau_val': ks_val[4],
            'log_likelihood': model_fit.llf,
            'null_log_likelihood': model_fit.llnull,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'c_ff': c_ff,
            'c_fb': c_fb,
            'L': L,
            'alpha_k': alpha_k,
            'alpha_h': alpha_h
        })
    return metrics
def kernel_data(all_metrics, ff_basis, fb_basis, c_ff, c_fb, L, alpha_k, alpha_h,k0, input_neuron, output_neuron, rank):
    ff_kernel_avg = np.dot(ff_basis, c_ff)
    fb_kernel_avg = np.dot(fb_basis, c_fb) 
    # Save kernels to pickle
    kernel_data = {
            'input_neuron': input_neuron,
            'output_neuron': output_neuron,
            'rank': rank,
            'c_ff': c_ff,
            'c_fb': c_fb,
            'ff_kernel': ff_kernel_avg,
            'fb_kernel': fb_kernel_avg,
            'L': L,
            'alpha_k': alpha_k,
            'alpha_h': alpha_h,
            'k0': k0,
            'all_metrics': all_metrics  # Add all fold metrics, including matrices
        }
    return kernel_data


def plot_fit_and_save_kernels(all_metrics, model_full, ff_basis, fb_basis, c_ff, c_fb, L, alpha_k, alpha_h,k0, input_neuron, output_neuron, x_full, y_full, rank):
    ff_kernel_avg = np.dot(ff_basis, c_ff)
    fb_kernel_avg = np.dot(fb_basis, c_fb)
    
    # Save kernels to pickle
    kernel_data = {
        'input_neuron': input_neuron,
        'output_neuron': output_neuron,
        'rank': rank,
        'ff_kernel': ff_kernel_avg,
        'fb_kernel': fb_kernel_avg,
        'L': L,
        'alpha_k': alpha_k,
        'alpha_h': alpha_h,
        'k0': k0,
        'all_metrics': all_metrics  # Add all fold metrics, including matrices
    }
    kernel_file = os.path.join(SAVE_DIR, f'kernels_{input_neuron}_{output_neuron}_rank{rank}.pkl')
    with open(kernel_file, 'wb') as f:
        pickle.dump(kernel_data, f)
    print(f'Saved kernels at {kernel_file}')

    # Generate individual fit plot
    fig = plt.figure(figsize=(18, 18))
    plt.suptitle(f'Fit for L={L} - Input: {input_neuron}, Output: {output_neuron}, '
                 f'alpha_k={alpha_k:.4f}, alpha_h={alpha_h:.4f}, k0={k0:.3f}, Rank={rank}')
    
    T_full = x_full.shape[-1]
    ff_basis_batched = ff_basis.unsqueeze(0).permute(2, 0, 1).flip(dims=[-1])
    fb_basis_batched = fb_basis.unsqueeze(0).permute(2, 0, 1).flip(dims=[-1])
    X_full_file = os.path.join(SAVE_DIR, f'X_full_temp_plot_rank{rank}.npy')
    X_full = np.memmap(X_full_file, dtype='float32', mode='w+', shape=(T_full, 2 * L + 1))
    X_full[:, 0] = 1
    
    def process_chunks_to_design(input_x, input_y, basis_ff, basis_fb, total_length, chunk_size, design_matrix):
        step_size = chunk_size
        num_chunks = (total_length + step_size - 1) // step_size
        padding = basis_ff.shape[-1] - 1
        for i in range(num_chunks):
            start = i * step_size
            end = min(start + step_size + padding, total_length)
            chunk_x = input_x[:, :, start:end]
            chunk_y = input_y[:, :, start:end]
            if chunk_x.shape[-1] < padding + 1:
                chunk_x = F.pad(chunk_x, (padding + 1 - chunk_x.shape[-1], 0), mode='constant', value=0)
                chunk_y = F.pad(chunk_y, (padding + 1 - chunk_y.shape[-1], 0), mode='constant', value=0)
            
            ff_conv = F.conv1d(chunk_x, basis_ff, padding=padding)[:, :, :-padding + 1]
            fb_conv = F.conv1d(chunk_y, basis_fb, padding=padding)[:, :, :-padding + 1]
            
            valid_length = total_length - start if i == num_chunks - 1 else step_size
            
            ff_conv_chunk = ff_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
            fb_conv_chunk = fb_conv[:, :, :valid_length].transpose(1, 2)[0].cpu().numpy()
            design_matrix[start:start + valid_length, 1:L + 1] = ff_conv_chunk
            design_matrix[start:start + valid_length, L + 1:] = fb_conv_chunk
    
    process_chunks_to_design(x_full, y_full, ff_basis_batched, fb_basis_batched, T_full, 50000, X_full)
    Pb_full = model_full.predict(X_full)
    y_full_np = y_full.flatten().cpu().numpy()
    ks_full = time_rescaling_ks(Pb_full, y_full_np)
    
    del X_full
    os.remove(X_full_file)

    def get_max_nonzero_idx(data, axis=None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if axis is not None:
            return np.max([np.max(np.where(np.abs(data[:, j]) > 0.001)[0]) + 1 
                           if np.any(np.abs(data[:, j]) > 0.001) else 0 
                           for j in range(data.shape[1])])
        else:
            return np.max(np.where(np.abs(data) > 0.001)[0]) + 1 if np.any(np.abs(data) > 0.001) else len(data)

    max_nonzero_idx_ff_basis = get_max_nonzero_idx(ff_basis, axis=1)
    max_nonzero_idx_fb_basis = get_max_nonzero_idx(fb_basis, axis=1)
    max_nonzero_idx_ff_kernel = get_max_nonzero_idx(ff_kernel_avg.squeeze())
    max_nonzero_idx_fb_kernel = get_max_nonzero_idx(fb_kernel_avg.squeeze())
    global_max_nonzero_idx = max(max_nonzero_idx_ff_basis, max_nonzero_idx_fb_basis, 
                                 max_nonzero_idx_ff_kernel, max_nonzero_idx_fb_kernel)

    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.5], hspace=0.4, wspace=0.4)
    ax_ff = fig.add_subplot(gs[0, 0])
    ax_ff.plot(ff_kernel_avg.squeeze(), label='FF Kernel', color='blue')
    ax_ff.set_title('Feedforward Kernel')
    ax_ff.set_xlabel('Time (ms)')
    ax_ff.set_ylabel('Amplitude')
    ax_ff.legend(loc='upper right')
    ax_ff.set_xlim(0, global_max_nonzero_idx)
    ax_ff.set_box_aspect(1)

    ax_fb = fig.add_subplot(gs[1, 0])
    ax_fb.plot(fb_kernel_avg.squeeze(), label='FB Kernel', color='blue')
    ax_fb.set_title('Feedback Kernel')
    ax_fb.set_xlabel('Time (ms)')
    ax_fb.set_ylabel('Amplitude')
    ax_fb.legend(loc='upper right')
    ax_fb.set_xlim(0, global_max_nonzero_idx)
    ax_fb.set_box_aspect(1)

    ax_ks = fig.add_subplot(gs[2, 0])
    ax_ks.step(ks_full[2], ks_full[1], where='post', label="Empirical CDF", color="black")
    ax_ks.step(ks_full[2], ks_full[2], where='post', label="Uniform CDF", color="red", linestyle='--')
    ax_ks.fill_between(ks_full[2], ks_full[3][:, 0], ks_full[3][:, 1], color="gray", alpha=0.3, label="95% CI")
    ax_ks.set_xlabel("Theoretical Quantiles")
    ax_ks.set_ylabel("Transformed Values")
    ax_ks.set_title(f"KS Score: {ks_full[0]:.3f}")
    ax_ks.legend()
    ax_ks.set_box_aspect(1)

    x_1d = x_full.flatten().numpy()
    y_1d = y_full.flatten().numpy()

    def compute_correlogram(x, y, max_lag_ms, bin_size, mode):
        n_original = len(x)
        resampling_factor = int(bin_size / 1)  # Assuming original data is in 1ms bins
        n_resampled = n_original // resampling_factor
        x_resampled = np.array([np.sum(x[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
        y_resampled = np.array([np.sum(y[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
        
        x_resampled = (x_resampled > 0).astype(float)
        y_resampled = (y_resampled > 0).astype(float)
        
        max_lag_bins = int(max_lag_ms / bin_size)
        corr = scipy.signal.correlate(y_resampled, x_resampled, mode='full')
        center = len(corr) // 2
        start_idx = center - max_lag_bins
        end_idx = center + max_lag_bins + 1
        corr_trimmed = corr[start_idx:end_idx]
        
        if mode == 'auto':
            corr_trimmed[max_lag_bins] = 0
        
        lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
        return lags, corr_trimmed

    lags_x_1000, autocorr_x_1000 = compute_correlogram(x_1d, x_1d, 1000, 80, mode='auto')
    lags_y_1000, autocorr_y_1000 = compute_correlogram(y_1d, y_1d, 1000, 80, mode='auto')
    lags_xy_1000, crosscorr_xy_1000 = compute_correlogram(x_1d, y_1d, 1000, 80, mode='cross')
    lags_x_100, autocorr_x_100 = compute_correlogram(x_1d, x_1d, 100, 4, mode='auto')
    lags_y_100, autocorr_y_100 = compute_correlogram(y_1d, y_1d, 100, 4, mode='auto')
    lags_xy_100, crosscorr_xy_100 = compute_correlogram(x_1d, y_1d, 100, 4, mode='cross')

    gs_corr_full = gs[0:3, 1].subgridspec(3, 1, hspace=0.3)
    ax_corr1 = fig.add_subplot(gs_corr_full[0])
    ax_corr1.bar(lags_x_1000[:-1], autocorr_x_1000[:-1], width=np.diff(lags_x_1000), color='blue', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr1.set_ylabel('x_ac')
    ax_corr1.set_title('Full Correlograms (±1000ms)')
    ax_corr1.set_box_aspect(1)

    ax_corr2 = fig.add_subplot(gs_corr_full[1])
    ax_corr2.bar(lags_y_1000[:-1], autocorr_y_1000[:-1], width=np.diff(lags_y_1000), color='green', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr2.set_ylabel('y_ac')
    ax_corr2.set_box_aspect(1)

    ax_corr3 = fig.add_subplot(gs_corr_full[2])
    ax_corr3.bar(lags_xy_1000[:-1], crosscorr_xy_1000[:-1], width=np.diff(lags_xy_1000), color='red', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr3.set_xlabel('Lag (ms)')
    ax_corr3.set_ylabel('cc')
    ax_corr3.set_box_aspect(1)

    gs_corr_zoom = gs[0:3, 2].subgridspec(3, 1, hspace=0.3)
    ax_corr1_zoom = fig.add_subplot(gs_corr_zoom[0])
    ax_corr1_zoom.bar(lags_x_100[:-1], autocorr_x_100[:-1], width=np.diff(lags_x_100), color='blue', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr1_zoom.set_ylabel('x_ac')
    ax_corr1_zoom.set_title('Zoomed Correlograms (±100ms)')
    ax_corr1_zoom.set_box_aspect(1)

    ax_corr2_zoom = fig.add_subplot(gs_corr_zoom[1])
    ax_corr2_zoom.bar(lags_y_100[:-1], autocorr_y_100[:-1], width=np.diff(lags_y_100), color='green', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr2_zoom.set_ylabel('y_ac')
    ax_corr2_zoom.set_box_aspect(1)

    ax_corr3_zoom = fig.add_subplot(gs_corr_zoom[2])
    ax_corr3_zoom.bar(lags_xy_100[:-1], crosscorr_xy_100[:-1], width=np.diff(lags_xy_100), color='red', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr3_zoom.set_xlabel('Lag (ms)')
    ax_corr3_zoom.set_ylabel('cc')
    ax_corr3_zoom.set_box_aspect(1)

    plot_length = x_full.shape[-1]
    x_spikes = torch.where(x_full[0, 0, :] == 1)[0].cpu().numpy()
    y_spikes = torch.where(y_full[0, 0, :] == 1)[0].cpu().numpy()

    bin_size = 1000
    bins = np.arange(0, plot_length + bin_size, bin_size)
    x_hist, _ = np.histogram(x_spikes, bins=bins)
    y_hist, _ = np.histogram(y_spikes, bins=bins)
    x_rate = x_hist / (bin_size / 1000)
    y_rate = y_hist / (bin_size / 1000)
    x_rate_smooth = gaussian_filter1d(x_rate, sigma=1)
    y_rate_smooth = gaussian_filter1d(y_rate, sigma=1)
    x_spike_rates = np.interp(x_spikes, bins[:-1], x_rate_smooth)
    y_spike_rates = np.interp(y_spikes, bins[:-1], y_rate_smooth)

    raster_gs = gs[3, :].subgridspec(1, 1, height_ratios=[0.05])  # Small height for raster
    ax_raster = fig.add_subplot(raster_gs[0])
    scatter_x = ax_raster.scatter(x_spikes, np.ones_like(x_spikes) * 0.2, c=x_spike_rates, cmap='Blues', label='Input (X) Spikes', s=20, vmin=0)
    scatter_y = ax_raster.scatter(y_spikes, np.ones_like(y_spikes) * 0.1, c=y_spike_rates, cmap='Reds', label='Output (Y) Spikes', s=20, vmin=0)
    ax_raster.set_yticks([0.1, 0.2])  # Adjusted to match new y-values
    ax_raster.set_yticklabels(['Y', 'X'])
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_title('Raster Plot with Temporal Firing Rate', fontsize=14)
    ax_raster.set_xlim(0, plot_length)
    ax_raster.set_ylim(0.05, 0.25)  # Tight range to keep them close
    ax_raster.grid(True, alpha=0.3)
    pos = ax_raster.get_position()  # Get current position (x0, y0, width, height)
    new_height = 0.05  # Desired height in figure coordinates (e.g., 5% of figure height)
    ax_raster.set_position([pos.x0, pos.y0, pos.width, new_height])  # Set new height
    
    divider = make_axes_locatable(ax_raster)
    cax1 = divider.append_axes("right", size="0.5%", pad=0.05)
    cax2 = divider.append_axes("right", size="0.5%", pad=0.05)
    cbar_x = plt.colorbar(scatter_x, cax=cax1)
    cbar_y = plt.colorbar(scatter_y, cax=cax2)
    


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(SAVE_DIR, f'fit_L{L}_input{input_neuron}_output{output_neuron}_ak{alpha_k:.4f}_ah{alpha_h:.4f}_rank{rank}.png')
    plt.savefig(filename)
    plt.close()

    print(f'Saved plot at {filename}')

def plot_fit(data, model_full, x_full, y_full, model_fsiso=None, save_dir=''):
    '''
    data = {
        'SISO': {
            'input_neuron': input_neuron,
            'output_neuron': output_neuron,
            'rank': rank,
            'ff_kernel': ff_kernel_avg,
            'fb_kernel': fb_kernel_avg,
            'L': L,
            'alpha_k': alpha_k,
            'alpha_h': alpha_h,
            'k0': k0,
            'all_metrics': all_metrics  # Contains 'fold_metrics' with fold-specific 'ff_kernel' and 'fb_kernel'
        },
        'F_SISO': {  # this item is optional
            'input_neuron': input_neuron,
            'output_neuron': output_neuron,
            'rank': rank,
            'ff_kernel': ff_kernel_avg,
            'fb_kernel': fb_kernel_avg,
            'L': L,
            'alpha_k': alpha_k,
            'alpha_h': alpha_h,
            'k0': k0,
            'all_metrics': all_metrics  # Contains 'fold_metrics' with fold-specific 'ff_kernel' and 'fb_kernel'
        }
    }
    Without F_SISO: 4x3 grid, last row (row 3) is full-width raster.
    With F_SISO: 4x4 grid, SISO in col 0, F_SISO in col 1, correlograms in cols 2-3, last row (row 3) is full-width raster.
    Plots FF and FB kernels with 95% CI shading based on fold-specific kernels from all_metrics['fold_metrics'].
    '''

    siso_data = data['SISO']
    has_fsiso = 'F_SISO' in data and model_fsiso is not None
    fsiso_data = data.get('F_SISO', None)

    # Extract SISO parameters
    input_neuron = siso_data['input_neuron']
    output_neuron = siso_data['output_neuron']
    rank = siso_data['rank']
    ff_kernel_siso = siso_data['ff_kernel']
    fb_kernel_siso = siso_data['fb_kernel']
    L = siso_data['L']
    alpha_k = siso_data['alpha_k']
    alpha_h = siso_data['alpha_h']
    if siso_data['k0'] is not None:
        k0_siso = siso_data['k0']
    else:
        k0_siso = 0

    # Extract fold-specific kernels for SISO
    siso_fold_metrics = siso_data['all_metrics'].get('fold_metrics', [])
    ff_kernels_siso_folds = np.array([m['ff_kernel'] for m in siso_fold_metrics if 'ff_kernel' in m])
    fb_kernels_siso_folds = np.array([m['fb_kernel'] for m in siso_fold_metrics if 'fb_kernel' in m])

    # Compute CI for SISO
    if len(ff_kernels_siso_folds) > 0:
        ff_mean_siso = ff_kernel_siso.squeeze()
        ff_std_siso = np.std(ff_kernels_siso_folds, axis=0)
        ff_ci_siso = 1.96 * ff_std_siso / np.sqrt(len(ff_kernels_siso_folds))  # 95% CI
    else:
        ff_mean_siso = ff_kernel_siso.squeeze()
        ff_ci_siso = np.zeros_like(ff_mean_siso)

    if len(fb_kernels_siso_folds) > 0:
        fb_mean_siso = fb_kernel_siso.squeeze()
        fb_std_siso = np.std(fb_kernels_siso_folds, axis=0)
        fb_ci_siso = 1.96 * fb_std_siso / np.sqrt(len(fb_kernels_siso_folds))  # 95% CI
    else:
        fb_mean_siso = fb_kernel_siso.squeeze()
        fb_ci_siso = np.zeros_like(fb_mean_siso)

    # Extract F_SISO parameters if present
    if has_fsiso:
        ff_kernel_fsiso = fsiso_data['ff_kernel']
        fb_kernel_fsiso = fsiso_data['fb_kernel']
        k0_fsiso = fsiso_data['k0']

        # Extract fold-specific kernels for F_SISO
        fsiso_fold_metrics = fsiso_data['all_metrics'].get('fold_metrics', [])
        ff_kernels_fsiso_folds = np.array([m['ff_kernel'] for m in fsiso_fold_metrics if 'ff_kernel' in m])
        fb_kernels_fsiso_folds = np.array([m['fb_kernel'] for m in fsiso_fold_metrics if 'fb_kernel' in m])

        # Compute CI for F_SISO
        if len(ff_kernels_fsiso_folds) > 0:
            ff_mean_fsiso = ff_kernel_fsiso.squeeze()
            ff_std_fsiso = np.std(ff_kernels_fsiso_folds, axis=0)
            ff_ci_fsiso = 1.96 * ff_std_fsiso / np.sqrt(len(ff_kernels_fsiso_folds))  # 95% CI
        else:
            ff_mean_fsiso = ff_kernel_fsiso.squeeze()
            ff_ci_fsiso = np.zeros_like(ff_mean_fsiso)

        if len(fb_kernels_fsiso_folds) > 0:
            fb_mean_fsiso = fb_kernel_fsiso.squeeze()
            fb_std_fsiso = np.std(fb_kernels_fsiso_folds, axis=0)
            fb_ci_fsiso = 1.96 * fb_std_fsiso / np.sqrt(len(fb_kernels_fsiso_folds))  # 95% CI
        else:
            fb_mean_fsiso = fb_kernel_fsiso.squeeze()
            fb_ci_fsiso = np.zeros_like(fb_mean_fsiso)

    # Set up figure with dynamic columns
    n_cols = 4 if has_fsiso else 3
    fig = plt.figure(figsize=(6 * n_cols, 18))
    if has_fsiso:
        plt.suptitle(f'Fit for L={L} - Input: {input_neuron}, Output: {output_neuron}, '
                     f'alpha_k={alpha_k:.4f}, alpha_h={alpha_h:.4f}, k0={k0_siso:.3f}/{k0_fsiso:.3f}, Rank={rank}')
    else:
        plt.suptitle(f'Fit for L={L} - Input: {input_neuron}, Output: {output_neuron}, '
                     f'alpha_k={alpha_k:.4f}, alpha_h={alpha_h:.4f}, k0={k0_siso:.3f}, Rank={rank}')

    # Define grid spec: 4 rows, last row spans all columns
    gs = fig.add_gridspec(4, n_cols, height_ratios=[1, 1, 1, 0.5], hspace=0.4, wspace=0.4)

    # KS data
    ks_full_siso = data['SISO']['all_metrics'].get('KS_full', None)
    ks_full_fsiso = data['F_SISO']['all_metrics'].get('KS_full', None) if has_fsiso else None

    # Determine max nonzero index for kernel plotting
    def get_max_nonzero_idx(data, axis=None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if axis is not None:
            return np.max([np.max(np.where(np.abs(data[:, j]) > 0.001)[0]) + 1 
                           if np.any(np.abs(data[:, j]) > 0.001) else 0 
                           for j in range(data.shape[1])])
        else:
            return np.max(np.where(np.abs(data) > 0.001)[0]) + 1 if np.any(np.abs(data) > 0.001) else len(data)

    max_nonzero_idx_ff_siso = get_max_nonzero_idx(ff_kernel_siso.squeeze())
    max_nonzero_idx_fb_siso = get_max_nonzero_idx(fb_kernel_siso.squeeze())
    global_max_nonzero_idx = max(max_nonzero_idx_ff_siso, max_nonzero_idx_fb_siso)
    if has_fsiso:
        max_nonzero_idx_ff_fsiso = get_max_nonzero_idx(ff_kernel_fsiso.squeeze())
        max_nonzero_idx_fb_fsiso = get_max_nonzero_idx(fb_kernel_fsiso.squeeze())
        global_max_nonzero_idx = max(global_max_nonzero_idx, max_nonzero_idx_ff_fsiso, max_nonzero_idx_fb_fsiso)

    # Time axis for kernel plots
    t = np.arange(len(ff_mean_siso))

    # Plot SISO kernels and KS (column 0)
    ax_ff_siso = fig.add_subplot(gs[0, 0])
    ax_ff_siso.plot(t, ff_mean_siso, label=f'k0={k0_siso:.3f}', color='blue')
    ax_ff_siso.fill_between(t, ff_mean_siso - ff_ci_siso, ff_mean_siso + ff_ci_siso, color='blue', alpha=0.2, label='95% CI')
    ax_ff_siso.set_title('Feedforward Kernel (SISO)')
    ax_ff_siso.set_xlabel('Time (ms)')
    ax_ff_siso.set_ylabel('Amplitude')
    ax_ff_siso.legend(loc='upper right')
    ax_ff_siso.set_xlim(0, global_max_nonzero_idx)
    ax_ff_siso.set_box_aspect(1)

    ax_fb_siso = fig.add_subplot(gs[1, 0])
    ax_fb_siso.plot(t, fb_mean_siso, color='blue')
    ax_fb_siso.fill_between(t, fb_mean_siso - fb_ci_siso, fb_mean_siso + fb_ci_siso, color='blue', alpha=0.2, label='95% CI')
    ax_fb_siso.set_title('Feedback Kernel (SISO)')
    ax_fb_siso.set_xlabel('Time (ms)')
    ax_fb_siso.set_ylabel('Amplitude')
    ax_fb_siso.legend(loc='upper right')
    ax_fb_siso.set_xlim(0, global_max_nonzero_idx)
    ax_fb_siso.set_box_aspect(1)

    ax_ks_siso = fig.add_subplot(gs[2, 0])
    if ks_full_siso:
        ax_ks_siso.step(ks_full_siso[2], ks_full_siso[1], where='post', label="Empirical CDF", color="black")
        ax_ks_siso.step(ks_full_siso[2], ks_full_siso[2], where='post', label="Uniform CDF", color="red", linestyle='--')
        ax_ks_siso.fill_between(ks_full_siso[2], ks_full_siso[3][:, 0], ks_full_siso[3][:, 1], color="gray", alpha=0.3, label="95% CI")
        ax_ks_siso.set_title(f"KS Score (SISO): {ks_full_siso[0]:.3f}")
    ax_ks_siso.set_xlabel("Theoretical Quantiles")
    ax_ks_siso.set_ylabel("Transformed Values")
    ax_ks_siso.legend()
    ax_ks_siso.set_box_aspect(1)

    # Plot F_SISO kernels and KS if present (column 1); share y-axis with SISO for FF and FB
    if has_fsiso:
        ax_ff_fsiso = fig.add_subplot(gs[0, 1], sharey=ax_ff_siso)
        ax_ff_fsiso.plot(t, ff_mean_fsiso, label=f'k0={k0_fsiso:.3f}', color='purple')
        ax_ff_fsiso.fill_between(t, ff_mean_fsiso - ff_ci_fsiso, ff_mean_fsiso + ff_ci_fsiso, color='purple', alpha=0.2, label='95% CI')
        ax_ff_fsiso.set_title('Feedforward Kernel (F_SISO)')
        ax_ff_fsiso.set_xlabel('Time (ms)')
        ax_ff_fsiso.set_ylabel('Amplitude')
        ax_ff_fsiso.legend(loc='upper right')
        ax_ff_fsiso.set_xlim(0, global_max_nonzero_idx)
        ax_ff_fsiso.set_box_aspect(1)
        plt.setp(ax_ff_fsiso.get_yticklabels(), visible=True)  # show y labels on shared axis

        ax_fb_fsiso = fig.add_subplot(gs[1, 1], sharey=ax_fb_siso)
        ax_fb_fsiso.plot(t, fb_mean_fsiso, color='purple')
        ax_fb_fsiso.fill_between(t, fb_mean_fsiso - fb_ci_fsiso, fb_mean_fsiso + fb_ci_fsiso, color='purple', alpha=0.2, label='95% CI')
        ax_fb_fsiso.set_title('Feedback Kernel (F_SISO)')
        ax_fb_fsiso.set_xlabel('Time (ms)')
        ax_fb_fsiso.set_ylabel('Amplitude')
        ax_fb_fsiso.legend(loc='upper right')
        ax_fb_fsiso.set_xlim(0, global_max_nonzero_idx)
        ax_fb_fsiso.set_box_aspect(1)
        plt.setp(ax_fb_fsiso.get_yticklabels(), visible=True)  # show y labels on shared axis

        ax_ks_fsiso = fig.add_subplot(gs[2, 1])
        if ks_full_fsiso:
            ax_ks_fsiso.step(ks_full_fsiso[2], ks_full_fsiso[1], where='post', label="Empirical CDF", color="black")
            ax_ks_fsiso.step(ks_full_fsiso[2], ks_full_fsiso[2], where='post', label="Uniform CDF", color="red", linestyle='--')
            ax_ks_fsiso.fill_between(ks_full_fsiso[2], ks_full_fsiso[3][:, 0], ks_full_fsiso[3][:, 1], color="gray", alpha=0.3, label="95% CI")
            ax_ks_fsiso.set_title(f"KS Score (F_SISO): {ks_full_fsiso[0]:.3f}")
        ax_ks_fsiso.set_xlabel("Theoretical Quantiles")
        ax_ks_fsiso.set_ylabel("Transformed Values")
        ax_ks_fsiso.legend()
        ax_ks_fsiso.set_box_aspect(1)

    # Correlograms (columns 1-2 without F_SISO, 2-3 with F_SISO)
    corr_col_start = 1 if not has_fsiso else 2
    x_1d = x_full.flatten().numpy()
    y_1d = y_full.flatten().numpy()

    def compute_correlogram(x, y, max_lag_ms, bin_size, mode):
        n_original = len(x)
        resampling_factor = int(bin_size / 1)  # Assuming original data is in 1ms bins
        n_resampled = n_original // resampling_factor
        x_resampled = np.array([np.sum(x[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
        y_resampled = np.array([np.sum(y[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
        x_resampled = (x_resampled > 0).astype(float)
        y_resampled = (y_resampled > 0).astype(float)
        max_lag_bins = int(max_lag_ms / bin_size)
        corr = scipy.signal.correlate(y_resampled, x_resampled, mode='full')
        center = len(corr) // 2
        start_idx = center - max_lag_bins
        end_idx = center + max_lag_bins + 1
        corr_trimmed = corr[start_idx:end_idx]
        if mode == 'auto':
            corr_trimmed[max_lag_bins] = 0
        lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
        return lags, corr_trimmed

    lags_x_1000, autocorr_x_1000 = compute_correlogram(x_1d, x_1d, 1000, 80, mode='auto')
    lags_y_1000, autocorr_y_1000 = compute_correlogram(y_1d, y_1d, 1000, 80, mode='auto')
    lags_xy_1000, crosscorr_xy_1000 = compute_correlogram(x_1d, y_1d, 1000, 80, mode='cross')
    lags_x_100, autocorr_x_100 = compute_correlogram(x_1d, x_1d, 100, 4, mode='auto')
    lags_y_100, autocorr_y_100 = compute_correlogram(y_1d, y_1d, 100, 4, mode='auto')
    lags_xy_100, crosscorr_xy_100 = compute_correlogram(x_1d, y_1d, 100, 4, mode='cross')

    gs_corr_full = gs[0:3, corr_col_start].subgridspec(3, 1, hspace=0.3)
    ax_corr1 = fig.add_subplot(gs_corr_full[0])
    ax_corr1.bar(lags_x_1000[:-1], autocorr_x_1000[:-1], width=np.diff(lags_x_1000), color='blue', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr1.set_ylabel('x_ac')
    ax_corr1.set_title('Full Correlograms (±1000ms)')
    ax_corr1.set_box_aspect(1)

    ax_corr2 = fig.add_subplot(gs_corr_full[1])
    ax_corr2.bar(lags_y_1000[:-1], autocorr_y_1000[:-1], width=np.diff(lags_y_1000), color='green', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr2.set_ylabel('y_ac')
    ax_corr2.set_box_aspect(1)

    ax_corr3 = fig.add_subplot(gs_corr_full[2])
    ax_corr3.bar(lags_xy_1000[:-1], crosscorr_xy_1000[:-1], width=np.diff(lags_xy_1000), color='red', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr3.set_xlabel('Lag (ms)')
    ax_corr3.set_ylabel('cc')
    ax_corr3.set_box_aspect(1)

    gs_corr_zoom = gs[0:3, corr_col_start + 1].subgridspec(3, 1, hspace=0.3)
    ax_corr1_zoom = fig.add_subplot(gs_corr_zoom[0])
    ax_corr1_zoom.bar(lags_x_100[:-1], autocorr_x_100[:-1], width=np.diff(lags_x_100), color='blue', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr1_zoom.set_ylabel('x_ac')
    ax_corr1_zoom.set_title('Zoomed Correlograms (±100ms)')
    ax_corr1_zoom.set_box_aspect(1)

    ax_corr2_zoom = fig.add_subplot(gs_corr_zoom[1])
    ax_corr2_zoom.bar(lags_y_100[:-1], autocorr_y_100[:-1], width=np.diff(lags_y_100), color='green', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr2_zoom.set_ylabel('y_ac')
    ax_corr2_zoom.set_box_aspect(1)

    ax_corr3_zoom = fig.add_subplot(gs_corr_zoom[2])
    ax_corr3_zoom.bar(lags_xy_100[:-1], crosscorr_xy_100[:-1], width=np.diff(lags_xy_100), color='red', alpha=0.8, linewidth=0.5, align='edge')
    ax_corr3_zoom.set_xlabel('Lag (ms)')
    ax_corr3_zoom.set_ylabel('cc')
    ax_corr3_zoom.set_box_aspect(1)

    # Raster plot (full width, row 3)
    plot_length = x_full.shape[-1]
    x_spikes = torch.where(x_full[0, 0, :] == 1)[0].cpu().numpy()
    y_spikes = torch.where(y_full[0, 0, :] == 1)[0].cpu().numpy()
    bin_size = 1000
    bins = np.arange(0, plot_length + bin_size, bin_size)
    x_hist, _ = np.histogram(x_spikes, bins=bins)
    y_hist, _ = np.histogram(y_spikes, bins=bins)
    x_rate = x_hist / (bin_size / 1000)
    y_rate = y_hist / (bin_size / 1000)
    x_rate_smooth = gaussian_filter1d(x_rate, sigma=1)
    y_rate_smooth = gaussian_filter1d(y_rate, sigma=1)
    x_spike_rates = np.interp(x_spikes, bins[:-1], x_rate_smooth)
    y_spike_rates = np.interp(y_spikes, bins[:-1], y_rate_smooth)

    ax_raster = fig.add_subplot(gs[3, :])  # Spans all columns in row 3
    scatter_x = ax_raster.scatter(x_spikes, np.ones_like(x_spikes) * 0.2, c=x_spike_rates, cmap='Blues', label='Input (X) Spikes', s=20, vmin=0)
    scatter_y = ax_raster.scatter(y_spikes, np.ones_like(y_spikes) * 0.1, c=y_spike_rates, cmap='Reds', label='Output (Y) Spikes', s=20, vmin=0)
    ax_raster.set_yticks([0.1, 0.2])
    ax_raster.set_yticklabels(['Y', 'X'])
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_title('Raster Plot with Temporal Firing Rate', fontsize=14)
    ax_raster.set_xlim(0, plot_length)
    ax_raster.set_ylim(0.05, 0.25)
    ax_raster.grid(True, alpha=0.3)
    pos = ax_raster.get_position()
    new_height = 0.05
    ax_raster.set_position([pos.x0, pos.y0, pos.width, new_height])

    divider = make_axes_locatable(ax_raster)
    cax1 = divider.append_axes("right", size="0.5%", pad=0.05)
    cax2 = divider.append_axes("right", size="0.5%", pad=0.05)
    cbar_x = plt.colorbar(scatter_x, cax=cax1)
    cbar_y = plt.colorbar(scatter_y, cax=cax2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename = os.path.join(save_dir, f'fit_L{L}_input{input_neuron}_output{output_neuron}_ak{alpha_k:.4f}_ah{alpha_h:.4f}_rank{rank}.png')
    plt.savefig(filename)
    plt.close()

    print(f'Saved plot at {filename}')
    return filename

def overlay_kernels(save_dir):
    kernel_files = [f for f in os.listdir(save_dir) if f.startswith('kernels_') and f.endswith('.pkl')]
    ff_kernels = []
    fb_kernels = []
    k0_list = []
    labels = []
    k0_values = []

    max_length = 0
    for kernel_file in kernel_files:
        with open(os.path.join(save_dir, kernel_file), 'rb') as f:
            kernel_data = pickle.load(f)
        ff_kernels.append(kernel_data['SISO']['ff_kernel'])
        fb_kernels.append(kernel_data['SISO']['fb_kernel'])

        k0_list.append(kernel_data['k0'])
        labels.append(f"{kernel_data['input_neuron']}-{kernel_data['output_neuron']} (Rank {kernel_data['rank']})")
        max_length = max(max_length, len(kernel_data['ff_kernel']), len(kernel_data['fb_kernel']))

    # Pad kernels to the same length
    for i in range(len(ff_kernels)):
        ff_kernels[i] = np.pad(ff_kernels[i], (0, max_length - len(ff_kernels[i])), mode='constant', constant_values=0)
        fb_kernels[i] = np.pad(fb_kernels[i], (0, max_length - len(fb_kernels[i])), mode='constant', constant_values=0)

    # Normalize kernels by absolute value of k0
    ff_kernels_normalized = []
    fb_kernels_normalized = []
    for i, kernel_file in enumerate(kernel_files):
        with open(os.path.join(save_dir, kernel_file), 'rb') as f:
            kernel_data = pickle.load(f)
        # Assuming k0 is stored in metrics or needs to be approximated; here we use max absolute value as a proxy
        ff_kernels_normalized.append(ff_kernels[i] / abs(k0_list[i]))
        fb_kernels_normalized.append(fb_kernels[i] / abs(k0_list[i]))

    # Create two figures
    time = np.arange(max_length)

    # Figure 1: Original Kernels
    fig1, (ax_ff1, ax_fb1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ff_kernel, label in zip(ff_kernels, labels):
        ax_ff1.plot(time, ff_kernel, label=label, alpha=0.5)
    ax_ff1.set_title('Feedforward Kernels Overlay (Original)')
    ax_ff1.set_ylabel('Amplitude')
    ax_ff1.legend(loc='upper right', fontsize='small')
    ax_ff1.grid(True, alpha=0.3)

    for fb_kernel, label in zip(fb_kernels, labels):
        ax_fb1.plot(time, fb_kernel, label=label, alpha=0.5)
    ax_fb1.set_title('Feedback Kernels Overlay (Original)')
    ax_fb1.set_xlabel('Time (ms)')
    ax_fb1.set_ylabel('Amplitude')
    ax_fb1.legend(loc='upper right', fontsize='small')
    ax_fb1.grid(True, alpha=0.3)

    plt.tight_layout()
    overlay_file_original = os.path.join(save_dir, f'kernel_overlay_original_L{L}_ak{alpha_k:.4f}_ah{alpha_h:.4f}.png')
    fig1.savefig(overlay_file_original)
    plt.close(fig1)
    print(f'Saved original kernel overlay plot at {overlay_file_original}')

    # Figure 2: Normalized Kernels
    fig2, (ax_ff2, ax_fb2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ff_kernel_norm, label in zip(ff_kernels_normalized, labels):
        ax_ff2.plot(time, ff_kernel_norm, label=label, alpha=0.5)
    ax_ff2.set_title('Feedforward Kernels Overlay (Normalized by Abs Max)')
    ax_ff2.set_ylabel('Normalized Amplitude')
    ax_ff2.legend(loc='upper right', fontsize='small')
    ax_ff2.grid(True, alpha=0.3)

    for fb_kernel_norm, label in zip(fb_kernels_normalized, labels):
        ax_fb2.plot(time, fb_kernel_norm, label=label, alpha=0.5)
    ax_fb2.set_title('Feedback Kernels Overlay (Normalized by Abs Max)')
    ax_fb2.set_xlabel('Time (ms)')
    ax_fb2.set_ylabel('Normalized Amplitude')
    ax_fb2.legend(loc='upper right', fontsize='small')
    ax_fb2.grid(True, alpha=0.3)

    plt.tight_layout()
    overlay_file_normalized = os.path.join(save_dir, f'kernel_overlay_normalized_L{L}_ak{alpha_k:.4f}_ah{alpha_h:.4f}.png')
    fig2.savefig(overlay_file_normalized)
    plt.close(fig2)
    print(f'Saved normalized kernel overlay plot at {overlay_file_normalized}')

def run_glm_fitting():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(SAVE_DIR, 'glm_fitting.log')), logging.StreamHandler()]
    )
    logger = logging.getLogger('glm_fitting')

    logger.info(f"Running GLM fitting with fixed parameters: L={L}, alpha_k={alpha_k:.4f}, alpha_h={alpha_h:.4f}")
    print(f"\n{'#'*60}")
    print(f"GLM Fitting Configuration:")
    print(f"  L={L}, alpha_k={alpha_k:.4f}, alpha_h={alpha_h:.4f}")
    print(f"  max_tau={max_tau}, folds={NUMBER_OF_FOLDS}, chunk_size={cmd_args.chunk_size if cmd_args else 10000}")
    print(f"  Processing {len(neuron_pairs)} neuron pair(s)")
    print(f"{'#'*60}\n")
    print_memory_status("Initial ")
    
    for pair_idx, (input_neuron, output_neuron, bump_score, rank) in enumerate(tqdm(neuron_pairs, desc="Neuron Pairs", total=len(neuron_pairs)), 1):
        print(f"\n[{pair_idx}/{len(neuron_pairs)}] Processing pair: {input_neuron} -> {output_neuron} (Rank {rank})")
        logger.info(f"Processing pair: {input_neuron} -> {output_neuron}, Rank: {rank}, BumpScore: {bump_score}")
        
        try:
            pair_data = preprocess_neuron_data(input_neuron, output_neuron, neurons, sample_rate)
            model_full, conf_int, all_metrics, avg_ff_coeffs, avg_fb_coeffs,k0, ff_basis, fb_basis = fit_glm_to_stdp(
                pair_data[0], pair_data[1], L, alpha_k, alpha_h,input_neuron, output_neuron,rank,max_tau=max_tau, folds=NUMBER_OF_FOLDS)
            # Print object sizes for memory usage tracking
            if f_SISO:
                print(f"\n[{pair_idx}/{len(neuron_pairs)}] Processing reverse direction: {output_neuron} -> {input_neuron} (Rank {rank})")
                print_memory_status("Before reverse fit ")
                pair_data_reverse = preprocess_neuron_data(output_neuron, input_neuron, neurons, sample_rate)
                model_full_reverse, conf_int_reverse, all_metrics_reverse, avg_ff_coeffs_reverse, avg_fb_coeffs_reverse,k0_reverse, ff_basis_reverse, fb_basis_reverse = fit_glm_to_stdp(
                    pair_data_reverse[0], pair_data_reverse[1], L, alpha_k, alpha_h,output_neuron, input_neuron,rank,max_tau=max_tau, folds=NUMBER_OF_FOLDS)
                print_memory_status("After reverse fit ")
            
            if model_full is None:
                logger.error(f"GLM fitting failed for {input_neuron} -> {output_neuron}")
                continue
            
            # Save metrics to CSV
            pair_metrics_file = os.path.join(SAVE_DIR, f'pair_metrics_{input_neuron}_{output_neuron}_rank{rank}.csv')
            if  os.path.exists(pair_metrics_file):
                os.remove(pair_metrics_file)
            with open(pair_metrics_file, 'a', newline='') as csvfile:
                fieldnames = ['input_neuron', 'output_neuron', 'rank', 'bump_score', 'fold', 'L', 'alpha_k', 'alpha_h', 'norm_ce_val', 'rmse_train',
                            'k0', 'deviance_reduction', 'log_likelihood', 'null_log_likelihood', 'aic', 'bic',
                            'nrmse_range_train', 'nrmse_mean_train', 'rmse_val', 'nrmse_range_val', 'nrmse_mean_val',
                            'precision', 'recall', 'f1', 'roc_auc', 'KS_score', 'KS_score_normalized', 'KS_score_val',
                            'KS_score_val_normalized', 'cross_entropy_train', 'cross_entropy_val', 'n_train', 'n_val',
                            'c_ff', 'c_fb']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for fold_idx, metrics in enumerate(all_metrics['fold_metrics']):
                    metrics['c_ff'] = ','.join(map(str, metrics['c_ff']))
                    metrics['c_fb'] = ','.join(map(str, metrics['c_fb']))
                    writer.writerow({
                        'input_neuron': input_neuron,
                        'output_neuron': output_neuron,
                        'rank': rank,
                        'bump_score': bump_score if bump_score is not None else '',
                        'fold': fold_idx + 1,
                        'L': L,
                        'alpha_k': alpha_k,
                        'alpha_h': alpha_h,
                        'norm_ce_val': metrics['cross_entropy_val_normalized'],
                        'rmse_train': metrics['rmse_train'],
                        'k0': metrics['k0'],
                        'deviance_reduction': metrics['deviance_reduction'],
                        'log_likelihood': metrics['log_likelihood'],
                        'null_log_likelihood': metrics['null_log_likelihood'],
                        'aic': metrics['aic'],
                        'bic': metrics['bic'],
                        'nrmse_range_train': metrics['nrmse_range_train'],
                        'nrmse_mean_train': metrics['nrmse_mean_train'],
                        'rmse_val': metrics['rmse_val'],
                        'nrmse_range_val': metrics['nrmse_range_val'],
                        'nrmse_mean_val': metrics['nrmse_mean_val'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'roc_auc': metrics['roc_auc'],
                        'KS_score': metrics['KS_score'],
                        'KS_score_normalized': metrics['KS_score_normalized'],
                        'KS_score_val': metrics['KS_score_val'],
                        'KS_score_val_normalized': metrics['KS_score_val_normalized'],
                        'cross_entropy_train': metrics['cross_entropy_train'],
                        'cross_entropy_val': metrics['cross_entropy_val'],
                        'n_train': metrics['n_train'],
                        'n_val': metrics['n_val'],
                        'c_ff': metrics['c_ff'],
                        'c_fb': metrics['c_fb']
                    })
                if f_SISO:
                    for fold_idx, metrics in enumerate(all_metrics_reverse['fold_metrics']):
                        metrics['c_ff'] = ','.join(map(str, metrics['c_ff']))
                        metrics['c_fb'] = ','.join(map(str, metrics['c_fb']))
                        writer.writerow({
                            'input_neuron': input_neuron,
                            'output_neuron': output_neuron,
                            'rank': rank,
                            'bump_score': bump_score if bump_score is not None else '',
                            'fold': fold_idx + 1,
                            'L': L,
                            'alpha_k': alpha_k,
                            'alpha_h': alpha_h,
                            'norm_ce_val': metrics['cross_entropy_val_normalized'],
                            'rmse_train': metrics['rmse_train'],
                            'k0': metrics['k0'],
                            'deviance_reduction': metrics['deviance_reduction'],
                            'log_likelihood': metrics['log_likelihood'],
                            'null_log_likelihood': metrics['null_log_likelihood'],
                            'aic': metrics['aic'],
                            'bic': metrics['bic'],
                            'nrmse_range_train': metrics['nrmse_range_train'],
                            'nrmse_mean_train': metrics['nrmse_mean_train'],
                            'rmse_val': metrics['rmse_val'],
                            'nrmse_range_val': metrics['nrmse_range_val'],
                            'nrmse_mean_val': metrics['nrmse_mean_val'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'roc_auc': metrics['roc_auc'],
                            'KS_score': metrics['KS_score'],
                            'KS_score_normalized': metrics['KS_score_normalized'],
                            'KS_score_val': metrics['KS_score_val'],
                            'KS_score_val_normalized': metrics['KS_score_val_normalized'],
                            'cross_entropy_train': metrics['cross_entropy_train'],
                            'cross_entropy_val': metrics['cross_entropy_val'],
                            'n_train': metrics['n_train'],
                            'n_val': metrics['n_val'],
                            'c_ff': metrics['c_ff'],
                            'c_fb': metrics['c_fb']
                        })
                    
            # Generate plot and save kernels
            siso_data = kernel_data(all_metrics, ff_basis, fb_basis,  avg_ff_coeffs, avg_fb_coeffs, L, alpha_k, alpha_h,k0, input_neuron, output_neuron, rank)
            
            # Compute and print average c_ff and c_fb across folds (like in read_out script)
            c_ff_values = []
            c_fb_values = []
            for fold in all_metrics['fold_metrics']:
                try:
                    c_ff = fold['c_ff']
                    c_fb = fold['c_fb']
                    # Check if c_ff and c_fb are strings (comma-separated) and convert to arrays
                    if isinstance(c_ff, str):
                        c_ff = np.array([float(x) for x in c_ff.split(',')])
                    if isinstance(c_fb, str):
                        c_fb = np.array([float(x) for x in c_fb.split(',')])
                    c_ff_values.append(c_ff)
                    c_fb_values.append(c_fb)
                except (ValueError, KeyError) as e:
                    print(f"Error parsing c_ff or c_fb in fold: {e}. Using zeros")
                    length = len(c_ff_values[0]) if c_ff_values else L
                    c_ff_values.append(np.zeros(length))
                    c_fb_values.append(np.zeros(length))

            if c_ff_values:
                c_ff_array = np.stack(c_ff_values)
                c_fb_array = np.stack(c_fb_values)

                c_ff_avg = np.mean(c_ff_array, axis=0)
                c_fb_avg = np.mean(c_fb_array, axis=0)
                num_folds = len(c_ff_values)
                print(f"Average c_ff across {num_folds} folds: {c_ff_avg}")
                print(f"Average c_fb across {num_folds} folds: {c_fb_avg}")
                
                # Add averages to siso_data for saving
                siso_data['c_ff_avg'] = c_ff_avg
                siso_data['c_fb_avg'] = c_fb_avg
                siso_data['num_folds'] = num_folds
            
            if f_SISO:
                siso_data_reverse = kernel_data(all_metrics_reverse, ff_basis_reverse, fb_basis_reverse,  avg_ff_coeffs_reverse, avg_fb_coeffs_reverse, L, alpha_k, alpha_h,k0_reverse, output_neuron, input_neuron, rank)
                
                # Compute and print average c_ff and c_fb across folds for reverse direction
                c_ff_values_reverse = []
                c_fb_values_reverse = []
                for fold in all_metrics_reverse['fold_metrics']:
                    try:
                        c_ff = fold['c_ff']
                        c_fb = fold['c_fb']
                        # Check if c_ff and c_fb are strings (comma-separated) and convert to arrays
                        if isinstance(c_ff, str):
                            c_ff = np.array([float(x) for x in c_ff.split(',')])
                        if isinstance(c_fb, str):
                            c_fb = np.array([float(x) for x in c_fb.split(',')])
                        c_ff_values_reverse.append(c_ff)
                        c_fb_values_reverse.append(c_fb)
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing c_ff or c_fb in reverse fold: {e}. Using zeros")
                        length = len(c_ff_values_reverse[0]) if c_ff_values_reverse else L
                        c_ff_values_reverse.append(np.zeros(length))
                        c_fb_values_reverse.append(np.zeros(length))

                if c_ff_values_reverse:
                    c_ff_array_reverse = np.stack(c_ff_values_reverse)
                    c_fb_array_reverse = np.stack(c_fb_values_reverse)
                    c_ff_avg_reverse = np.mean(c_ff_array_reverse, axis=0)
                    c_fb_avg_reverse = np.mean(c_fb_array_reverse, axis=0)
                    num_folds_reverse = len(c_ff_values_reverse)
                    print(f"Average c_ff (reverse) across {num_folds_reverse} folds: {c_ff_avg_reverse}")
                    print(f"Average c_fb (reverse) across {num_folds_reverse} folds: {c_fb_avg_reverse}")
                    
                    # Add averages to siso_data_reverse for saving
                    siso_data_reverse['c_ff_avg'] = c_ff_avg_reverse
                    siso_data_reverse['c_fb_avg'] = c_fb_avg_reverse
                    siso_data_reverse['num_folds'] = num_folds_reverse
                
                data = {'SISO':siso_data, 'F_SISO':siso_data_reverse}
                plot_fit(data, model_full, pair_data[2], pair_data[3],model_fsiso=model_full_reverse, save_dir=SAVE_DIR)

            else:
                data = {'SISO':siso_data}
                plot_fit(data, model_full, pair_data[2], pair_data[3], save_dir=SAVE_DIR)

            kernel_file = os.path.join(SAVE_DIR, f'kernels_{input_neuron}_{output_neuron}_rank{rank}.pkl')
            with open(kernel_file, 'wb') as f:
                pickle.dump(data, f)
            print(f'Saved kernels at {kernel_file}')

            
            # plot_fit_and_save_kernels(all_metrics, model_full, ff_basis, fb_basis, avg_ff_coeffs, avg_fb_coeffs, L, alpha_k, alpha_h,k0,
                                    # input_neuron, output_neuron, pair_data[2], pair_data[3], rank)
            avg_norm_ce_val = np.mean([m['cross_entropy_val_normalized'] for m in all_metrics['fold_metrics']])
            se_norm_ce_val = np.std([m['cross_entropy_val_normalized'] for m in all_metrics['fold_metrics']]) / np.sqrt(len(all_metrics['fold_metrics']))
            avg_rmse = np.mean([m['rmse_train'] for m in all_metrics['fold_metrics']])
            logger.info(f"Results for {input_neuron} -> {output_neuron}: Norm CE Val: {avg_norm_ce_val:.4f}, SE: {se_norm_ce_val:.4f}, RMSE: {avg_rmse:.4f}")
        
            if f_SISO:
                # plot_fit_and_save_kernels(all_metrics_reverse, model_full_reverse, ff_basis_reverse, fb_basis_reverse, avg_ff_coeffs_reverse, avg_fb_coeffs_reverse, L, alpha_k, alpha_h,k0_reverse,
                #                     output_neuron, input_neuron, pair_data[3], pair_data[2], rank)
                avg_norm_ce_val = np.mean([m['cross_entropy_val_normalized'] for m in all_metrics_reverse['fold_metrics']])
                se_norm_ce_val = np.std([m['cross_entropy_val_normalized'] for m in all_metrics_reverse['fold_metrics']]) / np.sqrt(len(all_metrics_reverse['fold_metrics']))
                avg_rmse = np.mean([m['rmse_train'] for m in all_metrics_reverse['fold_metrics']])
                logger.info(f"Results for {output_neuron} -> {input_neuron}: Norm CE Val: {avg_norm_ce_val:.4f}, SE: {se_norm_ce_val:.4f}, RMSE: {avg_rmse:.4f}")
        
        except Exception as e:
            logger.error(f"Error processing {input_neuron} -> {output_neuron}: {e}\n{traceback.format_exc()}")
            continue
        
        # Clean up after each pair to reduce memory usage
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_status(f"After pair {pair_idx} ")
    
    print(f"\n{'#'*60}")
    print(f"All pairs processed successfully!")
    print_memory_status("Final ")
    print(f"{'#'*60}\n")
    
    # # Generate overlay plot
    # overlay_kernels(SAVE_DIR)

def plot_glm_fit_results(results, x_full, y_full, input_neuron='input', output_neuron='output', rank=1, save_dir='', f_siso_results=None):
    """
    Wrapper function to easily plot GLM fitting results.
    
    Args:
        results: Tuple from fit_glm_to_stdp containing:
            (model_full, conf_int, all_metrics, avg_ff_coeffs, avg_fb_coeffs, k0, ff_basis, fb_basis)
        x_full: Input spike train data (torch tensor)
        y_full: Output spike train data (torch tensor)
        input_neuron: Name/ID of input neuron (str)
        output_neuron: Name/ID of output neuron (str)
        rank: Rank of the neuron pair (int)
        save_dir: Directory to save the plot (str)
        f_siso_results: Optional results from reverse direction fit (same tuple format as results)
    
    Returns:
        str: Path to the saved plot file
    """
    model_full, conf_int, all_metrics, avg_ff_coeffs, avg_fb_coeffs, k0, ff_basis, fb_basis = results
    
    # Create SISO data dictionary
    siso_data = kernel_data(all_metrics, ff_basis, fb_basis, avg_ff_coeffs, avg_fb_coeffs, 
                           all_metrics['fold_metrics'][0]['L'], 
                           all_metrics['fold_metrics'][0]['alpha_k'],
                           all_metrics['fold_metrics'][0]['alpha_h'],
                           k0, input_neuron, output_neuron, rank)
    
    # If f_siso_results provided, create F_SISO data dictionary
    if f_siso_results is not None:
        model_full_reverse, conf_int_reverse, all_metrics_reverse, avg_ff_coeffs_reverse, avg_fb_coeffs_reverse, k0_reverse, ff_basis_reverse, fb_basis_reverse = f_siso_results
        
        f_siso_data = kernel_data(all_metrics_reverse, ff_basis_reverse, fb_basis_reverse,
                                 avg_ff_coeffs_reverse, avg_fb_coeffs_reverse,
                                 all_metrics_reverse['fold_metrics'][0]['L'],
                                 all_metrics_reverse['fold_metrics'][0]['alpha_k'],
                                 all_metrics_reverse['fold_metrics'][0]['alpha_h'],
                                 k0_reverse, output_neuron, input_neuron, rank)
        
        data = {'SISO': siso_data, 'F_SISO': f_siso_data}
        file_name = plot_fit(data, model_full, x_full, y_full, model_fsiso=model_full_reverse, save_dir=save_dir)
    else:
        data = {'SISO': siso_data}
        file_name = plot_fit(data, model_full, x_full, y_full, save_dir=save_dir)
    
    # Return the expected plot filename
    L = all_metrics['fold_metrics'][0]['L']
    alpha_k = all_metrics['fold_metrics'][0]['alpha_k']
    alpha_h = all_metrics['fold_metrics'][0]['alpha_h']
    return file_name

# Example usage:
"""
# After running fit_glm_to_stdp:
results = fit_glm_to_stdp(data, target, L, alpha_k, alpha_h, input_neuron, output_neuron, rank)
plot_file = plot_glm_fit_results(results, x_full, y_full, input_neuron, output_neuron, rank)
print(f"Plot saved to: {plot_file}")

# If you have both forward and reverse fits:
forward_results = fit_glm_to_stdp(data, target, L, alpha_k, alpha_h, input_neuron, output_neuron, rank)
reverse_results = fit_glm_to_stdp(data_reverse, target_reverse, L, alpha_k, alpha_h, output_neuron, input_neuron, rank)
plot_file = plot_glm_fit_results(forward_results, x_full, y_full, input_neuron, output_neuron, rank, 
                                f_siso_results=reverse_results)
"""

if __name__ == "__main__":
    # Parse command line arguments and set up configuration
    cmd_args = parse_arguments()
    SAVE_DIR = cmd_args.save_dir
    max_tau = cmd_args.max_tau
    sample_rate = cmd_args.sample_rate
    NUMBER_OF_FOLDS = cmd_args.num_folds
    L = cmd_args.L
    alpha_k = cmd_args.alpha_k
    alpha_h = cmd_args.alpha_h
    print(f'L: {L}','max_tau: {max_tau},','alpha_k: {alpha_k},','alpha_h: {alpha_h}')

    RANKING_FILE = cmd_args.ranking_file
    RANK_RANGE = cmd_args.rank_range
    f_SISO = cmd_args.f_SISO
    RANK_RANGE = [int(rank) for rank in RANK_RANGE.split('-')]
    print(f'Analyzing ranks {RANK_RANGE[0]} to {RANK_RANGE[1]}')
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load neurons data
    with open(cmd_args.data_file, 'rb') as f:
        data = pickle.load(f)

        if 'neurons' in data.keys():
            neurons = data['neurons'] # this will be a list of dicts each item is {'name': 'n005_L_CA3_wire_2_cell_1', 'varVersion': 101, 'wireNumber': 0, 'unitNumber': 0, 'xPos': 0.0, 'yPos': 1.0, 'timestamps':
            # we clean it so neurons is a dict with keys like 'n001_X', 'n001_Y', 'n002_X', 'n002_Y', etc.
            neurons = {neuron['name']: neuron['timestamps'] for neuron in neurons}
            if 'tend' in data.keys() and 'events' in data.keys() and cmd_args.only_use_trial_data:
                # Find the first event with name 'TRIAL' and set cutoff_time to its 'tend' field
                cutoff_time = None
                for event in data['events']:
                    if event.get('name', '') == 'TRIAL':
                        cutoff_time = event['timestamps'][-1]
                        break
                print(f'Cutoff time: {cutoff_time}, recording len', data['tend'])
                if cutoff_time is not None:
                    # Trim each neuron's timestamps to <= cutoff_time
                    neurons = {name: np.array([t for t in ts if t <= cutoff_time]) for name, ts in neurons.items()}
                    print(f'Trimmed all neurons to t <= {cutoff_time}')
                else:
                    print('No TRIAL event found; using full recording.')

        else:
            neurons = data
        print(f'Using all data for {len(neurons)} neurons')


    # Read neuron pairs
    neuron_pairs = []
    if cmd_args.neuron_pairs:
        pairs_str = cmd_args.neuron_pairs.split(',')
        for i, pair in enumerate(pairs_str, 1):
            n1, n2 = pair.split(':')
            neuron_pairs.append((n1, n2, None, i))  # bump_score=None, rank=sequential
        print(f"Using provided neuron pairs: {neuron_pairs}")
    else:
        # Read ranking CSV
        with open(RANKING_FILE, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rank = int(row['Rank'])
                if RANK_RANGE[0] <= rank <= RANK_RANGE[1]:
                    neuron_pairs.append((row['Neuron1'], row['Neuron2'], float(row['BumpScore']), rank))

    run_glm_fitting()
    # overlay_kernels(SAVE_DIR)
    #  python glm_fit_cv_one_neuron.py --data_file DNMS_data/april14testsim24000s_with_runsiso_stdp_fffb_sim.pkl --ranking_file DNMS_data/analysis/april14testsim24000s_with_runsiso_stdp_fffb_sim/pair_rankings_ultra-fine.csv  --rank_range 1-2 --f_SISO True


#  python glm_fit_cv_one_neuron.py --data_file DNMS_data/april14testsim21000s08fffiring_freq054000s.pkl --ranking_file DNMS_data/analysis/april14testsim21000s08fffiring_freq054000s/pair_rankings_ultra-fine.csv  --rank_range 1-2 --f_SISO True

# april23testsim2000s1hz

# python glm_fit_cv_one_neuron.py --data_file DNMS_data/april23testsim2000s1hz.pkl --ranking_file DNMS_data/analysis/april23testsim2000s1hz/pair_rankings_ultra-fine.csv  --rank_range 1-2 --f_SISO True --save_dir DNMS_data/single_pairs_analysis_apr23200ssim


# python glm_fit_cv_one_neuron.py --data_file DNMS_data/april24sim_savetmax4000000ff3maxff1_0_maxfb3_1.pkl --ranking_file DNMS_data/analysis/april24sim_savetmax4000000ff3maxff1_0_maxfb3_1/pair_rankings_ultra-fine.csv  --rank_range 1-2 --f_SISO True --save_dir DNMS_data/singlie_pairs_april24sim_savetmax4000000ff3maxff1_0_maxfb3_1

# python glm_fit_cv_one_neuron.py --data_file DNMS_data/sim_jun3_4000s_opengtmax4000000ff3maxff1_0_maxfb2_0.pkl --ranking_file DNMS_data/analysis/sim_jun3_4000s_opengtmax4000000ff3maxff1_0_maxfb2_0/pair_rankings_ultra-fine.csv  --rank_range 1-2 --f_SISO True --save_dir DNMS_data/single_pairs_sim_jun3_4000s_opengtmax4000000ff3maxff1_0_maxfb2_0_L6  --L 6


# python glm_fit_cv_one_neuron.py --data_file DNMS_data/sim_jul30_50s_openg_multi_tmax50000ff3.pkl --neuron_pairs "steep_hebb_X:steep_hebb_Y,steep_anti_hebb_X:steep_anti_hebb_Y" --f_SISO True --save_dir DNMS_data/sim_jul30_50s_openg_multi_tmax50000ff3 --L 5

# python glm_fit_cv_one_neuron.py --data_file DNMS_data/sim_may24_4000s_opengtmax4000000ff3maxff2_0_maxfb3_1.pkl --neuron_pairs --rank_range 1-2 --f_SISO True --save_dir DNMS_data/sim_may24_4000s_opengtmax4000000ff3maxff2_0_maxfb3_1_ver2 --L 5  --alpha_k 0.7 --alpha_h 0.7

# python glm_fit_cv_one_neuron.py \
#     --data_file "DNMS_data/sim_aug18_4000s_k07h07i099_multi_tmax4000000ff3.pkl" \
#     --neuron_pairs "mid_anti_hebb_ver2_X:mid_anti_hebb_ver2_Y" \
#     --f_SISO True \
#     --save_dir "DNMS_data/sim_aug18_4000s_k07h07i099_multi_tmax4000000ff3/testmidantihebb" \
#     --alpha_k 0.7 \
#     --alpha_h 0.7 \
#     --L 5 \
#     --chunk_size 1000



'''
python glm_fit_cv_one_neuron.py \
    --data_file "DNMS_data/selected_neurons1150b032.pkl" \
    --ranking_file DNMS_data/analysis/april24sim_save2bufferstationsrynofftmax1000000ff0_3maxff1_0_maxfb3_1/pair_rankings_fine.csv  \
    --rank_range 1-2  \
    --f_SISO True \
    --save_dir "DNMS_data/april24sim_save2bufferstationsrynofftmax1000000ff0_3maxff1_0_maxfb3_1" \
    --alpha_k 0.7 \
    --alpha_h 0.7 \
    --L 5 \
    --chunk_size 1000



python glm_fit_cv_one_neuron.py \
    --data_file "DNMS_data/selected_neurons1150b032.pkl" \
    --neuron_pairs "n013_L_CA3_wire_4_cell_1:n017_L_CA3_wire_5_cell_1" \
    --f_SISO True \
    --save_dir "DNMS_data/selected_neurons1150b032_tau200" \
    --alpha_k 0.81 \
    --alpha_h 0.88 \
    --L 4 \
    --chunk_size 1000 \
    --max_tau 200

python glm_fit_cv_one_neuron.py \
    --data_file "DNMS_data/selected_neurons1150b032.pkl" \
    --neuron_pairs "n013_L_CA3_wire_4_cell_1:n069_R_CA1_wire_2_cell_1" \
    --f_SISO True \
    --save_dir "DNMS_data/selected_neurons1150b032_tau200" \
    --alpha_k 0.81 \
    --alpha_h 0.88 \
    --L 4 \
    --chunk_size 1000 \
    --max_tau 200 

    --num_folds


python utils/glm_fit_cv_one_neuron.py \
    --data_file "DNMS_data/selected_neurons1150b032.pkl" \
    --neuron_pairs "n017_L_CA3_wire_5_cell_1:n117_R_CA3_wire_6_cell_1" \
    --f_SISO True \
    --save_dir "DNMS_data/selected_neurons1150b032_tau200_L4" \
    --alpha_k 0.81 \
    --alpha_h 0.88 \
    --L 4 \
    --chunk_size 1000 \
    --max_tau 200 \
    --num_folds 1


python utils/glm_fit_cv_one_neuron.py \
    --data_file "data/Jan2010-Nonstationarity_Learning/1150_10_sec/1150b032merge-clean_cutoff_5.pkl" \
    --ranking_file "data/Jan2010-Nonstationarity_Learning/analysis/1150_10_sec/1150b032merge-clean_cutoff_5/pair_rankings_semifine.csv" \
    --save_dir "data/Jan2010-Nonstationarity_Learning/single_pair_analysis/1150_10_sec/1150b032merge-clean_cutoff_5_100_L5_k0_7_h0_7" \
    --rank_range "1-2" \
    --alpha_k 0.7 \
    --alpha_h 0.7 \
    --num_folds 5 \
    --L 5 \
    --max_tau 100 --only_use_trial_data
'''