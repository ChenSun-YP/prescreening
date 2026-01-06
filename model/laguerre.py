from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from scipy.special import comb
import numpy as np
import math
import os
from control import tf2ss, series, impulse_response, StateSpace


# def comb(n, k):
#     if n < 0 or k < 0 or n < k:
#         return 0  # Return 0 for invalid values
#     return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def laguerre_basis(tau, j, a):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    sums = 0
    for k in range(j + 1):
        sums += (-1)**k * comb(tau, k) * comb(j, k) * a**(j - k) * (1 - a)**k
    lag = (-1)**j * a**((tau - j) / 2) * torch.sqrt(1 - a) * sums
    return lag

def precompute_laguerre_basis(num_basis, max_tau, alpha):
    basis_matrix = torch.zeros(num_basis, max_tau)
    for j in range(num_basis):
        for tau in range(max_tau):
            basis_matrix[j, tau] = laguerre_basis(tau, j, alpha)
    return basis_matrix




class ParameterizedLaguerreBasis(nn.Module):
    def __init__(self, num_basis, max_tau, initial_alpha=0.5, name='',device='cpu'):
        super().__init__()
        assert num_basis > 0, "num_basis must be greater than 0."
        assert max_tau > 0, "max_tau must be greater than 0."
        self.num_basis = num_basis
        self.max_tau = max_tau
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32,device=device),requires_grad=False)
        self.name = name
        self.device = device
        # Add alpha to the basis file name
        self.basis_file = f'laguerre_basis_nb{num_basis}_mt{max_tau}_a{initial_alpha:.4f}_{name}_{device}.pt'
        self.cached_basis = None
        self.cached_alpha = None
        self.use_matlab_ver = False
        
        if os.path.exists(self.basis_file):
            self.load_precomputed_basis()
        else:
            self.compute_and_save_basis()

    @staticmethod
    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def compute_and_save_basis(self):
        if self.use_matlab_ver:

            def lagu_re(k, n, p, device='cpu'):

                # Convert p to a scalar if it's a tensor
                if isinstance(p, torch.Tensor):
                    if p.numel() != 1:
                        raise ValueError(f"p must be a scalar tensor, got shape {p.shape}")
                    p = p.item()  # Convert to Python scalar
                elif not isinstance(p, (int, float)):
                    raise ValueError(f"p must be a scalar, got type {type(p)}")
                
                if abs(p) >= 1:
                    raise ValueError(f"|p| must be less than 1, got {p}")

                # Compute nn
                nn = np.sqrt(1 - p**2)
                
                # Define transfer function coefficients as NumPy arrays
                n1 = np.array([0, nn], dtype=np.float64)  # Numerator: 0*z + nn
                p1 = np.array([1, -p], dtype=np.float64)  # Denominator: z - p
                
                # Convert first transfer function to state-space
                sys1 = tf2ss(n1, p1)
                a, b, c, d = sys1.A, sys1.B, sys1.C, sys1.D
                
                # Second transfer function
                n0 = np.array([-p, 1], dtype=np.float64)  # Numerator: -p*z + 1
                p0 = np.array([1, -p], dtype=np.float64)  # Denominator: z - p
                
                sys0 = tf2ss(n0, p0)
                a0, b0, c0, d0 = sys0.A, sys0.B, sys0.C, sys0.D
                
                # Series connection for k-1 iterations
                for i in range(2, k + 1):
                    sys = series(sys0, StateSpace(a, b, c, d))
                    a, b, c, d = sys.A, sys.B, sys.C, sys.D
                
                # Compute impulse response
                sys_final = StateSpace(a, b, c, d)
                T = np.arange(0, n + 1)  # Time steps 0 to n
                _, y1 = impulse_response(sys_final, T=T)
                
                # Extract y from y1, skipping first element
                y = y1[1:n + 1]
                
                # Convert to PyTorch tensor
                y_torch = torch.tensor(y.T, dtype=torch.float32, device=device)
    
                
                return y_torch
            B = torch.zeros(self.num_basis,self.max_tau) # L,M
            for i in range(self.num_basis): # for each L
                B[ i,:] = lagu_re(i, self.max_tau, self.alpha)
            self.register_buffer('basis_matrix', B)
        else:

            print(f"Computing and saving Laguerre basis for num_basis={self.num_basis}, max_tau={self.max_tau}, alpha={self.alpha.item()}, name={self.name}")
            comb_terms = torch.zeros(self.num_basis, self.max_tau, self.num_basis, dtype=torch.float32)
            for j in range(self.num_basis):
                for tau in range(self.max_tau):
                    for k in range(j + 1):
                        comb_terms[j, tau, k] = (-1)**k * self.comb(tau, k) * self.comb(j, k)
            self.register_buffer('comb_terms', comb_terms)
            
            torch.save({
                'comb_terms': comb_terms,
                'num_basis': self.num_basis,
                'max_tau': self.max_tau,
                'name': self.name,
                'alpha': self.alpha.item()  # Save alpha value
            }, self.basis_file)
            print(f"Laguerre basis saved to {self.basis_file}")



    def load_precomputed_basis(self):
        print(f"Loading pre-computed Laguerre basis from {self.basis_file}")
        saved_data = torch.load(self.basis_file)
        comb_terms = saved_data['comb_terms']
        if not isinstance(comb_terms, torch.Tensor):
            comb_terms = torch.tensor(comb_terms, dtype=torch.float32)
        comb_terms.requires_grad_(False)

        self.register_buffer('comb_terms', comb_terms)
        
        assert saved_data['num_basis'] == self.num_basis, "Loaded num_basis does not match"
        assert saved_data['max_tau'] == self.max_tau, "Loaded max_tau does not match"
        assert saved_data['name'] == self.name, "Loaded name does not match"
        assert math.isclose(saved_data['alpha'], self.alpha.item(), rel_tol=1e-5), "Loaded alpha does not match"

    def forward(self):
        a = self.alpha.clamp(0, 1)
        if self.cached_basis is not None and torch.isclose(a, self.cached_alpha):
            return self.cached_basis

        # Precompute common terms
        sqrt_1_minus_a = torch.sqrt(1 - a)
        # a_pow_j = a ** torch.arange(self.num_basis, device=self.alpha.device)
        a_pow_tau_minus_j_half = a ** ((torch.arange(self.max_tau, device=self.alpha.device)[None, :] - torch.arange(self.num_basis, device=self.alpha.device)[:, None]) / 2)

        # Compute sums in a partially vectorized way
        if self.use_matlab_ver:
            basis_matrix = self.basis_matrix
        else:
            basis_matrix = torch.zeros(self.num_basis, self.max_tau, device=self.alpha.device)
            for j in range(self.num_basis):
                k_indices = torch.arange(j + 1, device=self.alpha.device)
                powers = a ** (j - k_indices) * (1 - a) ** k_indices
                sums = torch.sum(self.comb_terms[j, :, :j+1] * powers, dim=-1)
                basis_matrix[j, :] = (-1) ** j * a_pow_tau_minus_j_half[j, :] * sqrt_1_minus_a * sums

        self.cached_basis = basis_matrix
        self.cached_alpha = a
        return basis_matrix


class ParameterizedLaguerreBasis_multineuron(torch.nn.Module):
    def __init__(self, num_basis_list, max_tau_list, initial_alpha=0.5, name='',device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        assert len(num_basis_list) == len(max_tau_list), "Lists must have same length."
        self.N = len(num_basis_list)  # Number of pairs
        self.num_basis_all = torch.tensor(num_basis_list, dtype=torch.int64, device=self.device)
        self.max_tau = torch.tensor(max_tau_list, dtype=torch.int64, device=self.device)
        self.num_basis = torch.max(self.num_basis_all).item()  # For output shape
        self.max_max_tau = torch.max(self.max_tau).item()      # For output shape
        self.alpha = torch.nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, device=self.device), requires_grad=True)
        self.name = name
        
        # Basis files and caches for each pair
        self.basis_files = [f'laguerre_basis_nb{nb}_mt{mt}_a{initial_alpha:.4f}_{name}_pair{i}.pt'
                            for i, (nb, mt) in enumerate(zip(num_basis_list, max_tau_list))]
        self.cached_bases = [None] * self.N
        self.cached_alpha = None
        self.use_matlab_ver = False
        
        # Load or compute bases
        self.basis_matrices = []
        for i in range(self.N):
            if os.path.exists(self.basis_files[i]):
                self.load_precomputed_basis(i)
            else:
                self.compute_and_save_basis(i)
            # Ensure basis is on device
            self.basis_matrices[i] = self.basis_matrices[i].to(self.device)

    # @staticmethod
    # def comb(n, k):
    #     if n < 0 or k < 0 or n < k:
    #         return 0
    #     return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0.0
        k = min(k, n - k)  # Use symmetry for efficiency
        res = 1.0
        for i in range(k):
            res *= (n - i) / (i + 1)
        return res

    def compute_and_save_basis(self, pair_idx):
        nb = self.num_basis_all[pair_idx].item()
        mt = self.max_tau[pair_idx].item()
        print(f"Computing Laguerre basis for pair {pair_idx}: num_basis={nb}, max_tau={mt}, alpha={self.alpha.item()}")
        
        if self.use_matlab_ver:
            # Placeholder for MATLAB-style computation
            B = torch.zeros(nb, mt, device=self.device)
            for i in range(nb):
                B[i, :] = torch.zeros(mt, device=self.device)  # Replace with lagu_re(i, mt, self.alpha)
            self.basis_matrices.append(B)
        else:
            comb_terms = torch.zeros(nb, mt, nb, dtype=torch.float32, device=self.device)
            for j in range(nb):
                for tau in range(mt):
                    for k in range(j + 1):
                        comb_terms[j, tau, k] = (-1)**k * self.comb(tau, k) * self.comb(j, k)
            
            torch.save({
                'comb_terms': comb_terms.cpu(),  # Save to CPU to reduce GPU memory usage
                'num_basis': nb,
                'max_tau': mt,
                'name': self.name,
                'alpha': self.alpha.item(),
                'pair_idx': pair_idx
            }, self.basis_files[pair_idx])
            print(f"Saved basis to {self.basis_files[pair_idx]}")
            self.basis_matrices.append(comb_terms)

    def load_precomputed_basis(self, pair_idx):
        print(f"Loading basis for pair {pair_idx} from {self.basis_files[pair_idx]}")
        saved_data = torch.load(self.basis_files[pair_idx], map_location=self.device)
        comb_terms = saved_data['comb_terms']
        if not isinstance(comb_terms, torch.Tensor):
            comb_terms = torch.tensor(comb_terms, dtype=torch.float32, device=self.device)
        comb_terms.requires_grad_(False)
        
        self.basis_matrices.append(comb_terms)
        assert saved_data['num_basis'] == self.num_basis_all[pair_idx].item()
        assert saved_data['max_tau'] == self.max_tau[pair_idx].item()
        assert saved_data['name'] == self.name
        assert math.isclose(saved_data['alpha'], self.alpha.item(), rel_tol=1e-5)

    def forward(self):
        a = self.alpha.clamp(0, 1)
        if all(b is not None for b in self.cached_bases) and torch.isclose(a, self.cached_alpha):
            # Stack cached bases with padding
            output = torch.zeros(self.N, self.num_basis, self.max_max_tau, device=self.device)
            for i, basis in enumerate(self.cached_bases):
                nb = self.num_basis_all[i].item()
                mt = self.max_tau[i].item()
                output[i, :nb, :mt] = basis
            return output

        # Precompute common terms
        sqrt_1_minus_a = torch.sqrt(1 - a)
        output = torch.zeros(self.N, self.num_basis, self.max_max_tau, device=self.device)
        
        for i in range(self.N):
            nb = self.num_basis_all[i].item()
            mt = self.max_tau[i].item()
            a_pow_tau_minus_j_half = a ** ((torch.arange(mt, device=self.device)[None, :] - 
                                           torch.arange(nb, device=self.device)[:, None]) / 2)
            
            if self.use_matlab_ver:
                basis_matrix = self.basis_matrices[i]
            else:
                basis_matrix = torch.zeros(nb, mt, device=self.device)
                for j in range(nb):
                    k_indices = torch.arange(j + 1, device=self.device)
                    powers = a ** (j - k_indices) * (1 - a) ** k_indices
                    sums = torch.sum(self.basis_matrices[i][j, :, :j+1] * powers, dim=-1)
                    basis_matrix[j, :] = (-1) ** j * a_pow_tau_minus_j_half[j, :] * sqrt_1_minus_a * sums
            
            output[i, :nb, :mt] = basis_matrix
            self.cached_bases[i] = basis_matrix
        
        self.cached_alpha = a
        return output

# class ParameterizedLaguerreBasis_multineuron(nn.Module):
#     def __init__(self, num_basis_list, max_tau_list, initial_alpha=0.5, name='', device='cpu'):
#         super().__init__()
#         self.device = torch.device(device)
#         assert len(num_basis_list) == len(max_tau_list), "Lists must have same length."
#         self.N = len(num_basis_list)  # Number of pairs
#         self.num_basis_all = torch.tensor(num_basis_list, dtype=torch.int64, device=self.device)
#         self.max_tau = torch.tensor(max_tau_list, dtype=torch.int64, device=self.device)
#         self.num_basis = torch.max(self.num_basis_all).item()  # For output shape
#         self.max_max_tau = torch.max(self.max_tau).item()      # For output shape
#         self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, device=self.device), requires_grad=True)
#         self.name = name
#         self.use_matlab_ver = False

#         # Basis files for saving/loading precomputed bases
#         self.basis_files = [f'laguerre_basis_nb{nb}_mt{mt}_a{initial_alpha:.4f}_{name}_pair{i}.pt'
#                             for i, (nb, mt) in enumerate(zip(num_basis_list, max_tau_list))]
        
#         # Initialize basis matrices as buffers
#         self.basis_matrices = []
#         for i in range(self.N):
#             if os.path.exists(self.basis_files[i]):
#                 self.load_precomputed_basis(i)
#             else:
#                 self.compute_and_save_basis(i)
#             # Register as buffer
#             self.register_buffer(f'basis_matrix_{i}', self.basis_matrices[i])
        
#         # Track alpha to detect changes
#         self.register_buffer('cached_alpha', torch.tensor(initial_alpha, dtype=torch.float32, device=self.device))

#     @staticmethod
#     def comb(n, k):
#         if n < 0 or k < 0 or n < k:
#             return 0.0
#         k = min(k, n - k)  # Use symmetry for efficiency
#         res = 1.0
#         for i in range(k):
#             res *= (n - i) / (i + 1)
#         return res

#     def compute_and_save_basis(self, pair_idx):
#         nb = self.num_basis_all[pair_idx].item()
#         mt = self.max_tau[pair_idx].item()
#         a = self.alpha.clamp(0, 1).detach()  # Detach to avoid graph
#         print(f"Computing Laguerre basis for pair {pair_idx}: num_basis={nb}, max_tau={mt}, alpha={a.item()}")

#         if self.use_matlab_ver:
#             # Placeholder for MATLAB-style computation
#             B = torch.zeros(nb, mt, device=self.device)
#             for i in range(nb):
#                 B[i, :] = torch.zeros(mt, device=self.device)  # Replace with lagu_re(i, mt, a)
#             self.basis_matrices.append(B)
#         else:
#             comb_terms = torch.zeros(nb, mt, nb, dtype=torch.float32, device=self.device)
#             for j in range(nb):
#                 for tau in range(mt):
#                     for k in range(j + 1):
#                         comb_terms[j, tau, k] = (-1)**k * self.comb(tau, k) * self.comb(j, k)
            
#             # Compute basis matrix
#             basis_matrix = torch.zeros(nb, mt, device=self.device)
#             sqrt_1_minus_a = torch.sqrt(1 - a)
#             for j in range(nb):
#                 k_indices = torch.arange(j + 1, device=self.device)
#                 powers = a ** (j - k_indices) * (1 - a) ** k_indices
#                 sums = torch.sum(comb_terms[j, :, :j+1] * powers, dim=-1)
#                 a_pow_tau_minus_j_half = a ** ((torch.arange(mt, device=self.device) - j) / 2)
#                 basis_matrix[j, :] = (-1) ** j * a_pow_tau_minus_j_half * sqrt_1_minus_a * sums
            
#             self.basis_matrices.append(basis_matrix)

#             # Save to file
#             torch.save({
#                 'basis_matrix': basis_matrix.cpu(),  # Save to CPU to reduce GPU memory
#                 'num_basis': nb,
#                 'max_tau': mt,
#                 'name': self.name,
#                 'alpha': a.item(),
#                 'pair_idx': pair_idx
#             }, self.basis_files[pair_idx])
#             print(f"Saved basis to {self.basis_files[pair_idx]}")

#     def load_precomputed_basis(self, pair_idx):
#         print(f"Loading basis for pair {pair_idx} from {self.basis_files[pair_idx]}")
#         saved_data = torch.load(self.basis_files[pair_idx], map_location=self.device)
#         basis_matrix = saved_data['basis_matrix']
#         if not isinstance(basis_matrix, torch.Tensor):
#             basis_matrix = torch.tensor(basis_matrix, dtype=torch.float32, device=self.device)
#         basis_matrix.requires_grad_(False)
        
#         self.basis_matrices.append(basis_matrix)
#         assert saved_data['num_basis'] == self.num_basis_all[pair_idx].item()
#         assert saved_data['max_tau'] == self.max_tau[pair_idx].item()
#         assert saved_data['name'] == self.name
#         assert math.isclose(saved_data['alpha'], self.alpha.item(), rel_tol=1e-5)

#     def update_basis_matrices(self):
#         """Update basis matrices if alpha has changed significantly."""
#         a = self.alpha.clamp(0, 1).detach()
#         if not torch.isclose(a, self.cached_alpha, rtol=1e-5):
#             print(f"Alpha changed from {self.cached_alpha.item()} to {a.item()}. Recomputing basis matrices.")
#             self.basis_matrices = []
#             for i in range(self.N):
#                 self.compute_and_save_basis(i)
#                 self.register_buffer(f'basis_matrix_{i}', self.basis_matrices[i])
#             self.cached_alpha = a

#     def forward(self):
#         # Check if alpha has changed and update basis matrices if needed
#         self.update_basis_matrices()
        
#         # Return padded basis matrices
#         output = torch.zeros(self.N, self.num_basis, self.max_max_tau, device=self.device)
#         for i in range(self.N):
#             nb = self.num_basis_all[i].item()
#             mt = self.max_tau[i].item()
#             output[i, :nb, :mt] = getattr(self, f'basis_matrix_{i}')
#         return output

if __name__ == "__main__":
    PLB = ParameterizedLaguerreBasis_multineuron([14], [200], 0.7, name='ff')
    basis = PLB()
    print(basis.shape)
    plt.plot(basis[0, :, :].T.detach().numpy())
    plt.show()