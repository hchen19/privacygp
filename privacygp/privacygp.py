import torch
import gpytorch
from gpytorch.utils.lanczos import lanczos_tridiag
# from gpytorch.lazy import DiagLazyTensor, SumLazyTensor, AddedDiagLazyTensor, LowRankRootLazyTensor
try:
    from gpytorch.lazy import DiagLazyTensor
except ImportError:
    from gpytorch.lazy.diag_lazy_tensor import DiagLazyTensor

from gpytorch.lazy import SumLazyTensor, AddedDiagLazyTensor, LowRankRootLazyTensor


from .train import GPModelTrainer

class PrivacyGP(torch.nn.Module):
    def __init__(
            self, 
            train_x,
            train_y,
            **kwargs,
        ):
        """
        Initialize the PrivacyGP model with all necessary parameters.

        :param train_x: Training input data
        :param train_y: Training target data
        :param alpha: Privacy tolerance parameter (default: 0.1)
        :param **kwargs: All parameters for training, obfuscation, and reconstruction
        """
        super(PrivacyGP, self).__init__()

        # Store all kwargs for later use in train, obfuscate, and reconstruct
        self.params = {
            # Default training parameters
            'train': True,
            'noiseless': False, 
            'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(train_x.device),
            'model_name': 'ExactGP',
            'noise': 0.0025,
            'lengthscale': 0.7,
            'outputscale': 1,
            'mean_constant': 0.0,
            'num_epochs': 50,
            'fix_lengthscale': False,
            
            # Default obfuscation parameters
            'privacy_idx': torch.arange(20, 40, device=train_x.device),
            'num_eigen_tol': 100,
            'seed': 0,
            'alpha': 0.1,
            'normalize': False, # Normalization flag for both inputs and targets

            # Analytical MLE parameter
            'use_analytical_mle': False,
            'zero_mean': False,
        }
        
        # Update with user-provided parameters
        self.params.update(kwargs)

        # Store original train_x and train_y
        self.origin_train_x = train_x
        self.origin_train_y = train_y
        
        if self.params['normalize']:
            # Compute and store normalization parameters for inputs
            self.params['train_x_mean'] = train_x.mean(dim=0, keepdim=True)
            self.params['train_x_std'] = train_x.std(dim=0, keepdim=True) + 1e-6
            
            # Compute and store normalization parameters for targets
            self.params['train_y_mean'] = train_y.mean()
            self.params['train_y_std'] = train_y.std() + 1e-6

            # Normalize inputs and targets
            self.train_x = (train_x - self.params['train_x_mean']) / self.params['train_x_std']
            self.train_y = (train_y - self.params['train_y_mean']) / self.params['train_y_std']
            
            
            print(f"Normalized train_x with mean {self.params['train_x_mean'].tolist()[0]} and std {self.params['train_x_std'].tolist()[0]}")
            print(f"Normalized train_y with mean {self.params['train_y_mean'].item():.4f} and std {self.params['train_y_std'].item():.4f}")
            
        else:
            # Use original data without normalization
            self.train_x = train_x
            self.train_y = train_y
    
    def train(self):

        """Train the GP model with parameters from initialization"""
        
        model = GPModelTrainer(
            train_x = self.train_x, 
            train_y = self.train_y,
            likelihood = self.params['likelihood'],
            noise = self.params['noise'],
            lengthscale = self.params['lengthscale'],
            outputscale = self.params['outputscale'],
            mean_constant = self.params['mean_constant'],
            model_name = self.params['model_name'],
            noiseless = self.params['noiseless'],  # Pass noiseless parameter
            fix_lengthscale = self.params['fix_lengthscale'],
            use_analytical_mle = self.params['use_analytical_mle'],
            zero_mean = self.params['zero_mean'],
        )
    
        if self.params['train']:
            model.train(
                num_epochs = self.params['num_epochs'],
            )
        self.gpmodel = model.model


    def privacy_tol(self):
        """
        privacy tolerance H is a PSD matrix
        In this work, we consider H(x_privacy, x_privacy) = alpha * K(x_privacy, x_privacy)
        """
        self.tol_covar_module = self.params['alpha'] * self.gpmodel.covar_module

    def obfuscate(self):

        """Step 2: Obfuscation"""

        if not hasattr(self, 'model'):
            self.train()

        ##########################################################
        # Compute cov_xx_privacy = G(S) = K_XS H_SS^{-1} K_SX in the paper
        ##########################################################
        # H_SS = alpha * K_SS
        # G(S) =  K_XS [(alpha) *K_SS ]^{-1} K_SX
        self.train_x_privacy = self.train_x[self.params['privacy_idx']]
        self.train_y_privacy = self.train_y[self.params['privacy_idx']]
        cov_xprivacy_xprivacy = self.params['alpha'] * self.gpmodel.covar_module(self.train_x_privacy, self.train_x_privacy)
        cov_x_xprivacy = self.gpmodel.covar_module(self.train_x, self.train_x_privacy)
        if self.train_x_privacy.size(-2) < 1e3:
            cov_xx_privacy = cov_x_xprivacy @ cov_xprivacy_xprivacy.solve(
                cov_x_xprivacy.transpose(-2, -1).to_dense()
                )
        else:
            root_decomp = cov_xprivacy_xprivacy.root_decomposition()# This returns R where R @ R.T ≈ cov_xprivacy_xprivacy
            inv_root = root_decomp.root_inv_decomposition() # This returns R^{-1} such that R^{-1}.T @ R^{-1} ≈ cov_xprivacy_xprivacy^{-1}
            temp = inv_root @ cov_x_xprivacy.transpose(-1, -2) # Apply the inverse root to cov_xs_x.T: computes R^{-1}.T @ R^{-1} @ cov_xs_x.T
            cov_xx_privacy = cov_x_xprivacy @ temp # Compute cov_xs_x @ temp = cov_xs_x @ R^{-1}.T @ R^{-1} @ cov_xs_x.T

        ##########################################################
        # Compute privacy_cov = G(S) - K_XX - V (V is noise matrix)
        ##########################################################
        cov_xx = self.gpmodel.covar_module(self.train_x, self.train_x)
        # Handle noise matrix differently based on noiseless parameter
        if self.params['noiseless']:
            # For noiseless models, no noise matrix needed
            self.noise_matrix = DiagLazyTensor(torch.zeros(self.train_x.size(-2), device=self.train_x.device, dtype=self.train_x.dtype))
        else:
            noise = self.gpmodel.likelihood.noise_covar.noise
            self.noise_matrix = DiagLazyTensor(noise.expand(self.train_x.size(-2)))

        privacy_cov = cov_xx_privacy - cov_xx - self.noise_matrix

        ##########################################################
        # Compute PSD part of matrix privacy_cov
        ##########################################################
        # opt_cov = privacy_cov^{+}
        # using square root of PSD A^+ = ( A + sqrt(A^2) ) / 2
        # Jacobi eigenvalue algorithm: A = U^T diag(lambda_1, lambda_n) U
       
        # Get the largest k eigenvalues and eigenvectors
        # eigenvectors^T @ eigenvalues @ eigenvectors = privacy_cov
        
        k = min(self.params['num_eigen_tol'], privacy_cov.size(-1))  # Number of eigenvalues to compute
        q_mat, t_mat = lanczos_tridiag(
            privacy_cov.matmul,
            k,
            device=privacy_cov.device,
            dtype=privacy_cov.dtype,
            matrix_shape=privacy_cov.shape,
        )
        # Convert the tridiagonal matrix to eigenvalues and eigenvectors
        eigenvalues_lanczos, eigenvectors_lanczos = torch.linalg.eigh(t_mat)
        # Map back to the original space
        eigenvectors = q_mat @ eigenvectors_lanczos # (k,k) size tensor

        # Keep only non-negative eigenvalues for PSD part
        pos_mask = eigenvalues_lanczos > 0
        pos_eigenvalues = eigenvalues_lanczos[pos_mask] # (k_pos, ) size tensor
        pos_eigenvectors = eigenvectors[:, pos_mask] # (k, k_pos) size tensor

        torch.manual_seed(self.params['seed'])
        std_samples = torch.randn(pos_eigenvalues.size(0), device=pos_eigenvalues.device, dtype=pos_eigenvalues.dtype)

        # Calculate weighted samples (Λ^(1/2) * z)
        eigenvalues_root = pos_eigenvalues.sqrt() # (k_pos,) size tensor
        weighted_sample = eigenvalues_root * std_samples # (k_pos,) size tensor, element-wise dot product
        
        # Save the optimal noise sample
        self.opt_sample = pos_eigenvectors @ weighted_sample # (k,) size tensor
        # Compute obfuscated data obfuscated_y = trian_y + optimized noise sample ~ N(0, opt_cov)
        self.obfuscated_y = self.train_y + self.opt_sample
        self.obfuscated_x = self.train_x

        # Save the optimal noise covariance (PSD part)
        self.opt_cov = LowRankRootLazyTensor(
            pos_eigenvectors * eigenvalues_root.unsqueeze(0)
        ) # (k,k) size tensor, self.opt_cov =  pos_eigenvectors @ pos_eigenvalues.diag() @ pos_eigenvectors.T # (k,k) size tensor

        # Denornalize
        if self.params['normalize']:
            self.origin_opt_sample = self.opt_sample * self.params['train_y_std'] + self.params['train_y_mean']
            self.origin_obfuscated_x = self.obfuscated_x * self.params['train_x_std'] + self.params['train_x_mean']
            self.origin_obfuscated_y = self.obfuscated_y * self.params['train_y_std'] + self.params['train_y_mean']
            self.origin_opt_cov = self.opt_cov * (self.params['train_y_std'] ** 2)
        else:
            self.origin_opt_sample = self.opt_sample
            self.origin_obfuscated_x = self.obfuscated_x
            self.origin_obfuscated_y = self.obfuscated_y
            self.origin_opt_cov = self.opt_cov




    def reconstruct(self, test_x):
        """
        Step 3: Reconstruction - Compute posterior distribution at test points using LazyTensors
        for efficient computation.
        
        :param test_x: Test points to make predictions at
        :return: tuple of (mean, covariance) for the posterior
        """
        test_x = self.normalize_x(test_x)

        if not hasattr(self, 'obfuscated_y') or not hasattr(self, 'opt_cov'):
            self.obfuscate()

        # Set model to evaluation mode
        self.gpmodel.eval()
        self.gpmodel.likelihood.eval()

        print("Reconstructing...")
        print(f" mean_constant: {self.gpmodel.mean_module.constant.data}")
        print(f" outputscale: {self.gpmodel.covar_module.outputscale.data}")
        print(f" lengthscale: {self.gpmodel.covar_module.base_kernel.lengthscale.data.tolist()[0]}")
        print(f" noise: {self.gpmodel.likelihood.noise_covar.noise.data.item()}")

        with torch.no_grad():
            # Get kernel matrices using GPyTorch's lazy tensors for efficient computation
            cov_xstar_x = self.gpmodel.covar_module(test_x, self.train_x)
            cov_x_x = self.gpmodel.covar_module(self.train_x, self.train_x)
            cov_xstar_xstar = self.gpmodel.covar_module(test_x, test_x)
            
            # Compute full covariance with privacy: K_X,X + V + Σ
            # Using LazyTensor operations to avoid materializing large matrices
            cov_x_x_noisy = SumLazyTensor(
                AddedDiagLazyTensor(cov_x_x, self.noise_matrix),  # K_X,X + V
                self.opt_cov  # Σ
            )            
            
            if test_x.size(-2) < 1e3:
                # For smaller test matrices, use direct solve
                # Solve for multiple right-hand sides (each column of cov_xstar_x.T)
                weight_matrix = cov_x_x_noisy.solve(
                    cov_xstar_x.transpose(-2, -1).to_dense()
                )
                # Now transpose to get the desired (cov_xstar_x @ cov_x_x_noisy^{-1})
                weight_matrix = weight_matrix.transpose(-2, -1)
            else:
                # For larger matrices, use root decomposition for efficiency
                root_decomp = cov_x_x_noisy.root_decomposition()
                inv_root = root_decomp.root_inv_decomposition()
                # compute inv_root @ cov_xstar_x.T
                scaled_cov  = inv_root @ cov_xstar_x.transpose(-1, -2)
                # compute (inv_root @ cov_xstar_x.T).T = cov_xstar_x @ inv_root.T
                weight_matrix = scaled_cov.transpose(-2, -1)
            
            ##############################################
            # Posterior mean
            ##############################################
            # Get mean values at training and test points
            mean_test = self.gpmodel.mean_module(test_x)
            mean_train = self.gpmodel.mean_module(self.train_x)
            
            # Difference between obfuscated observations and mean
            centered_obs = self.obfuscated_y - mean_train

            # Posterior mean: f(X*) = m(X*) + (cov_xstar_x @ cov_x_x_noisy^{-1}) (W - M)
            self.privacy_mean = mean_test + weight_matrix @ centered_obs

            ##############################################
            # Posterior covariance
            ##############################################
            # Posterior covariance: Cov(f*) = K(X*,X*) - (cov_xstar_x @ cov_x_x_noisy^{-1}) K(X,X*)
            # We already have (cov_xstar_x @ cov_x_x_noisy^{-1}) as weight_matrix
            # And K(X,X*) is just cov_xstar_x.T
            self.privacy_cov = cov_xstar_xstar - weight_matrix @ cov_xstar_x.transpose(-2, -1)

            # Denormalize the mean and variance if normalization was applied
            if self.params['normalize']:
                self.privacy_mean = self.privacy_mean * self.params['train_y_std'] + self.params['train_y_mean']
                self.privacy_cov = self.privacy_cov* (self.params['train_y_std'] ** 2)

            return self.privacy_mean, self.privacy_cov

    def forward(self, x):
        mean_x, covar_x = self.reconstruct(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    



    # -------------------------------
    # Helper Functions for Normalization
    # -------------------------------
    def normalize_x(self, x):
        """Normalize new inputs x using training input statistics stored in self.params."""
        if self.params['normalize']:
            return (x - self.params['train_x_mean']) / self.params['train_x_std']
        return x