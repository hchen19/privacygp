"""
reference to https://github.com/lionfish0/dp4gp/blob/master/dp4gp/dp4gp.py
"""

import math
import torch
import gpytorch

from .train import GPModelTrainer


class DPGP(torch.nn.Module):
    """(epsilon, delta)-Differentially Private Gaussian Process predictions.
    
    This class implements differential privacy methods for Gaussian Process regression,
    including the cloaking method described in Hall et al. 2013.
    """
    
    def __init__(
            self, 
            train_x,
            train_y,
            **kwargs
        ):
        """Initialize the DPGP model with all necessary parameters.
        
        :param train_x: Training input data
        :param train_y: Training target data
        :param **kwargs: All parameters for training, privacy, and reconstruction
        """
        super(DPGP, self).__init__()
        
        # Default parameters
        self.params = {
            # Training parameters
            'train': True,
            'noiseless': False, 
            'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(train_x.device),
            'model_name': 'ExactGP',
            'noise': 0.0025,
            'lengthscale': 0.7,
            'outputscale': 1.0,
            'mean_constant': 0.0,
            'num_epochs': 50,
            'fix_lengthscale': False,
            
            # Privacy parameters
            'privacy_idx': torch.arange(0, len(train_y), device=train_x.device),
            'epsilon': 0.1,
            'delta': 0.01,
            'sensitivity': 1.0,
            'seed': 0,
            'num_attempts': 7,
            'max_iterations': 1000,
            'num_samples': 100,
            'verbose': False,
            'normalize': True, # Normalization flag for both inputs and targets
            'optimizer': 'adam', # 'grad' or 'adam'
            'method': 'cloaking',  # 'cloaking' or 'standard'

            # Analytical MLE parameter
            'use_analytical_mle': False,
        }
        
        # Update with user-provided parameters
        self.params.update(kwargs)
        
        # # Assert epsilon constraint from Hall et al. 2013
        # assert self.params['epsilon'] <= 1, "The proof in Hall et al. 2013 is restricted to values of epsilon <= 1."


        # Store original train_x and train_y
        self.orig_train_x = train_x
        self.orig_train_y = train_y
        
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
        """Train the GP model with parameters from initialization."""
        
        model = GPModelTrainer(
            train_x=self.train_x, 
            train_y=self.train_y,
            likelihood=self.params['likelihood'],
            noise=self.params['noise'],
            lengthscale=self.params['lengthscale'],
            outputscale=self.params['outputscale'],
            mean_constant=self.params['mean_constant'],
            model_name=self.params['model_name'],
            noiseless=self.params['noiseless'],
            fix_lengthscale=self.params['fix_lengthscale'],
            use_analytical_mle = self.params['use_analytical_mle'],
        )
    
        if self.params['train']:
            model.train(
                num_epochs=self.params['num_epochs'],
            )
        self.gpmodel = model.model
    
    def calc_msense(self, matrix):
        """Calculate matrix sensitivity for standard method.
        
        Originally returned the infinity norm, but we've developed an improved value 
        which only cares about values of the same sign.
        
        :param matrix: Input matrix to compute sensitivity for
        :return: The matrix sensitivity measure
        """
        # Get positive and negative parts
        matrix_copy = matrix.clone()
        positive_sum = torch.sum(matrix_copy.clamp(min=0), dim=1)
        negative_sum = torch.sum((-matrix_copy).clamp(min=0), dim=1)
        
        v1 = torch.max(torch.abs(positive_sum))
        v2 = torch.max(torch.abs(negative_sum))
        
        return torch.max(torch.tensor([v1, v2], device=matrix.device))
    
    def calc_inv_cov(self):
        """Calculate inverse covariance matrix for standard method."""
        
        # Get the model into eval mode
        self.gpmodel.eval()
        
        with torch.no_grad():
            sigma_sqr = self.gpmodel.likelihood.noise.item()
            k_nn = self.gpmodel.covar_module(self.train_x, self.train_x)
            
            # Add noise to diagonal - convert sigma_sqr to a tensor with same device as k_nn
            sigma_sqr_tensor = torch.tensor(sigma_sqr, device=self.train_x.device)
            k_nn_plus_noise = k_nn.add_diag(sigma_sqr_tensor)
            
            # Compute inverse
            eye_n = torch.eye(k_nn_plus_noise.size(-1), device=self.train_x.device)
            self.inv_cov = k_nn_plus_noise.inv_matmul(eye_n)
    
    def calc_m(self, lambdas, cloak_cols):
        """Calculate the covariance matrix M as the lambda-weighted sum of c*c^T.
        
        :param lambdas: Vector of lambda weights
        :param cloak_cols: List of column vectors from the cloaking matrix
        :return: The weighted covariance matrix M
        """
        d = cloak_cols[0].size(0)
        m_matrix = torch.zeros((d, d), device=lambdas.device)
        
        for lambda_val, col in zip(lambdas, cloak_cols):
            cc_t = torch.matmul(col, col.t())
            m_matrix += lambda_val * cc_t
            
        return m_matrix
    
    def calc_loss(self, lambdas, cloak_cols):
        """Calculate the loss function L = -log |M| + sum(lambda_i * (1-c^T M^-1 c)).
        
        :param lambdas: Vector of lambda weights
        :param cloak_cols: List of column vectors from the cloaking matrix
        :return: The loss value
        """
        m_matrix = self.calc_m(lambdas, cloak_cols)
        m_inv = torch.pinverse(m_matrix)
        
        term_sum = 0
        for lambda_val, col in zip(lambdas, cloak_cols):
            # term_sum += lambda_val * ( 1 - col.t() @ m_matrix.solve(col) )
            term_sum += lambda_val * (1 - torch.matmul(torch.matmul(col.t(), m_inv), col).item())

        # Use log determinant of inverse (= -log determinant of M)
        sign, logdet = torch.slogdet(m_inv)
        if sign <= 0:  # Handling non-positive definite matrices
            return 1000 # Large penalty value
        return logdet + term_sum
    
    
    def calc_dl_dl(self, lambdas, cloak_cols):
        """Calculate the gradient dL/dlambda_j.
        
        :param lambdas: Vector of lambda weights
        :param cloak_cols: List of column vectors from the cloaking matrix
        :return: Gradient vector
        """
        m_matrix = self.calc_m(lambdas, cloak_cols)
        m_inv = torch.pinverse(m_matrix)
        
        grads = torch.zeros_like(lambdas)
        for j in range(len(cloak_cols)):
            grads[j] = -torch.trace(torch.matmul(m_inv, torch.matmul(cloak_cols[j], cloak_cols[j].t())))
        
        return grads + 1.0

    def find_lambdas_grad(self, cloak_cols, max_it=700, verbose=False, seed=0):
        """Find optimal lambda values using gradient descent.
        
        :param cloak_cols: List of column vectors from the cloaking matrix
        :param max_it: Maximum number of iterations
        :param verbose: Whether to print progress information
        :return: Vector of optimal lambda values
        """
        device = cloak_cols[0].device
        # Initialize lambdas as small random values
        # Set seed for reproducibility
        torch.manual_seed(seed)
        lambdas = 0.1 + torch.rand(len(cloak_cols), device=device) * 0.8 # uniform(0.1, 0.9)
        lr = 0.05  # Learning rate
        
        for it in range(max_it):
            lambdas_before = lambdas.clone()
            delta_lambdas = -self.calc_dl_dl(lambdas, cloak_cols) * lr
            lambdas = lambdas + delta_lambdas
            lambdas[lambdas < 0] = 0  # Enforce non-negativity constraint
            
            if torch.max(torch.abs(lambdas_before - lambdas)) < 1e-5:
                return lambdas
                
            if verbose and it % 100 == 0:
                print(".", end='', flush=True)
                
        if verbose:
            print("Stopped before convergence")
            
        return lambdas
    
    def find_lambdas_repeat(self, cloak_cols, n_attempts=7, n_its=1000, verbose=False):
        """Call find_lambdas repeatedly with different initial values to avoid local minima.
        
        :param cloak_cols: List of column vectors from the cloaking matrix
        :param n_attempts: Number of attempts with different starting points
        :param n_its: Maximum iterations per attempt
        :param verbose: Whether to print progress information
        :return: Vector of optimal lambda values
        """
        best_log_det_m = float('inf')
        best_lambdas = None
        count = 0
        
        while best_lambdas is None and count < 1000:
            for it in range(n_attempts):
                if verbose:
                    print("*", end='', flush=True)
                if self.params['optimizer'] == 'grad':
                    lambdas = self.find_lambdas_grad(cloak_cols, n_its, verbose=verbose, seed=it+self.params['seed'])
                elif self.params['optimizer'] == 'adam':
                    lambdas = self.find_lambdas_adam(cloak_cols, n_its, verbose=verbose, seed=it+self.params['seed'])
                
                if torch.min(lambdas) < -0.01:
                    continue
                    
                m_matrix = self.calc_m(lambdas, cloak_cols)
                sign, logdet = torch.slogdet(m_matrix)
                
                if sign <= 0:
                    log_det_m = -1000  # Penalty for non-positive definite matrices
                else:
                    log_det_m = logdet
                    
                if log_det_m < best_log_det_m:
                    best_log_det_m = log_det_m
                    best_lambdas = lambdas.clone()
            
            count += 1
                
        if best_lambdas is None:
            raise ValueError('Failed to find valid lambda values after 1000 attempts')
                
        return best_lambdas
    

    def find_lambdas_adam(self, cloak_cols, max_it=700, verbose=False, seed=0):
        """
        Optimize lambda values using Adam with a reparameterization.
        We set lambda = softplus(theta) to ensure lambdas are always positive.
        """
        device = cloak_cols[0].device
        torch.manual_seed(seed)
        theta = torch.randn(len(cloak_cols), device=device) * 0.1
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta], lr=0.05)
        best_loss = float('inf')
        best_lambdas = None
        for it in range(max_it):
            optimizer.zero_grad()
            lambdas = torch.nn.functional.softplus(theta)
            loss = self.calc_loss(lambdas, cloak_cols)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lambdas = lambdas.detach().clone()
            if verbose and it % 100 == 0:
                print(f"Iteration {it}: loss = {loss.item()}")
            if theta.grad.norm() < 1e-5:
                break
        return best_lambdas


    def calc_delta(self, lambdas, cloak_cols):
        """Calculate Delta that satisfies sup{D~D'} ||M^-.5(v_D-v_D')||_2 <= Delta.
        
        :param lambdas: Vector of lambda weights
        :param cloak_cols: List of column vectors from the cloaking matrix
        :return: The calculated Delta value
        """
        m_matrix = self.calc_m(lambdas, cloak_cols)
        m_inv = torch.pinverse(m_matrix)
        
        max_c_minv_c = torch.tensor(-float('inf'), device=lambdas.device)
        for lambda_val, col in zip(lambdas, cloak_cols):
            c_minv_c = torch.matmul(torch.matmul(col.t(), m_inv), col)
            if c_minv_c > max_c_minv_c:
                max_c_minv_c = c_minv_c
                
        return max_c_minv_c.item()
    
    def get_c_cloaking(self, test_x):
        """Compute the cloaking matrix (K_starN . K_NN^-1) for cloaking method.
        
        :param test_x: Test input locations
        :return: The cloaking matrix
        """
        # Get the model into eval mode
        self.gpmodel.eval()
        
        with torch.no_grad():
            # Extract privacy indices
            privacy_idx = self.params['privacy_idx']
            train_x_privacy = self.train_x[privacy_idx]
            
            # Compute covariance matrices for privacy points
            k_privacy = self.gpmodel.covar_module(train_x_privacy, train_x_privacy)
            k_test_privacy = self.gpmodel.covar_module(test_x, train_x_privacy)
            
            # Calculate cloaking matrix for privacy points
            # Handle potential numerical issues with large matrices
            if train_x_privacy.size(0) < 1000:
                c_matrix = k_test_privacy @ k_privacy.inv_matmul(
                    torch.eye(k_privacy.size(-1), device=k_privacy.device)
                )
            else:
                # Use more numerically stable approach for large matrices
                root_decomp = k_privacy.root_decomposition()
                inv_root = root_decomp.root_inv_decomposition()
                c_matrix = k_test_privacy.matmul(inv_root.matmul(inv_root.t()))
            
        return c_matrix
    
    def draw_noise_samples_standard(self, test_x, num_samples=1):
        """Generate differentially private noise samples using the standard method.
        
        :param test_x: Test input locations
        :param num_samples: Number of noise samples to generate
        :return: Tuple of (mean predictions, prediction covariance, noise covariance, noise samples)
        """
            
        # Calculate inverse covariance if not already done
        if not hasattr(self, 'inv_cov'):
            self.calc_inv_cov()
        
        # Get covariance matrices
        with torch.no_grad():
            self.gpmodel.eval()
            test_cov = self.gpmodel.covar_module(test_x, test_x)
            
            # Calculate matrix sensitivity
            msense = self.calc_msense(self.inv_cov)
            
            # Get mean predictions
            output = self.gpmodel(test_x)
            self.pred_mean = output.mean
            self.pred_cov = output.covariance_matrix
            
            # Compute noise covariance
            c_param = math.sqrt( 2.0 * math.log(2.0 / self.params['delta']) )
            scale_factor = (self.params['sensitivity'] * c_param * msense / self.params['epsilon']) ** 2
            
            # Scale covariance
            self.noise_cov = scale_factor * test_cov.evaluate()
            
            # Generate noise samples with robustness to non-PD matrices
            self.noise_samples = self.sample_from_covariance(self.noise_cov, num_samples, test_x.device)
        

    
    def draw_noise_samples_cloaking(self, test_x, num_samples=1):
        """Generate differentially private noise samples using the cloaking method.
        
        :param test_x: Test input locations
        :param num_samples: Number of noise samples to generate
        :return: Tuple of (mean predictions, prediction covariance, noise covariance, noise samples)
        """
            
        # Ensure model is in eval mode
        self.gpmodel.eval()
        
        # Compute the cloaking matrix
        c_matrix = self.get_c_cloaking(test_x)
        
        # Extract column vectors for optimization
        cloak_cols = [c_matrix[:, i:i+1] for i in range(c_matrix.size(1))]
        
        # Find optimal lambda values
        lambdas = self.find_lambdas_repeat(
            cloak_cols, 
            n_attempts=self.params['num_attempts'],
            n_its=self.params['max_iterations'],
            verbose=self.params['verbose']
        )
        
        # Compute the covariance matrix M
        m_matrix = self.calc_m(lambdas, cloak_cols)
        
        # Compute DP parameters and noise covariance
        c_param = math.sqrt( 2.0 * math.log(2.0 / self.params['delta']) )
        delta_param = self.calc_delta(lambdas, cloak_cols)
        
        # Scale factor for noise
        scale_factor = (self.params['sensitivity'] * c_param * torch.sqrt(torch.tensor(delta_param)) / self.params['epsilon']) ** 2
        
        # Compute the sample covariance
        self.noise_cov = scale_factor * m_matrix
        
        # Get model predictions
        with torch.no_grad():
            self.gpmodel.eval()
            output = self.gpmodel(test_x)
            self.pred_mean = output.mean
            self.pred_cov = output.covariance_matrix
            
        # Generate noise samples with robustness to non-PD matrices
        self.noise_samples = self.sample_from_covariance(self.noise_cov, num_samples, test_x.device)
        
    
    def draw_noise_samples(self, test_x, num_samples=1):
        """Generate differentially private noise samples using the selected method.
        
        :param test_x: Test input locations
        :param num_samples: Number of noise samples to generate
        :return: Tuple of (mean predictions, prediction covariance, noise covariance, noise samples)
        """
        # Make sure model is trained
        if not hasattr(self, 'model'):
            self.train()

        if self.params['method'] == 'standard':
            self.draw_noise_samples_standard(test_x, num_samples)
        else:  # Default to cloaking method
            self.draw_noise_samples_cloaking(test_x, num_samples)
    

    def draw_prediction_samples(self, test_x, num_samples=1):
        """Generate differentially private predictions.
        
        :param test_x: Test input locations
        :param num_samples: Number of prediction samples
        :return: Tuple of (DP mean predictions, DP covariance, DP samples)
        """
        test_x = self.normalize_x(test_x)
        self.draw_noise_samples(test_x, num_samples)
        
        # Add noise to mean for DP predictions
        dp_samples = self.pred_mean.unsqueeze(0) + self.noise_samples # [num_samples. n_test]
        
        # Calculate privacy-preserving mean and covariance
        self.privacy_mean = dp_samples.mean(dim=0)
        self.privacy_samples = dp_samples
        privacy_cov = dp_samples.t().cov() / num_samples #dp_samples.t().cov() #[self.pred_cov, dp_samples.t().cov()]

        # Denormalize the mean and variance if normalization was applied
        if self.params['normalize']:
            self.privacy_mean = self.privacy_mean * self.params['train_y_std'] + self.params['train_y_mean']
            privacy_cov = privacy_cov* (self.params['train_y_std'] ** 2)
            self.privacy_samples = self.privacy_samples * self.params['train_y_std'] + self.params['train_y_mean']
        
        self.privacy_cov = self.ensure_pd(privacy_cov, epsilon=0)# Ensure the combined covariance is positive-definite
        return self.privacy_mean, self.privacy_cov
    
    def forward(self, x):
        """Make predictions with the DPGP model.
        
        :param x: Test input locations
        :return: GP distribution with differentially private predictions
        """
        # Make DP prediction samples
        mean_x, covar_x = self.draw_prediction_samples(x, num_samples=self.params['num_samples'])
        
        # Return as a gpytorch MultivariateNormal distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



    def sample_from_covariance(self, cov_matrix, num_samples=1, device=None):
        """Sample from a multivariate normal distribution with given covariance matrix.
        Uses multiple methods to ensure numerical stability.
        
        :param cov_matrix: Covariance matrix
        :param num_samples: Number of samples to generate
        :param device: Torch device
        :return: Samples from multivariate normal
        """
        n = cov_matrix.size(0)
        
        # Method 1: Try using GPyTorch's MultivariateNormal and rsample
        try:
            mvn = gpytorch.distributions.MultivariateNormal(
                torch.zeros(n, device=device), 
                covariance_matrix=cov_matrix
            )
            return mvn.rsample(torch.Size([num_samples]))
        except Exception as e:
            if self.params['verbose']:
                print(f"GPyTorch sampling failed: {e}, trying method 2...")
        
        # Method 2: Try using Cholesky with a small jitter
        try:
            # Add a small jitter to the diagonal to ensure positive definiteness
            jitter = 1e-6
            jitter_matrix = torch.eye(n, device=device) * jitter
            L = torch.linalg.cholesky(cov_matrix + jitter_matrix)
            torch.manual_seed(self.params['seed'])
            z = torch.randn(num_samples, n, device=device)
            return torch.matmul(z, L.t())
        except Exception as e:
            if self.params['verbose']:
                print(f"Cholesky with jitter failed: {e}, trying method 3...")
        
        # Method 3: Use eigendecomposition approach
        try:
            # Compute eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
            # Handle numerical issues by setting tiny negative eigenvalues to a small positive value
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            
            # Use eigenvalues and eigenvectors to generate samples
            torch.manual_seed(self.params['seed'])
            z = torch.randn(num_samples, n, device=device)
            scaled_eigenvectors = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)
            return torch.matmul(z, scaled_eigenvectors.t())
        except Exception as e:
            if self.params['verbose']:
                print(f"Eigendecomposition approach failed: {e}, using fallback method...")
        
        # Method 4: Fallback to a simplified approach with more aggressive regularization
        jitter = 1e-4 * torch.max(torch.diag(cov_matrix))
        reg_cov = cov_matrix + torch.eye(n, device=device) * jitter
        
        # Use SVD as the ultimate fallback
        U, S, V = torch.linalg.svd(reg_cov)
        
        # Ensure all singular values are positive
        S = torch.clamp(S, min=1e-10)
        
        # Generate samples
        torch.manual_seed(self.params['seed'])
        z = torch.randn(num_samples, n, device=device)
        return torch.matmul(z, torch.matmul(U, torch.diag(torch.sqrt(S))))
    

    def ensure_pd(self, matrix, epsilon=1e-4):
        """Ensure a matrix is positive definite by checking eigenvalues and adding jitter if needed.
        
        :param matrix: Input matrix to ensure positive definiteness
        :param epsilon: Minimum eigenvalue threshold
        :return: Positive definite matrix
        """
        # Check if matrix is None or has zero size
        if matrix is None or matrix.numel() == 0:
            return torch.eye(1, device=self.train_x.device) * epsilon
        
        # Make sure the matrix is symmetric first
        matrix = 0.5 * (matrix + matrix.t())
        
        # Try eigendecomposition approach
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            min_eig = torch.min(eigenvalues)
            
            # If smallest eigenvalue is already positive and sufficiently large, return original matrix
            if min_eig > epsilon:
                return matrix
            
            # If minimum eigenvalue is negative or too small, return a diagonal matrix with clamped values
            else:
                diag_values = torch.diag(matrix)
                diag_values = torch.clamp(diag_values, min=epsilon)
                return torch.diag(diag_values)
                   
            # # Otherwise, fix the eigenvalues and reconstruct the matrix
            # else:
            #     eigenvalues = torch.clamp(eigenvalues, min=epsilon)
            #     fixed_matrix = torch.matmul(
            #         eigenvectors, 
            #         torch.matmul(torch.diag(eigenvalues), eigenvectors.t())
            #     )
                
            #     # Ensure symmetry again after reconstruction
            #     fixed_matrix = 0.5 * (fixed_matrix + fixed_matrix.t())
                
            #     return fixed_matrix
        
        except Exception as e:
            if self.params['verbose']:
                print(f"Error in ensure_pd eigendecomposition: {e}")
    

    # -------------------------------
    # Helper Functions for Normalization
    # -------------------------------
    def normalize_x(self, x):
        """Normalize new inputs x using training input statistics stored in self.params."""
        if self.params['normalize']:
            return (x - self.params['train_x_mean']) / self.params['train_x_std']
        return x