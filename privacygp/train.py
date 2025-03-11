import math
from tqdm import tqdm
import torch
import gpytorch

from .exactgp import ExactGPModel


class GPModelTrainer(object):
    def __init__(
            self, 
            train_x, train_y,
            **kwargs
        ):
        self.train_x = train_x
        self.train_y = train_y
        
        # Check if the model should be noiseless
        self.noiseless = kwargs.get('noiseless', False)
        self.fix_lengthscale = kwargs.get('fix_lengthscale', False)
        self.use_analytical_mle = kwargs.get('use_analytical_mle', False) # Add analytical method flag
        self.zero_mean = kwargs.get('zero_mean', False)
        likelihood = kwargs.get('likelihood', gpytorch.likelihoods.GaussianLikelihood().to(train_x.device))
        self.model_name = kwargs.get('model_name', 'ExactGP')

        # For noiseless models, we still create a likelihood but set noise to a very small value
        # This is because GPyTorch's ExactGP requires a likelihood
        if self.noiseless:
            # Create a custom noise constraint that allows very small values
            noise_constraint = gpytorch.constraints.GreaterThan(1e-8)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint
            ).to(train_x.device)
            # Initialize with a very small noise value
            likelihood.initialize(noise=1e-8)
        else:
            likelihood = kwargs.get('likelihood', gpytorch.likelihoods.GaussianLikelihood().to(train_x.device))

        # Initialize the model
        if self.model_name == 'ExactGP':
            self.model = ExactGPModel(train_x, train_y, likelihood).to(train_x.device)
        

        # Set initial hyperparameters
        noise = kwargs.get('noise', 0.0025) if not self.noiseless else 1e-8
        lengthscale = kwargs.get('lengthscale', 0.7)
        outputscale = kwargs.get('outputscale', 1)
        mean_constant = kwargs.get('mean_constant', 0)

        # Initialize hyperparameters
        hypers = {
            'likelihood.noise_covar.noise': noise,
            'covar_module.base_kernel.lengthscale': lengthscale,
            'covar_module.outputscale': outputscale,
            'mean_module.constant': mean_constant,
        }
        self.model.initialize(**hypers)


    def compute_analytical_mle(self):
        """
        Compute the Maximum Likelihood Estimate for GP parameters analytically
        using the formulas from the provided equations.
        
        This method computes:
        1. The optimal constant mean (beta)
        2. The optimal kernel variance (sigma^2)
        
        This version uses GPyTorch's linear operators for efficient computation,
        avoiding explicit matrix evaluations where possible.
        
        :return: Updated hyperparameters (Dict)
        """
        
        # Compute the kernel matrix K as a lazy tensor
        with torch.no_grad():
            # Get the base kernel matrix (without the outputscale)
            base_kernel = self.model.covar_module.base_kernel
            corr_x_x = base_kernel(self.train_x, self.train_x)
            
            # Add small jitter to ensure positive definiteness
            # Using GPyTorch's built-in method to add diagonal
            jitter = 1e-6
            corr_x_x = corr_x_x.add_jitter(jitter)
            
            # Prepare vector of ones for computing beta
            ones = torch.ones(self.train_x.size(-2), 1, device=self.train_x.device, dtype=self.train_x.dtype)
            y_vec = self.train_y.reshape(-1, 1)
            
            # Solve the linear systems efficiently
            
            # Formula: beta = (1^T K^-1 y) / (1^T K^-1 1)
            if self.zero_mean:
                beta = 0.
            else:
                numerator = ones.t() @ corr_x_x.solve(y_vec)
                denominator = ones.t() @ ( corr_x_x.solve(ones) )
                beta = (numerator / denominator).item()
            
            
            # Compute centered response y_centered = y - beta
            y_centered_vec = (self.train_y - beta).reshape(-1, 1)
            
            # Compute optimal sigma^2 (outputscale)
            # Formula: sigma^2 = (y_centered^T K^-1 y_centered) / n
            # Using linear operations 
            corr_inv_y_centered = corr_x_x.solve(y_centered_vec)
            sigma_squared = (y_centered_vec.t() @ corr_inv_y_centered).item() / (self.train_x.size(-2))
            if sigma_squared < 1e-04:
                sigma_squared *= 15

            # Update model hyperparameters with the MLE values
            self.model.mean_module.constant.data.fill_(beta)
            self.model.covar_module.outputscale = sigma_squared
            

            # # For outputscale, we need to handle the constraints properly
            # # Get the constraint for the outputscale parameter
            # # Convert the desired value through the inverse transform
            # # This converts from the constrained space to the unconstrained space
            # # where the parameter actually lives
            # constraint = self.model.covar_module.raw_outputscale_constraint
            # raw_value = constraint.inverse_transform(torch.tensor(sigma_squared, device=self.train_x.device, dtype=self.train_x.dtype))
            # self.model.covar_module.raw_outputscale.data.fill_(raw_value.item()) # Now update the raw parameter
            
            # Since we're using analytical MLE, we don't need to recompute lengthscale
            # We keep the existing lengthscale value
            
            return {
                'mean_constant': beta,
                'outputscale': sigma_squared
            }

    def analytical_posterior(self, test_x, return_cov=True, diag=True):
        """
        Compute the analytical posterior directly without using GPyTorch's methods.
        This is analogous to the fitting method used in the R code.
        
        This version uses GPyTorch's linear operators for efficient computation,
        avoiding explicit matrix evaluations where possible.
        
        :param test_x: (torch.Tensor) Test points 
        :param return_cov: (bool) Whether to return the covariance
        :param diag: (bool) Whether to return diagonal of covariance or full matrix
        :return: (mean, variance) if return_cov=True, else mean
        """
        
        # Get current parameters
        beta = self.model.mean_module.constant.item()
        sigma_squared = self.model.covar_module.outputscale.item()
        
        # Compute kernel matrices using lazy tensors
        with torch.no_grad():
            # Get base kernel (without outputscale)
            base_kernel = self.model.covar_module.base_kernel
            
            # Compute K(X, X) as a lazy tensor multiplied by sigma_squared (outputscale)
            cov_x_x = sigma_squared * base_kernel(self.train_x, self.train_x)
            
            # Add noise if not noiseless
            if not self.noiseless:
                noise = self.model.likelihood.noise.item()
                cov_x_x = cov_x_x.add_diag(noise)
            
            # Compute K(X*, X) - cross-covariance between test and training points
            cov_xstar_x = sigma_squared * base_kernel(test_x, self.train_x)
            
            # Compute K(X*, X*) - test points covariance (only if needed)
            cov_xstar_xstar = sigma_squared * base_kernel(test_x, test_x, diag=diag)

            # Centered training targets
            y_centered = self.train_y - beta
            y_centered_vec = y_centered.reshape(-1, 1)
            
            # Solve the linear system K_XX^(-1) * y_centered efficiently
            cov_xx_inv_y = cov_x_x.solve(y_centered_vec)
            
            # Compute posterior mean: f(X*) = beta + K(X*,X) [K(X,X)]^(-1) (y - beta)
            # We directly multiply K_star_X_lazy by the solution vector
            mean = beta + (cov_xstar_x @ cov_xx_inv_y).squeeze(-1)
            
            if return_cov:
                # Compute posterior covariance efficiently
                # Cov = K(X*,X*) - K(X*,X) [K(X,X)]^(-1) K(X,X*)
                
                if diag:
                    # For diagonal covariance (variances), we compute it directly
                    # Using the woodbury identity to avoid explicitly inverting K_XX
                    # We compute the diagonal of K_star_star - K_star_X K_XX^(-1) K_star_X^T
                    
                    # First, we compute K_XX^(-1/2) K_star_X^T 
                    # via solving linear system K_XX^(1/2) X = K_star_X^T
                    
                    # Since we need the diagonal, we can leverage batched operations
                    variances = torch.zeros(test_x.size(-2), device=self.train_x.device, dtype=self.train_x.dtype)
                    
                    # Compute the posterior variance reduction
                    # We use batched computations with solve_triangular for efficiency
                    if test_x.size(-2) < 1000:  # For moderate-sized test sets
                        # Direct approach: solve K_XX * X = K_star_X^T
                        tmp = cov_x_x.solve(cov_xstar_x.t())
                        variance_reduction = (cov_xstar_x @ tmp).diag()
                        variances = cov_xstar_xstar - variance_reduction
                    else:
                        # For large test sets, process in batches to avoid memory issues
                        batch_size = 500
                        for i in range(0, test_x.size(-2), batch_size):
                            end_idx = min(i + batch_size, test_x.size(-2))
                            batch_indices = torch.arange(i, end_idx, device=self.train_x.device)
                            cov_xstar_x_batch = cov_xstar_x[batch_indices]
                            
                            # Solve K_XX * X = K_star_X_batch^T
                            tmp = cov_x_x.solve(cov_xstar_x_batch.t())
                            var_reduction = (cov_xstar_x_batch @ tmp).diag()
                            variances[batch_indices] = cov_xstar_xstar[batch_indices] - var_reduction
                    
                    return mean, variances
                else:
                    # For full covariance matrix
                    # We leverage the Cholesky decomposition for numerical stability
                    # and efficient matrix operations
                    
                    # Compute: K_star_star - K_star_X K_XX^(-1) K_star_X^T
                    tmp = cov_x_x.solve(cov_xstar_x.t())
                    cov = cov_xstar_xstar - cov_xstar_x @ tmp
                    
                    # Convert to dense matrix at the very end
                    return mean, cov
            else:
                return mean

    def train(self, **kwargs):
        # Get options
        num_epochs = kwargs.get('num_epochs', 50)

        # If analytical MLE is requested, use it instead of optimization
        if self.use_analytical_mle:
            
            # Compute parameters analytically
            mle_params = self.compute_analytical_mle()
            
            print(f"Analytical MLE parameters: β = {mle_params['mean_constant']:.6f}, σ² = {mle_params['outputscale']:.6f}")
            
            # If we're not fixing the noise, we could compute it analytically too
            # For now, we're keeping the provided noise value
            
            return
        
        # If not using analytical MLE, proceed with standard optimization
        self.model.train()
        self.model.likelihood.train()

        params_optim = [
            # {'params': self.model.parameters()},

            {'params': self.model.mean_module.parameters()},
            {'params': [self.model.covar_module.raw_outputscale]}
            # {'params': self.model.likelihood.parameters()},
            # #{'params': self.model.covar_module.base_kernel.parameters()},
        ]   
        if not self.fix_lengthscale:
            params_optim.append({'params': self.model.covar_module.base_kernel.raw_lengthscale})
        if not self.noiseless:
            params_optim.append({'params': self.model.likelihood.parameters()})



        epochs_iter = tqdm(range(num_epochs), desc=f"Training {self.model_name}")
        
        optimizer = torch.optim.Adam(params_optim, lr=0.1, weight_decay=1e-4)
        stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        for i in epochs_iter:
            optimizer.zero_grad() # Zero gradients from previous iteration
            output = self.model(self.train_x) # Output from model
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            stepLR.step()
            epochs_iter.set_postfix(loss=loss.item())



    def test(
            self, 
            test_x, test_y=None, 
            **kwargs
        ):
        """Test the model on new data points."""
        noiseless = kwargs.get('noiseless', False)
        diag = kwargs.get('diag', True)
        return_cov = kwargs.get('return_cov', True)
        verbose = kwargs.get('verbose', False)

        # If using analytical posterior calculation
        if self.use_analytical_mle:
            pred_mean, pred_cov = self.analytical_posterior(test_x, return_cov=True, diag=diag)
            
            if verbose and test_y is not None:
                error = (pred_mean - test_y).abs().mean()
                print(f"Test {self.model_name} MAE (Analytical): {error.item()}")
                
            if return_cov:
                return pred_mean, pred_cov
            else:
                return pred_mean
        
        # Otherwise use GPyTorch's prediction methods
        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            preds = self.model(test_x) if noiseless else self.model.likelihood(self.model(test_x))
            pred_mean = preds.mean

            if verbose and test_y is not None:
                error = (pred_mean - test_y).abs().mean()
                print(f"Test {self.model_name} MAE: {error.item()}")

            if return_cov:
                pred_cov = preds.variance if diag else preds.covariance_matrix # return vars or covs
                return pred_mean, pred_cov
            else:
                return pred_mean