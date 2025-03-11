"""
ExcludeGP: A GP implementation that excludes privacy points from training.

This class trains a standard Gaussian Process on only the non-private data points,
providing a baseline for comparison with privacy-preserving GP methods.
"""

import torch
import gpytorch
from .train import GPModelTrainer

class ExcludeGP(torch.nn.Module):
    """
    Gaussian Process implementation that excludes specified privacy points.
    
    This model completely removes privacy-sensitive data from training,
    providing a simple baseline approach to privacy protection.
    """
    
    def __init__(
            self, 
            train_x,
            train_y,
            **kwargs,
        ):
        """
        Initialize the ExcludeGP model with training data and parameters.

        :param train_x: Training input data (full dataset)
        :param train_y: Training target data (full dataset)
        :param privacy_idx: Indices of privacy-sensitive data points to exclude
        :param **kwargs: Additional parameters for GP training
        """
        super(ExcludeGP, self).__init__()

        # Store original data
        self.origin_train_x = train_x
        self.origin_train_y = train_y

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
            'normalize': False, # Normalization flag for both inputs and targets
            
            # Required parameters for compatibility
            'privacy_idx': torch.arange(0, device=train_x.device),  # Default empty array
            'use_analytical_mle': False,
        }
        
        # Update with user-provided parameters
        self.params.update(kwargs)
        
        # Create mask for non-privacy points
        self.exclude_mask = self.create_exclusion_mask(train_x.shape[0], self.params['privacy_idx'])
        
        # Extract non-private training data
        train_x = train_x[self.exclude_mask]
        train_y = train_y[self.exclude_mask]
        
        print(f"Created ExcludeGP model: {len(train_x)} points (excluded {len(self.params['privacy_idx'])} privacy points)")

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

    def create_exclusion_mask(self, total_length, privacy_idx):
        """
        Create a boolean mask where True indicates non-private points to keep.
        
        :param total_length: Total number of data points
        :param privacy_idx: Indices of privacy points to exclude
        :return: Boolean mask for non-privacy points
        """
        # Create a tensor of all True values
        mask = torch.ones(total_length, dtype=torch.bool, device=privacy_idx.device)
        
        # Set privacy indices to False (to exclude them)
        if len(privacy_idx) > 0:
            mask[privacy_idx] = False
            
        return mask
        
    def train(self):
        """Train the GP model on non-private data points."""
        
        if len(self.train_x) == 0:
            raise ValueError("No non-private data points available for training. All points were excluded.")
            
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
            use_analytical_mle=self.params['use_analytical_mle'],
        )
    
        if self.params['train']:
            model.train(
                num_epochs=self.params['num_epochs'],
            )
        self.gpmodel = model.model
        
        # Ensure we keep a record of the original data for comparison with other methods
        self.origin_non_privacy_x = self.train_x
        self.origin_non_privacy_y = self.train_y
        
    def test(self, test_x):
        """
        Make predictions at test points using the trained model.
        
        :param test_x: Test input points
        :return: Tuple of (mean, covariance) predictions
        """
        test_x = self.normalize_x(test_x)
        if not hasattr(self, 'gpmodel'):
            self.train()
            
        # Set model to evaluation mode
        self.gpmodel.eval()
        self.gpmodel.likelihood.eval()
        
        # Make predictions
        with torch.no_grad():
            preds = self.gpmodel(test_x)
            if self.params['noiseless'] is False:
                preds = self.gpmodel.likelihood(preds)
            self.privacy_mean = preds.mean
            self.privacy_cov = preds.covariance_matrix

            # Denormalize the mean and variance if normalization was applied
            if self.params['normalize']:
                self.privacy_mean = self.privacy_mean * self.params['train_y_std'] + self.params['train_y_mean']
                self.privacy_cov = self.privacy_cov* (self.params['train_y_std'] ** 2)
            
            return self.privacy_mean, self.privacy_cov
            
    def forward(self, x):
        """
        Forward pass to get predictions for test points.
        
        :param x: Test input points
        :return: GPyTorch MultivariateNormal distribution
        """
        mean_x, covar_x = self.test(x)
        return
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


    # -------------------------------
    # Helper Functions for Normalization
    # -------------------------------
    def normalize_x(self, x):
        """Normalize new inputs x using training input statistics stored in self.params."""
        if self.params['normalize']:
            return (x - self.params['train_x_mean']) / self.params['train_x_std']
        return x