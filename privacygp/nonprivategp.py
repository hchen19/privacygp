import torch
import gpytorch
from .train import GPModelTrainer

class NonPrivateGP(torch.nn.Module):
    """
    Standard Gaussian Process trained on true observations without any modifications.
    """
    
    def __init__(
            self, 
            train_x,
            train_y,
            **kwargs,
        ):
        """
        Initialize the NonPrivateGP model with training data and parameters.

        :param train_x: Training input data (full dataset)
        :param train_y: Training target data (full dataset)
        :param **kwargs: Additional parameters for GP training
        """
        super(NonPrivateGP, self).__init__()

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
            'noise': 0.001,
            'lengthscale': 0.25,
            'outputscale': 1.0,
            'mean_constant': 0.0,
            'num_epochs': 200,
            'fix_lengthscale': False,
            'normalize': True,
            
            # Compatibility parameters
            'use_analytical_mle': True,
            'zero_mean': False,
        }
        
        # Update with user-provided parameters
        self.params.update(kwargs)
        
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
        """Train the GP model on true data points."""
        
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
            zero_mean=self.params['zero_mean'],
        )
    
        if self.params['train']:
            model.train(
                num_epochs=self.params['num_epochs'],
            )
        self.gpmodel = model.model
        
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
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def normalize_x(self, x):
        """Normalize new inputs x using training input statistics stored in self.params."""
        if self.params['normalize']:
            return (x - self.params['train_x_mean']) / self.params['train_x_std']
        return x