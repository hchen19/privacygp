from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.likelihood = likelihood
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
        self.covar_module = ScaleKernel(base_kernel)        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)