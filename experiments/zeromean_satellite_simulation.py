import csv
import time
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from torchdiffeq import odeint

from privacygp.privacygp import PrivacyGP
from privacygp.dpgp import DPGP
from privacygp.excludegp import ExcludeGP
from privacygp.noisygp import NoisyGP


# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
figs_dir = os.path.join(current_dir, 'figs')
results_dir = os.path.join(current_dir, 'results')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


class SatelliteDynamics(torch.nn.Module):
    """
    PyTorch module defining the satellite dynamics for torchdiffeq.
    
    Implements the differential equations for a satellite with optional J2 perturbation.
    """
    def __init__(self, mu_normalized, j2_normalized, use_j2=True):
        """
        Initialize the dynamics model with normalized parameters.
        
        :param mu_normalized: Normalized gravitational parameter.
        :param j2_normalized: Normalized J2 perturbation coefficient.
        :param use_j2: Boolean flag to include J2 perturbation effects.
        """
        super().__init__()
        self.mu_normalized = mu_normalized
        self.j2_normalized = j2_normalized
        self.use_j2 = use_j2
        
    def forward(self, t, state):
        """
        Compute the derivatives of the state vector.
        
        :param t: Current time (scalar).
        :param state: Current state tensor [r, r_dot, theta, theta_dot].
        :return: State derivatives.
        """
        # Extract state variables
        r = state[0]
        r_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        
        # Initialize state derivatives
        state_dot = torch.zeros_like(state)
        
        # J2 perturbation forces (initialize to zero)
        radial_force = torch.tensor(0.0, device=state.device)
        tangential_force = torch.tensor(0.0, device=state.device)
        
        # Add J2 perturbation effects if enabled
        if self.use_j2:
            # J2 perturbation forces
            radial_force = self.j2_normalized * (3/2) * (3 * torch.sin(theta)**2 - 1) / r**4
            tangential_force = -self.j2_normalized * 3 * torch.cos(theta) * torch.sin(theta) / r**4
        
        # Satellite dynamics equations
        state_dot[0] = r_dot  # dr/dt = r_dot
        state_dot[1] = -self.mu_normalized / r**2 + r * theta_dot**2 + radial_force  # dr_dot/dt
        state_dot[2] = theta_dot  # dtheta/dt = theta_dot
        state_dot[3] = -2 * r_dot * theta_dot / r + tangential_force  # dtheta_dot/dt
        
        return state_dot


def simulate_satellite_trajectory():
    """
    Simulates and plots the satellite trajectory.
    """
    # Set device - use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Constants
    d2r = torch.tensor(np.pi / 180, device=device)  # degrees to radians
    
    # LEO Parameters
    earth_radius_km = torch.tensor(6378.1363, device=device)  # km radius of earth
    altitude_km = torch.tensor(340.0, device=device)  # height of satellite in km
    orbit_radius_km = earth_radius_km + altitude_km
    normalized_radius = orbit_radius_km / earth_radius_km  # Normalized R0 = R/Re
    initial_angle_rad = torch.tensor(0.0, device=device) * d2r  # th0
    
    # Orbital parameters
    mu_km3_s2 = torch.tensor(398600.4415, device=device)  # Earth's gravitational parameter
    orbit_velocity = torch.sqrt(mu_km3_s2 / orbit_radius_km)  # orbital velocity
    orbit_period = torch.sqrt(orbit_radius_km**3 * 4 * torch.tensor(np.pi**2, device=device) / mu_km3_s2)
    
    velocity_norm = earth_radius_km / orbit_period  # Velocity normalizing factor Vb
    normalized_radial_rate = torch.tensor(0.0, device=device)  # Rd
    normalized_angular_rate = (orbit_velocity / velocity_norm) / normalized_radius  # thd
    
    # Initial state [R0, Rd, th0, thd]
    initial_state = torch.tensor([
        normalized_radius,
        normalized_radial_rate,
        initial_angle_rad,
        normalized_angular_rate
    ], device=device, dtype=torch.float32)
    
    # Number of orbits to simulate
    num_orbits = 3
    
    # Normalized time span
    t_span = torch.linspace(0, num_orbits, 301, device=device)
    
    # Enable J2 perturbation
    j2_flag = True
    
    # Calculate normalized parameters for dynamics
    mu_normalized = mu_km3_s2 * orbit_period**2 / earth_radius_km**3  # mub
    j2_coefficient = torch.tensor(1.7555e10, device=device)  # J2
    j2_normalized = j2_coefficient * orbit_period**2 / earth_radius_km**5  # J2b
    
    # Create dynamics model
    dynamics = SatelliteDynamics(mu_normalized, j2_normalized, j2_flag).to(device)
    
    # Integrate the dynamics
    with torch.no_grad():
        # odeint returns shape [time_steps, state_dim]
        states = odeint(
            dynamics,
            initial_state,
            t_span,
            method='dopri5',  # Use Dormand-Prince (RK45) method, similar to ode45 in MATLAB
            rtol=1e-6,
            atol=1e-6
        )
    
    # Convert to numpy for plotting
    t_np = t_span.detach().cpu().numpy()
    y_np = states.detach().cpu().numpy()  # Shape is [time_steps, state_dim]
    
    # Find private segment indices (between normalized time 1 and 2)
    private_indices = np.where((t_np >= 1) & (t_np < 2))[0]

    ##########################################
    # Save data to csv
    ##########################################
    # Prepare data for saving to CSV
    # Create a matrix where first column is time, next 4 columns are state variables
    csv_data = np.column_stack([t_np, y_np, np.isin(np.arange(len(t_np)), private_indices)])
    csv_headers = ['time', 'r', 'r_dot', 'theta', 'theta_dot', 'is_private']
    
    # Save data to CSV file
    csv_path = os.path.join(data_dir, 'satellite_data.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header=','.join(csv_headers), comments='')
    print(f"Trajectory data saved to: {csv_path}")
    
    plot_trajectory(t_np, y_np, private_indices)
    
    return {
        'time': t_np,
        'states': y_np,
        'private_indices': private_indices
    }

def plot_trajectory(t_np, y_np, private_indices):
    """Plots the satellite trajectory."""
    ##########################################
    # Plot 
    ##########################################
    # Plot titles
    plot_titles = [r'$r/R_e$', r'$\dot{r}/\bar{v}$', r'$\theta$ (rad)', r'$\dot{\theta}$ (rad/s)']
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(4.5*4, 4))
    
    for i in range(4):
        ax = axes[i]
        
        # Plot the entire trajectory - extract the i-th state variable for all time steps
        ax.plot(t_np, y_np[:, i], 'k', linewidth=2, label='True Trajectory')
        
        # Highlight the private segment
        ax.plot(t_np[private_indices], y_np[private_indices, i], 'r', linewidth=2, 
                label='Private Segments' if i == 0 else None)
        
        # Set labels and grid
        ax.set_xlabel('Time $T/T_{orb}$', fontsize=15)
        ax.set_title(plot_titles[i], fontsize=15)
        ax.grid(True)
    
    # Add legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), 
               ncol=2, fontsize=20)
    
    fig.suptitle('Satellite Trajectory with J2 Perturbation', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.3) # Adjust layout to make room for the title and legend
    
    fig_path = os.path.join(figs_dir, 'satellite_trajectory.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to: {fig_path}")


def get_satellite_data():
    """Get or generate the satellite data."""
    data_path = os.path.join(data_dir, 'zeromean_satellite_adata.csv')
    
    if os.path.exists(data_path):
        # Load existing data
        csv_data = np.loadtxt(data_path, delimiter=',')
        if csv_data.ndim == 1:
            csv_data = csv_data.reshape(-1, 1)
        
        t_np = np.linspace(0, 3, csv_data.shape[0])
        private_indices = np.where((t_np >= 1) & (t_np < 2))[0]
        
        return {
            'time': t_np,
            'states': csv_data,
            'private_indices': private_indices,
        }
    else:
        # Generate new data
        return simulate_satellite_trajectory()


def get_privacy_indices(data_time, sampling_interval=5):
    """
    Extract privacy indices for sampled data points within the private segment.
    
    :param data_time: Full time points array
    :param sampling_interval: Interval for sampling training data (default: 5)
    :return: Tensor of indices to use for privacy protection in the sampled dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First get the sampled time points
    sampled_time = data_time[::sampling_interval]
    
    # Identify which points in the sampled data are in the private segment (time 1-2)
    sampled_private_mask = (sampled_time >= 1) & (sampled_time <= 2)
    sampled_private_indices = np.where(sampled_private_mask)[0]
    
    # Convert to tensor
    privacy_idx = torch.tensor(sampled_private_indices, device=device)
    
    # Ensure we have at least some indices
    if len(privacy_idx) == 0:
        # Fall back to default indices if extraction failed
        default_indices = [i for i in range(len(sampled_time)) if sampled_time[i] >= 1 and sampled_time[i] < 2]
        
        if not default_indices:
            # If still no indices, use a broad default range
            privacy_idx = torch.arange(len(sampled_time)//3, 2*len(sampled_time)//3, device=device)
            print("Warning: Could not extract privacy indices, using middle third of data.")
        else:
            privacy_idx = torch.tensor(default_indices, device=device)
            print(f"Warning: Using {len(privacy_idx)} default indices from time range.")
    else:
        print(f"Using {len(privacy_idx)} privacy indices from sampled data (interval={sampling_interval}).")
    
    return privacy_idx


def run_with_metrics(
        seed,
        privacy_idx, 
        test_x=None, 
        test_y=None,
        algs=['privacy-aware', 'dp-cloaking'], 
        alphas=[0.1, 0.5], 
        epsilons=[0.5, 1],
        sampling_interval=5,
        verbose=True,
    ):
    """Run privacy GP models for all algorithms, components and alphas."""
    # Set device - use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup
    if test_x is None:
        test_x = torch.linspace(0, 3, 300).view(-1, 1)[::5]
    test_x = test_x.to(device)
    
    # Get data
    data = get_satellite_data()
    
    # Initialize metrics storage
    metrics = {
        'seed': seed,
        'time': {},
        'rmse': {},
    }

    # Initialize storage
    results = {
        'seed': seed,
        'test_x': test_x.detach().cpu(),
        'train_data': {},
        'test_data': {},
        'preds': {alpha: {} for alpha in alphas},
        'private_indices': data['private_indices'],
    }

    # Initialize result structure for each algorithm
    if 'privacy-aware' in algs:
        results['preds']['privacy-aware'] = {alpha: {} for alpha in alphas}
        results['obfuscated_data'] = {alpha: {} for alpha in alphas}
        metrics['time']['privacy-aware'] = {alpha: {} for alpha in alphas}
        metrics['rmse']['privacy-aware'] = {alpha: {} for alpha in alphas}
    
    if 'dp-cloaking' in algs:
        results['preds']['dp-cloaking'] = {eps: {} for eps in epsilons}
        metrics['time']['dp-cloaking'] = {eps: {} for eps in epsilons}
        metrics['rmse']['dp-cloaking'] = {eps: {} for eps in epsilons}
    
    if 'dp-standard' in algs:
        results['preds']['dp-standard'] = {eps: {} for eps in epsilons}
        metrics['time']['dp-standard'] = {eps: {} for eps in epsilons}
        metrics['rmse']['dp-standard'] = {eps: {} for eps in epsilons}

    if 'exclude' in algs:
        results['preds']['exclude'] = {'none': {}}
        metrics['time']['exclude'] = {'none': {}}
        metrics['rmse']['exclude'] = {'none': {}}
    
    if 'attack' in algs:
        results['preds']['attack'] = {alpha: {} for alpha in alphas}
        metrics['time']['attack'] = {alpha: {} for alpha in alphas}
        metrics['rmse']['attack'] = {alpha: {} for alpha in alphas}
    
    # Common GP parameters
    gp_params = {
        'train': True,
        'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(device),
        'model_name': 'ExactGP',
        'noise': 0.0025,
        'lengthscale': 1/400,
        'outputscale': 0.6, # [0.6, math.sqrt(9.854893e-8)]
        'mean_constant': 1.0, #1.052471,
        'num_epochs': 50,
        'privacy_idx': privacy_idx,
        'num_eigen_tol': 1e4,
        'seed': seed,#5122023,
        'noiseless': True,
        'fix_lengthscale': True,
        'use_analytical_mle': True, # Analytical MLE parameter
        'zero_mean': True,
    }

    # DP parameters
    dp_params = {
        #'epsilon': 0.1,
        'delta': 0.01,
        'sensitivity': 1,
        'num_attempts': 7,
        'max_iterations': 1000,
        'num_samples': 100,
        'optimizer': 'grad', # 'grad' or 'adam'
        'normalize': True, # Normalization flag and (later) normalization parameters
    }

    # Attack parameters
    attack_params = {
        'train': True,
        'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(device),
        'model_name': 'ExactGP',
        'noise': 0.01,
        'lengthscale': 0.7,
        'outputscale': 1.0, # [0.6, math.sqrt(9.854893e-8)]
        'mean_constant': 0.0, #1.052471,
        'num_epochs': 50,
        'privacy_idx': privacy_idx,
        'num_eigen_tol': 1e4,
        'seed': seed,#5122023,
        'noiseless': False,
        'fix_lengthscale': False,
        'use_analytical_mle': False, # Analytical MLE parameter
        'normalize': True, # Normalization flag and (later) normalization parameters
    }
    
    # Run models for each state component
    for component in range(min(4, data['states'].shape[1])):
        # Prepare component data
        train_x = torch.tensor(data['time'][::sampling_interval], dtype=torch.float32, device=device).view(-1, 1)
        train_y = torch.tensor(data['states'][:, component][::sampling_interval], dtype=torch.float32, device=device)

        # Get ground truth test values for this component
        if test_y is None:
            # Use the full trajectory data as ground truth
            test_y_comp = torch.tensor(data['states'][:, component], dtype=torch.float32, device=device)
            # Interpolate to match test_x if needed
            if len(test_y_comp) != test_x.shape[0]:
                # Simple linear interpolation
                from scipy.interpolate import interp1d
                interp_func = interp1d(data['time'], data['states'][:, component], kind='linear')
                test_y_comp = torch.tensor(interp_func(test_x.squeeze().cpu().numpy()), device=device)
        else:
            test_y_comp = test_y[:, component].to(device)

        # Store training data once
        results['train_data'][component] = {
            'x': train_x.detach().cpu(),
            'y': train_y.detach().cpu()
        }
        # Store ground truth test data
        results['test_data'][component] = {
            'x': test_x.detach().cpu(),
            'y': test_y_comp.detach().cpu()
        }
        

        # Run each privacy algorithm with their respective parameters
        for alg in algs:
            if alg == 'privacy-aware':
                for alpha in alphas:
                    print(f"Running privacy-aware GP for component {component} with alpha={alpha}")
                    
                    # Create and run privacy-aware GP model
                    start_time = time.time()

                    model = PrivacyGP(train_x=train_x, train_y=train_y, alpha=alpha, normalize=False, **gp_params)
                    preds = model(test_x)
                    if not gp_params['noiseless']:
                        preds = model.gpmodel.likelihood(preds)
                    
                    # Store results
                    metrics['time'][alg][alpha][component] = time.time() - start_time
                    pred_mean = model.privacy_mean.detach().cpu()
                    pred_var = model.privacy_cov.diag().detach().cpu().clamp(min=0)
                    metrics['rmse'][alg][alpha][component] = ((pred_mean - test_y_comp)**2).mean().sqrt().numpy()
                    results['preds'][alg][alpha][component] = {
                        'mean': pred_mean,
                        'variance': pred_var,
                    }
                    
                    results['obfuscated_data'][alpha][component] = {
                        'x': model.origin_obfuscated_x.detach().cpu(),
                        'y': model.origin_obfuscated_y.detach().cpu()
                    }

                    if 'attack' in algs:
                        print(f"Running Attack for component {component} with alpha={alpha}")

                        # Create and run Attack model
                        start_time = time.time()
                        model = NoisyGP(
                            train_x=results['obfuscated_data'][alpha][component]['x'],
                            train_y=results['obfuscated_data'][alpha][component]['y'],
                            **attack_params,
                            )
                        mean, cov = model.test(test_x=test_x)
                    
                        # Store results
                        metrics['time']['attack'][alpha][component] = time.time() - start_time
                        pred_mean = mean.detach().cpu()
                        pred_var = cov.diag().detach().cpu().clamp(min=0)
                        metrics['rmse']['attack'][alpha][component] = ((pred_mean - test_y_comp)**2).mean().sqrt().numpy()
                        results['preds']['attack'][alpha][component] = {
                            'mean': pred_mean,
                            'variance': pred_var,
                        }
            
            elif alg in ['dp-cloaking', 'dp-standard']:
                for eps in epsilons:
                    print(f"Running {alg} for component {component} with epsilon={eps}")
                    
                    # Configure method and epsilon
                    start_time = time.time()
                    method = 'cloaking' if alg == 'dp-cloaking' else 'standard'
                    model = DPGP(
                        train_x=train_x, train_y=train_y,
                        epsilon=eps, method=method,
                        **gp_params, **dp_params, 
                    )
                    preds = model(test_x)
                    if gp_params['noiseless'] is False:
                        preds = model.gpmodel.likelihood(preds)
                    
                    # Store results
                    metrics['time'][alg][eps][component] = time.time() - start_time
                    pred_mean = model.privacy_mean.detach().cpu()
                    pred_var = model.privacy_cov.diag().detach().cpu().clamp(min=0)
                    metrics['rmse'][alg][eps][component] = ((pred_mean - test_y_comp)**2).mean().sqrt().numpy()
                    results['preds'][alg][eps][component] = {
                        'mean': pred_mean,
                        'variance': pred_var,
                    }
            
            elif alg == 'exclude':
                print(f"Running ExcludeGP for component {component}")
                
                # Create and run ExcludingGP model
                start_time = time.time()
                model = ExcludeGP(train_x=train_x, train_y=train_y, normalize=True, **gp_params)
                mean, cov = model.test(test_x=test_x)

                # Store results
                metrics['time'][alg]['none'][component] = time.time() - start_time
                pred_mean = mean.detach().cpu()
                pred_var = cov.diag().detach().cpu().clamp(min=0)
                metrics['rmse'][alg]['none'][component] = ((pred_mean - test_y_comp)**2).mean().sqrt().numpy()
                results['preds'][alg]['none'][component] = {
                    'mean': pred_mean,
                    'variance': pred_var,
                }
                    
    return results, metrics

def run_multiple_seeds(
        test_x=None,
        seeds=range(5),  # Run with 5 different seeds by default
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5, 0.9],
        epsilons=[0.5, 1],
        sampling_interval=5,
        verbose=True,
        run_again=False,
    ):
    """Run experiments with multiple seeds and collect aggregate metrics."""
    results_path = os.path.join(results_dir, 'zeromean_satellite_all_results.pkl')
    metrics_path = os.path.join(results_dir, 'zeromean_satellite_all_metrics.pkl')
    
    # Check if results already exist and we can skip computation
    if os.path.exists(results_path) and os.path.exists(metrics_path) and not run_again:
        print(f"Loading existing results from {results_path} and {metrics_path}")
        with open(metrics_path, 'rb') as f:
            all_metrics = pickle.load(f)
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
            
        # Compute aggregate metrics from existing data
        aggregate_metrics = compute_aggregate_metrics(all_metrics, algs, alphas, epsilons)
        return aggregate_metrics, all_results
    
    
    # Get data once
    data = get_satellite_data()
    
    # Get test data
    if test_x is None:
        test_x = torch.linspace(0, 3, 301).view(-1, 1)[::sampling_interval]
    
    # Initialize collections
    all_metrics = []
    all_results = []
    
    # Run for each seed
    for seed in seeds:
        if verbose:
            print(f"\n=== Running with seed {seed} ===")
        
        # Get privacy indices for this seed
        privacy_idx = get_privacy_indices(
            data_time=data['time'], 
            sampling_interval=sampling_interval,
        )
        
        # Run experiment and collect metrics
        results, metrics = run_with_metrics(
            seed=seed,
            privacy_idx=privacy_idx,
            test_x=test_x,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons,
            sampling_interval=sampling_interval,
            verbose=verbose,
        )
        
        all_metrics.append(metrics)
        all_results.append(results)
        
    # Save all metrics and all results
    with open(metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"Metrics saved to: {metrics_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to: {results_path}")
    
    # Calculate aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(all_metrics, algs, alphas, epsilons)
    
    # Save aggregate metrics
    aggregate_path = os.path.join(results_dir, 'zeromean_satellite_aggregate_metrics.pkl')
    with open(aggregate_path, 'wb') as f:
        pickle.dump(aggregate_metrics, f)
    print(f"\nAggregate metrics saved to: {aggregate_path}")
    
    return aggregate_metrics


def compute_aggregate_metrics(all_metrics, algs, alphas, epsilons):
    """
    Compute aggregate metrics (mean and std) across all seeds.
    
    :param all_metrics: List of metrics dictionaries from different seeds
    :param algs: List of algorithms
    :param alphas: List of alpha values
    :param epsilons: List of epsilon values
    :return: Dictionary with aggregate metrics
    """
    # Initialize aggregate metrics structure
    aggregate = {
        'mean_time': {alg: {} for alg in algs},
        'std_time': {alg: {} for alg in algs},
        'mean_rmse': {alg: {} for alg in algs},
        'std_rmse': {alg: {} for alg in algs},
    }
    
    # Initialize parameters for each algorithm type
    for alg in algs:
        params = []
        if alg in ['privacy-aware', 'attack']:
            params = alphas
        elif alg in ['dp-cloaking', 'dp-standard']:
            params = epsilons
        elif alg == 'exclude':
            params = ['none']
        
        for metric_type in ['mean_time', 'std_time', 'mean_rmse', 'std_rmse']:
            for param in params:
                aggregate[metric_type][alg][param] = {}
    
    # Compute means and standard deviations
    for alg in algs:
        # Determine parameters based on algorithm type
        if alg in ['privacy-aware', 'attack']:
            params = alphas
        elif alg in ['dp-cloaking', 'dp-standard']:
            params = epsilons
        elif alg == 'exclude':
            params = ['none']
        
        for param in params:
            # Collect all components
            components = set()
            for metrics in all_metrics:
                if alg in metrics['time'] and param in metrics['time'][alg]:
                    components.update(metrics['time'][alg][param].keys())
            
            for component in components:
                # Collect metrics across seeds
                times = []
                rmses = []
                
                for metrics in all_metrics:
                    if (alg in metrics['time'] and param in metrics['time'][alg] and
                            component in metrics['time'][alg][param]):
                        times.append(metrics['time'][alg][param][component])
                        rmses.append(metrics['rmse'][alg][param][component])
                
                # Only compute if we have data
                if times and rmses:
                    # Compute mean and std
                    aggregate['mean_time'][alg][param][component] = np.mean(times)
                    aggregate['std_time'][alg][param][component] = np.std(times)
                    aggregate['mean_rmse'][alg][param][component] = np.mean(rmses)
                    aggregate['std_rmse'][alg][param][component] = np.std(rmses)
    
    return aggregate


def save_metrics_to_csv(aggregate_metrics, algs, alphas, epsilons, output_file='zeromean_satellite_metrics.csv'):
    """
    Save the aggregated metrics to a CSV file with mean ± std format.
    Includes averages across all seeds and components for each Algorithm-Parameter combination.
    
    :param aggregate_metrics: Dictionary containing aggregated metrics
    :param algs: List of algorithms used
    :param alphas: List of alpha values used
    :param epsilons: List of epsilon values used
    :param output_file: Path to save the CSV file
    :return: Path to the saved CSV file
    """
    # Prepare CSV headers and rows
    headers = ['Algorithm', 'Parameter', 'Component', 'Time (s)', 'RMSE']
    rows = []
    
    # Dictionaries to track values for calculating averages and std devs
    overall_time_values = {}  # (alg, param) -> list of all mean times
    overall_rmse_values = {}  # (alg, param) -> list of all mean rmses
    
    # Collect data for each algorithm and parameter
    for alg in algs:
        # Determine parameter type based on algorithm
        if alg in ['privacy-aware', 'attack']:
            param_values = alphas
            param_name = 'alpha'
        elif alg in ['dp-cloaking', 'dp-standard']:
            param_values = epsilons
            param_name = 'epsilon'
        elif alg == 'exclude':
            param_values = ['none']
            param_name = ''
        
        for param in param_values:
            # Initialize tracking for this algorithm-parameter combination
            key = (alg, param)
            overall_time_values[key] = []  # list of all mean times
            overall_rmse_values[key] = []  # list of all mean rmses
            
            # Get components for this algorithm-parameter combination
            components = []
            
            if (alg in aggregate_metrics['mean_time'] and 
                param in aggregate_metrics['mean_time'][alg]):
                components = aggregate_metrics['mean_time'][alg][param].keys()
            
            for component in components:
                # Extract metrics
                mean_time = aggregate_metrics['mean_time'][alg][param].get(component, float('nan'))
                std_time = aggregate_metrics['std_time'][alg][param].get(component, float('nan'))
                mean_rmse = aggregate_metrics['mean_rmse'][alg][param].get(component, float('nan'))
                std_rmse = aggregate_metrics['std_rmse'][alg][param].get(component, float('nan'))
                
                # Format as mean ± std
                time_formatted = f"{mean_time:.4f} ± {std_time:.4f}"
                rmse_formatted = f"{mean_rmse:.6f} ± {std_rmse:.6f}"
                
                # Add row
                param_display = f"{param_name}={param}" if param_name else "N/A"
                rows.append([
                    alg,
                    param_display,
                    f"Component {component}",
                    time_formatted,
                    rmse_formatted
                ])
                
                # Add values to lists for calculating averages and std devs
                if not math.isnan(mean_time):
                    overall_time_values[key].append(mean_time)
                
                if not math.isnan(mean_rmse):
                    overall_rmse_values[key].append(mean_rmse)
    
    # Add a separator row
    rows.append(["", "", "", "", ""])
    rows.append(["=== AVERAGES ACROSS SEEDS AND COMPONENTS ===", "", "", "", ""])
    
    # Calculate and add overall averages for each algorithm-parameter combination
    for alg in algs:
        # Determine parameter type based on algorithm
        if alg in ['privacy-aware', 'attack']:
            param_values = alphas
            param_name = 'alpha'
        elif alg in ['dp-cloaking', 'dp-standard']:
            param_values = epsilons
            param_name = 'epsilon'
        elif alg == 'exclude':
            param_values = ['none']
            param_name = ''
        
        for param in param_values:
            key = (alg, param)
            
            # Calculate averages and standard deviations
            time_values = np.array(overall_time_values[key])
            rmse_values = np.array(overall_rmse_values[key])
            
            if len(time_values) > 0:
                avg_time = np.mean(time_values)
                std_time = np.std(time_values)
                time_formatted = f"{avg_time:.4f} ± {std_time:.4f}"
            else:
                time_formatted = "N/A"
                
            if len(rmse_values) > 0:
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                rmse_formatted = f"{avg_rmse:.6f} ± {std_rmse:.6f}"
            else:
                rmse_formatted = "N/A"
            
            # Add row
            param_display = f"{param_name}={param}" if param_name else "N/A"
            rows.append([
                alg,
                param_display,
                "AVERAGE",
                time_formatted,
                rmse_formatted
            ])
    
    # Write to CSV
    csv_path = os.path.join(results_dir, output_file)
    try:
        # Check if directory exists
        if not os.path.exists(results_dir):
            print(f"Creating directory: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
            
        # Write the CSV file
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
            
        print(f"Metrics summary saved to: {csv_path}")
        
        # Verify the file was written
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            print(f"File verification: Created file of size {file_size} bytes")
        else:
            print("Warning: File was not created successfully after write operation")
            
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        
    return csv_path


def plot_privacy_results(
        test_x=None,
        seed=0,
        algs=['privacy-aware', 'dp-cloaking'],
        alphas=[0.1, 0.5],
        epsilons=[0.5, 1], 
        confidence_level=0.95,
        run_again=False,
    ):
    """Create plots showing privacy-aware GP results."""
    # Load results or run experiments if needed
    results_path = os.path.join(results_dir, 'zeromean_satellite_all_results.pkl')
    
    if not os.path.exists(results_path) or run_again:
        print("Results not found or run_again=True. Running experiments...")
        _, all_results = run_multiple_seeds(
            test_x=test_x if test_x is not None else torch.linspace(0, 3, 301).view(-1, 1)[::5],
            seeds=[seed],
            algs=algs, 
            alphas=alphas, 
            epsilons=epsilons,
            run_again=True
        )
    else:
        # Load saved results
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
    
    # Get results for the specified seed
    seed_results = None
    for result in all_results:
        if result['seed'] == seed:
            seed_results = result
            break
    
    if seed_results is None:
        print(f"No results found for seed {seed}. Using first available result.")
        seed_results = all_results[0]
    
    # Extract data
    test_x_np = seed_results['test_x'].numpy().flatten()
    component_names = [r'$r/R_e$', r'$\dot{r}/\bar{v}$', r'$\theta$ (rad)', r'$\dot{\theta}$ (rad/s)']
    n_components = len(seed_results['train_data'])
    
    # Convert confidence level to z-score using PyTorch
    z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + confidence_level / 2)).detach().cpu().item()
    

    # Define visual settings
    viz_settings = {
        'privacy-aware': {'color': 'green', 'style': '-', 'label': 'Privacy-aware GP', 'alpha': 0.2},
        'dp-cloaking': {'color': 'brown', 'style': '--', 'label': 'DP-Cloaking GP', 'alpha': 0.2},
        'dp-standard': {'color': 'yellow', 'style': '-.', 'label': 'Standard DP GP', 'alpha': 0.2},
        'exclude': {'color': 'darkorange', 'style': '--', 'label': 'GP for non-private data (Dropout)', 'alpha': 0.1},
        'attack': {'color': 'darkviolet', 'style': '--', 'label': 'Noisy GP for obfuscated data (Attack)', 'alpha': 0.1},
    }
    
    # DP epsilon colors
    eps_colors = ['green', 'royalblue', 'darkviolet', 'blue', 'darkviolet', 'brown']
    
    # Helper function to setup subplot with base data
    def setup_subplot(ax, comp_idx):
        train_data = seed_results['train_data'][comp_idx]
        train_x_np = train_data['x'].numpy().flatten()
        train_y_np = train_data['y'].numpy()
        
        # Identify private segment
        private_mask = (train_x_np >= 1) & (train_x_np < 2)
        
        # Plot true trajectory and private segment
        ax.plot(train_x_np, train_y_np, 'k', linewidth=1.5, label='True')
        ax.plot(train_x_np[private_mask], train_y_np[private_mask], 'r', linewidth=1.5, 
               label='Private segment')
        
        # Set title and labels
        ax.set_title(component_names[comp_idx], fontsize=18)
        ax.set_xlabel('Time $T/T_{orb}$', fontsize=18)
        ax.grid(True, alpha=0.3)
        
        return train_x_np, train_y_np, private_mask
    
    # Helper function to add predictions to subplot
    def add_predictions(ax, alg, param_value, comp_idx, custom_label=None, custom_color=None):
        if alg in seed_results['preds'] and param_value in seed_results['preds'][alg] and comp_idx in seed_results['preds'][alg][param_value]:
            preds = seed_results['preds'][alg][param_value][comp_idx]
            
            if preds:
                mean_np = preds['mean'].numpy()
                var_np = preds['variance'].numpy()
                
                # Get visualization settings
                settings = viz_settings[alg]
                color = custom_color if custom_color else settings['color']
                label = custom_label if custom_label else settings['label']
                
                # Plot mean line
                ax.plot(
                    test_x_np, mean_np, 
                    color=color, 
                    linestyle=settings['style'], 
                    linewidth=1.8, 
                    label=label,
                )
                
                # Plot confidence interval
                ax.fill_between(
                    test_x_np, 
                    mean_np - z_score*np.sqrt(var_np), 
                    mean_np + z_score*np.sqrt(var_np),
                    color=color, 
                    alpha=settings['alpha'],
                )
    
    ############################################################
    # PLOT TYPE 1: Privacy-aware figures for each alpha level
    ############################################################
    for alpha in alphas:
        if 'privacy-aware' not in algs:
            continue
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))
        if n_components == 1:
            axes = [axes]  # Handle case with single component
        
        # Loop through components
        for comp_idx in range(n_components):
            ax = axes[comp_idx]
            
            # Setup the subplot with base data
            setup_subplot(ax, comp_idx)
            
            # Plot obfuscated data if available
            if 'obfuscated_data' in seed_results and alpha in seed_results['obfuscated_data'] and comp_idx in seed_results['obfuscated_data'][alpha]:
                obf_data = seed_results['obfuscated_data'][alpha][comp_idx]
                obf_x_np = obf_data['x'].numpy()
                obf_y_np = obf_data['y'].numpy()
                ax.scatter(obf_x_np, obf_y_np, c='b', s=15, label='Privacy-aware GP obfuscated data (Ours)')
            
            # Plot privacy-related algorithms
            privacy_related_algs = ['privacy-aware', 'exclude', 'attack']
            for alg in [a for a in privacy_related_algs if a in algs]:
                # Get parameter value based on algorithm type
                if alg == 'privacy-aware':
                    param_value = alpha
                    custom_label = 'Privacy-aware GP reconstructed prediction (Ours)'
                elif alg == 'exclude':
                    param_value = 'none'
                    custom_label = None
                elif alg == 'attack':
                    param_value = alpha
                    custom_label = None
                
                add_predictions(ax, alg, param_value, comp_idx, custom_label)
        
        # Create legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                   ncol=min(3, len(handles)), fontsize=21)
        
        # Set title
        fig.suptitle(f'Privacy-aware GP with zero mean (H={alpha}K)', color='blue', fontsize=24, fontweight='bold', y=1.05)
        
        # Save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.33)
        fig_path = os.path.join(figs_dir, f'zeromean_satellite_pa_alpha_{alpha}_seed_{seed}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Privacy-aware comparison figure for α={alpha} seed={seed} saved to: {fig_path}")
        plt.close(fig)

    ############################################################
    # PLOT TYPE 2: DP methods figures
    ############################################################
    dp_algs = [alg for alg in ['dp-cloaking', 'dp-standard'] if alg in algs]
    
    if dp_algs in algs and 'exclude' in algs:  # Add 'exclude' check here
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))
        if n_components == 1:
            axes = [axes]
        
        # Loop through components
        for comp_idx in range(n_components):
            ax = axes[comp_idx]
            
            # Setup the subplot with base data
            setup_subplot(ax, comp_idx)
            
            # Plot DP algorithms with different epsilon values
            for alg in dp_algs:
                for i, eps in enumerate(epsilons):
                    # Create label with epsilon and delta values
                    custom_label = f'($\epsilon$={eps}, $\delta$=0.01)-{viz_settings[alg]["label"]}'
                    custom_color = eps_colors[i % len(eps_colors)]
                    
                    # Add predictions
                    add_predictions(ax, alg, eps, comp_idx, custom_label, custom_color)
            
            # Add 'exclude' algorithm if it's in the requested algorithms
            if 'exclude' in algs:
                add_predictions(ax, 'exclude', 'none', comp_idx)
        
        # Create legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                   ncol=min(3, len(handles)), fontsize=21)
        
        # Set title
        fig.suptitle('Differential Privacy with GP', color='blue', fontsize=24, fontweight='bold', y=1.05)
        
        # Save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.33)
        fig_path = os.path.join(figs_dir, f'zeromean_satellite_dp_all_eps_seed_{seed}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"DP comparison figure with all epsilon values for seed={seed} saved to: {fig_path}")
        plt.close(fig)


    ############################################################
    # PLOT TYPE 2.1: DP methods figures with zoomed region
    ############################################################
    dp_algs = [alg for alg in ['dp-cloaking', 'dp-standard'] if alg in algs]

    if dp_algs in algs and 'exclude' in algs:  # Add 'exclude' check here
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))
        if n_components == 1:
            axes = [axes]
        
        # Loop through components
        for comp_idx in range(n_components):
            ax = axes[comp_idx]
            
            # Setup the subplot with base data
            train_x_np, train_y_np, private_mask = setup_subplot(ax, comp_idx)
            
            # Plot DP algorithms with different epsilon values
            for alg in dp_algs:
                for i, eps in enumerate(epsilons):
                    # Create label with epsilon and delta values
                    custom_label = f'($\epsilon$={eps}, $\delta$=0.01)-{viz_settings[alg]["label"]}'
                    custom_color = eps_colors[i % len(eps_colors)]
                    
                    # Add predictions
                    add_predictions(ax, alg, eps, comp_idx, custom_label, custom_color)
            
            # Add 'exclude' algorithm if it's in the requested algorithms
            if 'exclude' in algs:
                add_predictions(ax, 'exclude', 'none', comp_idx)
            
            # Create inset axes for zoomed region (only for the first component)
            if comp_idx in [0, 1, 2, 3]:  # You can change this to zoom into a different component
                # Define the region to zoom in (covering the private segment more completely)
                zoom_start, zoom_end = 0.9, 2.1  # Modified zoom range as requested
                zoom_mask = (test_x_np >= zoom_start) & (test_x_np <= zoom_end)
                
                if any(zoom_mask):
                    # Create inset axes BELOW the original subplot with increased height
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    # Place the zoom box below the subplot and make it taller
                    axins = inset_axes(ax, width="100%", height="150%", 
                                    bbox_to_anchor=(0, -2.0, 1, 1), 
                                    bbox_transform=ax.transAxes,
                                    loc="lower center")
                    
                    # Plot data in inset axes
                    axins.plot(train_x_np, train_y_np, 'k', linewidth=1.0, label='True')
                    
                    # Add private segment to inset
                    inset_private_mask = (train_x_np >= zoom_start) & (train_x_np <= zoom_end) & private_mask
                    if any(inset_private_mask):
                        axins.plot(train_x_np[inset_private_mask], train_y_np[inset_private_mask], 
                                'r', linewidth=1.0)
                    
                    # Add predictions to inset
                    for alg in dp_algs:
                        for i, eps in enumerate(epsilons):
                            if alg in seed_results['preds'] and eps in seed_results['preds'][alg] and comp_idx in seed_results['preds'][alg][eps]:
                                preds = seed_results['preds'][alg][eps][comp_idx]
                                if preds:
                                    mean_np = preds['mean'].numpy()
                                    var_np = preds['variance'].numpy()
                                    custom_color = eps_colors[i % len(eps_colors)]
                                    
                                    # Plot mean line
                                    axins.plot(
                                        test_x_np[zoom_mask], mean_np[zoom_mask], 
                                        color=custom_color, 
                                        linestyle=viz_settings[alg]['style'], 
                                        linewidth=1.5
                                    )
                                    
                                    # Add confidence interval shading as requested
                                    axins.fill_between(
                                        test_x_np[zoom_mask], 
                                        mean_np[zoom_mask] - z_score*np.sqrt(var_np[zoom_mask]), 
                                        mean_np[zoom_mask] + z_score*np.sqrt(var_np[zoom_mask]),
                                        color=custom_color, 
                                        alpha=viz_settings[alg]['alpha']
                                    )
                    
                    # If exclude algorithm is available, add it to inset
                    if 'exclude' in algs and 'none' in seed_results['preds']['exclude'] and comp_idx in seed_results['preds']['exclude']['none']:
                        exclude_preds = seed_results['preds']['exclude']['none'][comp_idx]
                        if exclude_preds:
                            exclude_mean = exclude_preds['mean'].numpy()
                            exclude_var = exclude_preds['variance'].numpy()
                            
                            # Plot mean line
                            axins.plot(
                                test_x_np[zoom_mask], exclude_mean[zoom_mask],
                                color=viz_settings['exclude']['color'],
                                linestyle=viz_settings['exclude']['style'],
                                linewidth=1.5
                            )
                            
                            # Add confidence interval shading
                            axins.fill_between(
                                test_x_np[zoom_mask], 
                                exclude_mean[zoom_mask] - z_score*np.sqrt(exclude_var[zoom_mask]), 
                                exclude_mean[zoom_mask] + z_score*np.sqrt(exclude_var[zoom_mask]),
                                color=viz_settings['exclude']['color'],
                                alpha=viz_settings['exclude']['alpha']
                            )
                    
                    # Set the inset axes limits
                    axins.set_xlim(zoom_start, zoom_end)
                    
                    # Automatically determine y-limits with some padding
                    y_data = []
                    y_data.extend(train_y_np[(train_x_np >= zoom_start) & (train_x_np <= zoom_end)])
                    
                    for alg in dp_algs:
                        for eps in epsilons:
                            if alg in seed_results['preds'] and eps in seed_results['preds'][alg] and comp_idx in seed_results['preds'][alg][eps]:
                                preds = seed_results['preds'][alg][eps][comp_idx]
                                if preds:
                                    mean_np = preds['mean'].numpy()
                                    y_data.extend(mean_np[zoom_mask])
                    
                    if y_data:
                        y_min, y_max = min(y_data), max(y_data)
                        y_range = y_max - y_min
                        y_padding = 0.1 * y_range  # 10% padding
                        axins.set_ylim(y_min - y_padding, y_max + y_padding)
                    
                    # Add a title to the inset axes
                    axins.set_title(f"Zoomed Region $T/T_{{orb}} \in $[{zoom_start}, {zoom_end}]", fontsize=12)
                    
                    # Add a light box in the main figure showing what's being zoomed
                    # Use light gray instead of blue
                    ax.axvspan(zoom_start, zoom_end, alpha=0.3, color='lightgray')
                    
                    # Add an annotation in the main plot indicating the zoom area
                    ax.annotate('Zoomed\nRegion', 
                            xy=((zoom_start + zoom_end)/2, ax.get_ylim()[0]),
                            xytext=((zoom_start + zoom_end)/2, ax.get_ylim()[0]), 
                            ha='center', va='bottom',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="whitesmoke", ec="darkgray", alpha=0.7))
        
        # Create legend with lower position to accommodate the taller zoom box
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.45), 
                ncol=min(3, len(handles)), fontsize=22)
        
        # Set title
        fig.suptitle('Differential Privacy with GP', color='blue', fontsize=24, fontweight='bold', y=1.05)
        
        # Save figure - adjust the bottom margin significantly to make room for zoomed insets and legend
        # Increased bottom margin even more
        plt.subplots_adjust(left=0.1, right=0.9, top=0.90, bottom=0.55, wspace=0.25, hspace=0.4)
        fig_path = os.path.join(figs_dir, f'zeromean_satellite_dp_all_eps_seed_{seed}_with_zoom.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"DP comparison figure with zoom for seed={seed} saved to: {fig_path}")
        plt.close(fig)
        
    return seed_results


def run(
        test_x=None,
        seeds=range(5),
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5, 0.9],
        epsilons=[0.5, 1],
        sampling_interval=5,
        verbose=True,
        plot_seed=0,  # Seed to use for plotting
        run_again=False, # If True, run experiments again even if saved results exist
    ):
    """
    Run the full experimental pipeline: generate data, run models with multiple seeds,
    compute metrics, save to CSV, and generate plots.
    """
    # Paths for results
    results_path = os.path.join(results_dir, 'zeromean_satellite_all_results.pkl')
    metrics_path = os.path.join(results_dir, 'zeromean_satellite_all_metrics.pkl')
    
    # Run experiments or load existing results
    if not os.path.exists(results_path) or not os.path.exists(metrics_path) or run_again:
        aggregate_metrics = run_multiple_seeds(
            test_x=test_x,
            seeds=seeds,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons,
            sampling_interval=sampling_interval,
            verbose=verbose,
            run_again=run_again,
        )
        
        # Save metrics summary to CSV
        csv_path = save_metrics_to_csv(
            aggregate_metrics,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons
        )
    else:
        # Load existing aggregate metrics
        with open(metrics_path, 'rb') as f:
            aggregate_metrics = pickle.load(f)

        csv_path = os.path.join(results_dir, 'zeromean_satellite_metrics.csv')
    
    # Plot results for the specified seed
    seed_results = plot_privacy_results(
        seed=plot_seed,
        algs=algs,
        alphas=alphas,
        epsilons=epsilons,
        run_again=False  # We already checked run_again above
    )
    
    return aggregate_metrics, csv_path, seed_results


if __name__ == "__main__":
    # Set parameters
    seeds = range(1)  # Run with 20 seeds: 0, 1, 2, 3, 4, ..., 19
    sampling_interval = 5
    test_x = torch.linspace(0, 3, 301).view(-1, 1)[::sampling_interval]
    algs = ['privacy-aware', 'exclude', 'attack'] #['privacy-aware', 'dp-cloaking', 'exclude', 'attack']
    alphas = [0.1, 0.5, 0.9]
    epsilons = [0.3, 0.5, 1]
    verbose = True
    plot_seed = seeds[0]  # Seed to use for plotting
    run_again = False # Set to True to regenerate all results
   
    
    # Run the full experimental pipeline
    aggregate_metrics, csv_path, seed_results = run(
        test_x=test_x,
        seeds=seeds,
        algs=algs,
        alphas=alphas,
        epsilons=epsilons,
        sampling_interval=sampling_interval,
        verbose=verbose,
        plot_seed=plot_seed,
        run_again=run_again,
    )
    
    print(f"Experiments completed. Results saved to: {csv_path}")