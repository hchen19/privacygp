import csv
import time
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from tqdm import tqdm

# Import privacy methods
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


def load_census_data():
    """
    Load the preprocessed census data from the data directory.
    
    :return: Dictionary containing the train and test data
    :rtype: dict
    """
    data_path = os.path.join(data_dir, 'census_data.pt')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Census data file not found at {data_path}. Please run dataset.py first.")
    
    data_dict = torch.load(data_path, weights_only=True)
    print(f"Loaded census data with {len(data_dict['train_inputs'])} training samples "
          f"and {len(data_dict['test_inputs'])} test samples")
    
    # Extract feature information
    if 'feature_info' in data_dict:
        feature_info = data_dict['feature_info']
    else:
        # If feature_info not available in the data file, create a basic one
        feature_info = {
            'feature_names': ['PUMA', 'POWPUMA', 'AGEP', 'ANC1P', 'ANC2P', 'YOEP', 'POBP', 'WKHP', 'WKWN'],
            'feature_means': data_dict['train_inputs'].mean(dim=0).tolist(),
            'feature_stds': data_dict['train_inputs'].std(dim=0).tolist()
        }
        data_dict['feature_info'] = feature_info
    
    return data_dict


def get_privacy_indices(data, feature_info, privacy_criteria, seed=42):
    """
    Generate indices for private data points based on specified criteria.
    
    :param data: Feature tensor (can be training or test data)
    :param feature_info: Dictionary with feature names and their indices
    :param privacy_criteria: Dictionary with criteria for private data selection
                            (e.g., {'AGEP': (65, 75), 'PUMA': (1000, 2000)})
    :param seed: Random seed for reproducibility when fallback to random selection
    :return: Indices of private samples
    :rtype: torch.Tensor
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = data.shape[0]
    
    # Check if we have all needed inputs
    if data is not None and feature_info is not None and privacy_criteria is not None:
        # Get feature names
        feature_names = feature_info.get('feature_names', [])
        
        # Convert to numpy for easier processing
        data_np = data.detach().cpu().numpy()
        
        # Start with all records as non-private
        mask = np.ones(num_samples, dtype=bool)
        criteria_applied = False
        
        # Apply each criterion to narrow down the private data
        for feature_name, value_range in privacy_criteria.items():
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                min_val, max_val = value_range
                
                # Update mask to include only records that match this criterion
                feature_mask = (data_np[:, feature_idx] >= min_val) & (data_np[:, feature_idx] <= max_val)
                mask = mask & feature_mask
                criteria_applied = True
                
                print(f"Applied {feature_name} criterion ({min_val}-{max_val}): {np.sum(feature_mask)} records match")
            else:
                print(f"Warning: Feature '{feature_name}' not found in feature_names")
        
        # Get indices where mask is True
        private_indices = np.where(mask)[0]
        
        # If no matches found or no criteria applied, fall back to random selection
        if len(private_indices) == 0 or not criteria_applied:
            print("Warning: No data points match privacy criteria. Using random selection.")
            np.random.seed(seed)
            private_indices = np.random.choice(num_samples, max(1, int(num_samples * 0.3)), replace=False)
    else:
        # Random selection fallback
        print("Missing required inputs. Using random selection for privacy indices.")
        np.random.seed(seed)
        private_indices = np.random.choice(num_samples, max(1, int(num_samples * 0.3)), replace=False)
    
    # Convert to tensor
    privacy_idx = torch.tensor(private_indices, device=device)
    
    print(f"Using {len(privacy_idx)} privacy indices ({len(privacy_idx)/num_samples*100:.1f}% of data)")
    
    return privacy_idx


def run_with_metrics(
        seed,
        train_x,
        train_y,
        test_x,
        test_y,
        privacy_idx,
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5], 
        epsilons=[0.5, 1],
        verbose=True,
    ):
    """
    Run privacy GP models for all algorithms and parameters.
    
    :param seed: Random seed for reproducibility
    :param train_x: Training features
    :param train_y: Training targets
    :param test_x: Test features
    :param test_y: Test targets
    :param privacy_idx: Indices of private data points
    :param feature_dim: Dimensionality of features
    :param algs: List of algorithms to run
    :param alphas: List of alpha values for privacy-aware GP
    :param epsilons: List of epsilon values for DP methods
    :param verbose: Whether to print verbose output
    :return: Tuple containing results and metrics dictionaries
    :rtype: tuple
    """
    # Set device - use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    # Initialize metrics storage
    metrics = {
        'seed': seed,
        'time': {},
        'rmse': {},
        'privacy_loss': {},
    }

    # Initialize results storage
    results = {
        'seed': seed,
        'test_x': test_x.detach().cpu(),
        'test_y': test_y.detach().cpu(),
        'train_data': {
            'x': train_x.detach().cpu(),
            'y': train_y.detach().cpu()
        },
        'preds': {},
        'privacy_idx': privacy_idx.cpu(),
    }

    # Initialize result structure for each algorithm
    if 'privacy-aware' in algs:
        results['preds']['privacy-aware'] = {alpha: {} for alpha in alphas}
        results['obfuscated_data'] = {alpha: {} for alpha in alphas}
        metrics['time']['privacy-aware'] = {alpha: {} for alpha in alphas}
        metrics['rmse']['privacy-aware'] = {alpha: {} for alpha in alphas}
        metrics['privacy_loss']['privacy-aware'] = {alpha: {} for alpha in alphas}
    
    if 'dp-cloaking' in algs:
        results['preds']['dp-cloaking'] = {eps: {} for eps in epsilons}
        metrics['time']['dp-cloaking'] = {eps: {} for eps in epsilons}
        metrics['rmse']['dp-cloaking'] = {eps: {} for eps in epsilons}
        metrics['privacy_loss']['dp-cloaking'] = {eps: {} for eps in epsilons}
    
    if 'dp-standard' in algs:
        results['preds']['dp-standard'] = {eps: {} for eps in epsilons}
        metrics['time']['dp-standard'] = {eps: {} for eps in epsilons}
        metrics['rmse']['dp-standard'] = {eps: {} for eps in epsilons}
        metrics['privacy_loss']['dp-standard'] = {eps: {} for eps in epsilons}

    if 'exclude' in algs:
        results['preds']['exclude'] = {'none': {}}
        metrics['time']['exclude'] = {'none': {}}
        metrics['rmse']['exclude'] = {'none': {}}
        metrics['privacy_loss']['exclude'] = {'none': {}}
    
    if 'attack' in algs:
        results['preds']['attack'] = {alpha: {} for alpha in alphas}
        metrics['time']['attack'] = {alpha: {} for alpha in alphas}
        metrics['rmse']['attack'] = {alpha: {} for alpha in alphas}
        metrics['privacy_loss']['attack'] = {alpha: {} for alpha in alphas}
    
    # Common GP parameters - adjusted for census income prediction
    gp_params = {
        'train': True,
        'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(device),
        'model_name': 'ExactGP',
        'noise': 0.1,  # Adjusted for income prediction
        'lengthscale': 1.2, #1.2 for noisy gptrain, 0.85 for mle # Adjusted for multi-dimensional feature space
        'outputscale': 1.0,
        'mean_constant': 0.0,
        'num_epochs': 100,
        'privacy_idx': privacy_idx,
        'num_eigen_tol': 1e4,
        'seed': seed,
        'noiseless': False,  # Income prediction is inherently noisy
        'fix_lengthscale': False,
        'use_analytical_mle': False,
    }

    # DP parameters
    dp_params = {
        'delta': 0.01,
        'sensitivity': 1,
        'num_attempts': 7,
        'max_iterations': 300,
        'num_samples': 100,
        'optimizer': 'grad',
        'normalize': True,
    }

    # Attack parameters
    attack_params = {
        'train': True,
        'likelihood': gpytorch.likelihoods.GaussianLikelihood().to(device),
        'model_name': 'ExactGP',
        'noise': 0.1,
        'lengthscale': 1.3,
        'outputscale': 1.0,
        'mean_constant': 0.0,
        'num_epochs': 100,
        'privacy_idx': privacy_idx,
        'num_eigen_tol': 1e4,
        'seed': seed,
        'noiseless': False,
        'fix_lengthscale': False,
        'use_analytical_mle': False,
        'normalize': True,
    }
    
    # Run each privacy algorithm with their respective parameters
    for alg in algs:
        if alg == 'privacy-aware':
            for alpha in alphas:
                if verbose:
                    print(f"Running privacy-aware GP with alpha={alpha}")
                
                # Create and run privacy-aware GP model
                start_time = time.time()

                model = PrivacyGP(train_x=train_x, train_y=train_y, alpha=alpha, normalize=True, **gp_params)
                preds = model(test_x)
                if not gp_params['noiseless']:
                    preds = model.gpmodel.likelihood(preds)
                
                # Calculate privacy loss
                private_indices = privacy_idx.detach().cpu().numpy()
                privacy_loss = calculate_privacy_loss(
                    original_data=train_y.detach().cpu().numpy(),
                    obfuscated_data=model.origin_obfuscated_y.detach().cpu().numpy(),
                    private_indices=private_indices
                )
                
                # Store results
                metrics['time'][alg][alpha] = time.time() - start_time
                pred_mean = preds.mean.detach().cpu()
                pred_var = preds.variance.detach().cpu().clamp(min=0)
                metrics['rmse'][alg][alpha] = torch.sqrt(((pred_mean - test_y.cpu())**2).mean()).item()
                metrics['privacy_loss'][alg][alpha] = privacy_loss
                
                results['preds'][alg][alpha] = {
                    'mean': pred_mean,
                    'variance': pred_var,
                }
                
                results['obfuscated_data'][alpha] = {
                    'x': model.origin_obfuscated_x.detach().cpu(),
                    'y': model.origin_obfuscated_y.detach().cpu()
                }

                if 'attack' in algs:
                    if verbose:
                        print(f"Running Attack with alpha={alpha}")

                    # Create and run Attack model
                    start_time = time.time()
                    model = NoisyGP(
                        train_x=results['obfuscated_data'][alpha]['x'].to(device),
                        train_y=results['obfuscated_data'][alpha]['y'].to(device),
                        **attack_params,
                    )
                    mean, cov = model.test(test_x=test_x)
                
                    # Store results
                    metrics['time']['attack'][alpha] = time.time() - start_time
                    pred_mean = mean.detach().cpu()
                    pred_var = cov.diag().detach().cpu().clamp(min=0)
                    metrics['rmse']['attack'][alpha] = torch.sqrt(((pred_mean - test_y.cpu())**2).mean()).item()
                    
                    # Calculate attack privacy loss (how much private information the attack can recover)
                    attack_privacy_loss = calculate_attack_effectiveness(
                        original_data=train_y.detach().cpu().numpy(),
                        attack_prediction=pred_mean.detach().cpu().numpy(),
                        private_indices=private_indices
                    )
                    metrics['privacy_loss']['attack'][alpha] = attack_privacy_loss
                    
                    results['preds']['attack'][alpha] = {
                        'mean': pred_mean,
                        'variance': pred_var,
                    }
        
        elif alg in ['dp-cloaking', 'dp-standard']:
            for eps in epsilons:
                if verbose:
                    print(f"Running {alg} with epsilon={eps}")
                
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
                
                # Calculate privacy loss
                private_indices = privacy_idx.detach().cpu().numpy()
                # For DP methods, the privacy loss is measured differently
                privacy_loss = eps  # Epsilon is the formal privacy budget
                
                # Store results
                metrics['time'][alg][eps] = time.time() - start_time
                pred_mean = preds.mean.detach().cpu()
                pred_var = preds.variance.detach().cpu().clamp(min=0)
                metrics['rmse'][alg][eps] = torch.sqrt(((pred_mean - test_y.cpu())**2).mean()).item()
                metrics['privacy_loss'][alg][eps] = privacy_loss
                
                results['preds'][alg][eps] = {
                    'mean': pred_mean,
                    'variance': pred_var,
                }
        
        elif alg == 'exclude':
            if verbose:
                print(f"Running ExcludeGP")
            
            # Create and run ExcludingGP model
            start_time = time.time()
            model = ExcludeGP(train_x=train_x, train_y=train_y, normalize=True, **gp_params)
            mean, cov = model.test(test_x=test_x)

            # Store results
            metrics['time'][alg]['none'] = time.time() - start_time
            pred_mean = mean.detach().cpu()
            pred_var = cov.diag().detach().cpu().clamp(min=0)
            metrics['rmse'][alg]['none'] = torch.sqrt(((pred_mean - test_y.cpu())**2).mean()).item()
            
            # Calculate privacy loss - for exclude method, this is theoretically zero
            # since private data is completely excluded
            metrics['privacy_loss'][alg]['none'] = 0.0
            
            results['preds'][alg]['none'] = {
                'mean': pred_mean,
                'variance': pred_var,
            }
    
    return results, metrics


def calculate_privacy_loss(original_data, obfuscated_data, private_indices):
    """
    Calculate privacy loss as the normalized difference between original and obfuscated private data.
    
    :param original_data: Original data values
    :param obfuscated_data: Data after privacy transformations
    :param private_indices: Indices of private data points
    :return: Privacy loss metric
    :rtype: float
    """
    # Extract private data points
    original_private = original_data[private_indices]
    obfuscated_private = obfuscated_data[private_indices]
    
    # Calculate normalized root mean squared difference
    diff = np.abs(original_private - obfuscated_private)
    max_val = np.max(np.abs(original_private))
    
    if max_val > 0:
        normalized_diff = diff / max_val
        privacy_loss = np.sqrt(np.mean(normalized_diff**2))
    else:
        privacy_loss = 0.0
    
    return privacy_loss


def calculate_attack_effectiveness(original_data, attack_prediction, private_indices):
    """
    Calculate how effective an attack is at recovering private information.
    Lower values mean better privacy protection.
    
    :param original_data: Original training data
    :param attack_prediction: Model predictions from attack
    :param private_indices: Indices of private data points
    :return: Attack effectiveness metric (0-1, lower is better for privacy)
    :rtype: float
    """
    # Extract private data points
    original_private = original_data[private_indices]
    
    # We'll use a simple correlation measure
    correlation = np.corrcoef(original_private, attack_prediction[:len(private_indices)])[0, 1]
    
    # Convert to absolute value and normalize to [0, 1]
    effectiveness = min(1.0, abs(correlation))
    
    return effectiveness


def run_multiple_seeds(
        feature_indices=None,
        seeds=range(5),
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5, 0.9],
        epsilons=[0.5, 1],
        target_scale=1e5,
        verbose=True,
        run_again=False,
        privacy_criteria=None,
    ):
    """
    Run experiments with multiple seeds and collect aggregate metrics.
    
    :param feature_indices: Indices of features to use (None for all)
    :param seeds: Range of seeds to use
    :param algs: List of algorithms to run
    :param alphas: List of alpha values for privacy-aware GP
    :param epsilons: List of epsilon values for DP methods
    :param target_scale: Scale factor used for targets
    :param verbose: Whether to print verbose output
    :param run_again: Whether to run experiments again even if results exist
    :param privacy_criteria: Dictionary with criteria for private data selection
    :return: Tuple containing aggregate metrics and all results
    :rtype: tuple
    """
    results_path = os.path.join(results_dir, 'census_all_results.pkl')
    metrics_path = os.path.join(results_dir, 'census_all_metrics.pkl')
    
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
    
    # Load data
    data_dict = load_census_data()
    train_inputs = data_dict['train_inputs']
    train_targets = data_dict['train_targets'] / target_scale
    test_inputs = data_dict['test_inputs']
    test_targets = data_dict['test_targets'] / target_scale
    
    # Get feature information for privacy selection
    feature_info = data_dict.get('feature_info', {})
    
    # Select features if specified
    if feature_indices is not None:
        train_inputs = train_inputs[:, feature_indices]
        test_inputs = test_inputs[:, feature_indices]
        
        # Update feature_info if needed
        if feature_indices and 'feature_names' in feature_info:
            feature_info['feature_names'] = [feature_info['feature_names'][i] for i in feature_indices]
    
    feature_dim = train_inputs.shape[1]
    print(f"Using {feature_dim} features for modeling")
    
    # Initialize collections
    all_metrics = []
    all_results = []
    
    # Run for each seed
    for seed in seeds:
        if verbose:
            print(f"\n=== Running with seed {seed} ===")
        
        # Get privacy indices based on specified criteria
        privacy_idx = get_privacy_indices(
            data=train_inputs,
            feature_info=feature_info,
            privacy_criteria=privacy_criteria,
            seed=seed,
        )
        
        # Run experiment and collect metrics
        results, metrics = run_with_metrics(
            seed=seed,
            train_x=train_inputs,
            train_y=train_targets,
            test_x=test_inputs,
            test_y=test_targets,
            privacy_idx=privacy_idx,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons,
            verbose=verbose,
        )
        
        all_metrics.append(metrics)
        all_results.append(results)
        
    # Save all results
    with open(metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results and metrics saved to {results_dir}")
    
    # Calculate and save aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(all_metrics, algs, alphas, epsilons)
    with open(os.path.join(results_dir, 'census_aggregate_metrics.pkl'), 'wb') as f:
        pickle.dump(aggregate_metrics, f)
    
    return aggregate_metrics, all_results


def compute_aggregate_metrics(all_metrics, algs, alphas, epsilons):
    """
    Compute aggregate metrics (mean and std) across all seeds.
    
    :param all_metrics: List of metrics dictionaries from different seeds
    :param algs: List of algorithms
    :param alphas: List of alpha values
    :param epsilons: List of epsilon values
    :return: Aggregate metrics dictionary
    :rtype: dict
    """
    # Initialize aggregate metrics structure
    aggregate = {
        'mean_time': {alg: {} for alg in algs},
        'std_time': {alg: {} for alg in algs},
        'mean_rmse': {alg: {} for alg in algs},
        'std_rmse': {alg: {} for alg in algs},
        'mean_privacy_loss': {alg: {} for alg in algs},
        'std_privacy_loss': {alg: {} for alg in algs},
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
        
        for metric_type in ['mean_time', 'std_time', 'mean_rmse', 'std_rmse', 'mean_privacy_loss', 'std_privacy_loss']:
            for param in params:
                aggregate[metric_type][alg][param] = 0.0
    
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
            # Collect metrics across seeds
            times = []
            rmses = []
            privacy_losses = []
            
            for metrics in all_metrics:
                if alg in metrics['time'] and param in metrics['time'][alg]:
                    times.append(metrics['time'][alg][param])
                    rmses.append(metrics['rmse'][alg][param])
                    privacy_losses.append(metrics['privacy_loss'][alg][param])
            
            # Only compute if we have data
            if times and rmses and privacy_losses:
                # Compute mean and std
                aggregate['mean_time'][alg][param] = np.mean(times)
                aggregate['std_time'][alg][param] = np.std(times)
                aggregate['mean_rmse'][alg][param] = np.mean(rmses)
                aggregate['std_rmse'][alg][param] = np.std(rmses)
                aggregate['mean_privacy_loss'][alg][param] = np.mean(privacy_losses)
                aggregate['std_privacy_loss'][alg][param] = np.std(privacy_losses)
    
    return aggregate


def save_metrics_to_csv(aggregate_metrics, algs, alphas, epsilons, output_file='census_metrics.csv'):
    """
    Save the aggregated metrics to a CSV file with mean ± std format.
    
    :param aggregate_metrics: Dictionary containing aggregated metrics
    :param algs: List of algorithms used
    :param alphas: List of alpha values used
    :param epsilons: List of epsilon values used
    :param output_file: Path to save the CSV file
    :return: Path to the saved CSV file
    :rtype: str
    """
    # Prepare CSV headers and rows
    headers = ['Algorithm', 'Parameter', 'Time (s)', 'RMSE']  # Removed Privacy Loss
    rows = []
    
    # Dictionaries to track values for calculating averages
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
            overall_time_values[key] = []
            overall_rmse_values[key] = []
            
            # Extract metrics
            mean_time = aggregate_metrics['mean_time'][alg].get(param, float('nan'))
            std_time = aggregate_metrics['std_time'][alg].get(param, float('nan'))
            mean_rmse = aggregate_metrics['mean_rmse'][alg].get(param, float('nan'))
            std_rmse = aggregate_metrics['std_rmse'][alg].get(param, float('nan'))
            
            # Format as mean ± std
            time_formatted = f"{mean_time:.4f} ± {std_time:.4f}"
            rmse_formatted = f"{mean_rmse:.6f} ± {std_rmse:.6f}"
            
            # Add row
            param_display = f"{param_name}={param}" if param_name else "N/A"
            rows.append([
                alg,
                param_display,
                time_formatted,
                rmse_formatted
            ])
            
            # Add values to lists for calculating overall averages
            if not math.isnan(mean_time):
                overall_time_values[key].append(mean_time)
            
            if not math.isnan(mean_rmse):
                overall_rmse_values[key].append(mean_rmse)
    
    # Add a separator row
    rows.append(["", "", "", ""])
    rows.append(["=== AVERAGE METRICS ACROSS SEEDS ===", "", "", ""])
    
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
        feature_indices=None,
        plot_feature_indices=None,
        seed=0,
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5],
        epsilons=[0.5, 1], 
        target_scale=1e5,
        confidence_level=0.95,
        run_again=False,
        privacy_criteria=None,
        plot_interval=1  # Parameter to control plot density for algorithm curves and obfuscated data
    ):
    """
    Create plots showing privacy algorithm results for the census dataset.
    
    :param feature_indices: Indices of features used for modeling
    :param plot_feature_indices: Indices of features to use for plotting (subset of feature_indices)
                                If None, uses features from privacy criteria
    :param seed: Random seed for which results to use for plotting
    :param algs: List of algorithms to include in plots
    :param alphas: List of alpha values for privacy-aware GP
    :param epsilons: List of epsilon values for DP methods
    :param target_scale: Scale factor used for targets
    :param confidence_level: Confidence level for plotting prediction intervals
    :param run_again: Whether to run experiments again if needed
    :param privacy_criteria: Dictionary with criteria for private data selection
    :param plot_interval: If an integer, represents the interval for sampling points to plot for algorithm curves and obfuscated data.
                        If a float, represents the minimum distance between consecutive points (in feature value units).
                        Note: All true data points will be plotted regardless of this setting.
    :return: Results for the specified seed
    :rtype: dict
    """
    # Load results or run experiments if needed
    results_path = os.path.join(results_dir, 'census_all_results.pkl')
    
    if not os.path.exists(results_path) or run_again:
        print("Results not found or run_again=True. Running experiments...")
        _, all_results = run_multiple_seeds(
            feature_indices=feature_indices,
            seeds=[seed],
            algs=algs, 
            alphas=alphas, 
            epsilons=epsilons,
            target_scale=target_scale,
            privacy_criteria=privacy_criteria,
            run_again=True,
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
    
    # Load data to get feature information
    data_dict = load_census_data()
    feature_info = data_dict.get('feature_info', {})
    feature_names = feature_info.get('feature_names', ['Feature_0', 'Feature_1', 'Feature_2', 'Feature_3'])
    
    # Default to PUMA and AGEP if feature_indices is None
    if feature_indices is None:
        puma_idx = feature_names.index('PUMA') if 'PUMA' in feature_names else 0
        age_idx = feature_names.index('AGEP') if 'AGEP' in feature_names else 2
        feature_indices = [puma_idx, age_idx]
    
    # Create a modified feature_info that accounts for the subset of features used
    if feature_indices is not None:
        # Create a mapping from original indices to the new positions
        idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(feature_indices)}
        
        # Update feature_info with only the selected features
        modified_feature_info = {
            'feature_names': [feature_names[i] for i in feature_indices],
            'feature_means': [feature_info.get('feature_means', [])[i] if i < len(feature_info.get('feature_means', [])) else 0 for i in feature_indices],
            'feature_stds': [feature_info.get('feature_stds', [])[i] if i < len(feature_info.get('feature_stds', [])) else 1 for i in feature_indices]
        }
    else:
        # If no feature subsetting, use the original feature_info
        modified_feature_info = feature_info
        idx_mapping = {i: i for i in range(len(feature_names))}
    
    # If plot_feature_indices is None, determine which features to plot based on privacy_criteria
    if plot_feature_indices is None:
        if privacy_criteria is not None:
            # Use features mentioned in privacy criteria for plotting
            plot_feature_indices = []
            for feature_name in privacy_criteria.keys():
                if feature_name in feature_names:
                    orig_feature_idx = feature_names.index(feature_name)
                    # Check if this feature is included in our selected features
                    if orig_feature_idx in idx_mapping:
                        # Map to new index position
                        new_feature_idx = idx_mapping[orig_feature_idx]
                        plot_feature_indices.append(new_feature_idx)
            print(f"Automatically selected plot features from privacy criteria: {plot_feature_indices}")
            
        # If still None or empty, default to all feature_indices
        if not plot_feature_indices:
            # If feature_indices is None, use all available features
            if feature_indices is None:
                plot_feature_indices = list(range(min(2, len(feature_names))))
            else:
                # Use the new indices (0, 1, ...) since we've already remapped
                plot_feature_indices = list(range(min(2, len(feature_indices))))
            print(f"Using default features for plotting: {plot_feature_indices}")
    else:
        # Ensure plot_feature_indices are within range of our modified feature set
        plot_feature_indices = [idx for idx in plot_feature_indices 
                               if idx < len(modified_feature_info['feature_names'])]
        if not plot_feature_indices:
            plot_feature_indices = list(range(min(2, len(modified_feature_info['feature_names']))))
            print(f"Adjusted plot features to: {plot_feature_indices}")
    
    # Extract test data
    test_x = seed_results['test_x']
    test_y = seed_results['test_y']
    
    # Modify privacy_criteria to use new indices if we're using a subset of features
    modified_privacy_criteria = None
    if privacy_criteria is not None:
        modified_privacy_criteria = {}
        for feature_name, value_range in privacy_criteria.items():
            if feature_name in modified_feature_info['feature_names']:
                modified_privacy_criteria[feature_name] = value_range
    
    # Get privacy indices in test data using the modified criteria and feature info
    test_private_idx = get_privacy_indices(
        data=test_x,
        feature_info=modified_feature_info,
        privacy_criteria=modified_privacy_criteria,
        seed=seed
    ).numpy()
    
    # Create mask for private individuals in test data
    test_private_mask = np.zeros(len(test_x), dtype=bool)
    test_private_mask[test_private_idx] = True
    
    # Get privacy indices from seed_results
    privacy_idx = seed_results['privacy_idx'].numpy()
    
    # Convert confidence level to z-score
    z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + confidence_level / 2)).detach().cpu().item()
    
    # Define visual settings
    viz_settings = {
        'privacy-aware': {'color': 'green', 'style': '-', 'label': 'Privacy-aware GP', 'alpha': 0.2},
        'dp-cloaking': {'color': 'brown', 'style': '--', 'label': 'DP-Cloaking GP', 'alpha': 0.2},
        'dp-standard': {'color': 'yellow', 'style': '-.', 'label': 'Standard DP GP', 'alpha': 0.2},
        'exclude': {'color': 'darkorange', 'style': '--', 'label': 'GP for non-private data (Dropout)', 'alpha': 0.1},
        'attack': {'color': 'darkviolet', 'style': '--', 'label': 'Stationary GP for obfuscated data (Stationary)', 'alpha': 0.1},
    }
    
    # DP epsilon colors
    eps_colors = ['green', 'royalblue', 'darkviolet', 'blue', 'brown']
    
    # Convert test data to numpy
    test_x_np = test_x.numpy()
    test_y_np = test_y.numpy()
    
    # Helper function to plot predictions
    def add_predictions(ax, feature_idx, alg, param_value, custom_label=None, custom_color=None):
        if alg in seed_results['preds'] and param_value in seed_results['preds'][alg]:
            preds = seed_results['preds'][alg][param_value]
            
            # Sort by feature value for better visualization
            sort_idx = np.argsort(test_x_np[:, feature_idx])
            x_sorted = test_x_np[sort_idx, feature_idx]
            pred_mean = preds['mean'].numpy()[sort_idx]
            pred_var = preds['variance'].numpy()[sort_idx]
            
            # Sample points based on distance or interval
            if isinstance(plot_interval, float):
                # Distance-based sampling
                sample_idx = [0]  # Always include the first point
                last_x_value = x_sorted[0]
                
                for i in range(1, len(x_sorted)):
                    if abs(x_sorted[i] - last_x_value) >= plot_interval:
                        sample_idx.append(i)
                        last_x_value = x_sorted[i]
                
                # Always include the last point
                if len(x_sorted) > 1 and sample_idx[-1] != len(x_sorted) - 1:
                    sample_idx.append(len(x_sorted) - 1)
                
                sample_idx = np.array(sample_idx)
            else:
                # Regular interval-based sampling
                sample_idx = np.arange(0, len(x_sorted), plot_interval)
            
            x_sampled = x_sorted[sample_idx]
            mean_sampled = pred_mean[sample_idx]
            var_sampled = pred_var[sample_idx]
            
            # Get visualization settings
            settings = viz_settings[alg]
            color = custom_color if custom_color else settings['color']
            label = custom_label if custom_label else f"{settings['label']} ({param_value})"
            
            # Plot mean prediction
            ax.plot(x_sampled, mean_sampled, color=color, linestyle=settings['style'], linewidth=1.8, label=label)
            
            # Add confidence interval
            ax.fill_between(
                x_sampled,
                mean_sampled - z_score * np.sqrt(var_sampled),
                mean_sampled + z_score * np.sqrt(var_sampled),
                color=color,
                alpha=settings['alpha']
            )
    
    # Helper function to setup subplot with true data
    def setup_subplot(ax, feature_idx, is_leftmost=False):
        # Get feature name based on the feature index in our modified feature list
        feature_name = modified_feature_info['feature_names'][feature_idx] if feature_idx < len(modified_feature_info['feature_names']) else f"Feature_{feature_idx}"

        # Sort by feature value
        sort_idx = np.argsort(test_x_np[:, feature_idx])
        x_sorted = test_x_np[sort_idx, feature_idx]
        y_sorted = test_y_np[sort_idx]
        
        # FIXED: Instead of using the sorted mask, directly apply the privacy criteria to the sorted data
        # This ensures private data is correctly identified in the sorted view
        private_mask_sorted = np.zeros(len(x_sorted), dtype=bool)
        
        if modified_privacy_criteria:
            for _feature_name, value_range in modified_privacy_criteria.items():
                if _feature_name in modified_feature_info['feature_names']:
                    feat_idx = modified_feature_info['feature_names'].index(_feature_name)
                    min_val, max_val = value_range
                    
                    # Get the sorted values for this feature
                    if feat_idx == feature_idx:
                        # We're already sorted by this feature
                        feat_sorted = x_sorted
                    else:
                        # We need the sorted values of another feature
                        feat_sorted = test_x_np[sort_idx, feat_idx]
                    
                    # Update mask to include only records that match this criterion
                    feature_mask = (feat_sorted >= min_val) & (feat_sorted <= max_val)
                    
                    # For the first criterion, initialize the mask
                    if np.all(private_mask_sorted == False):
                        private_mask_sorted = feature_mask
                    else:
                        # For subsequent criteria, AND with existing mask
                        private_mask_sorted = private_mask_sorted & feature_mask
        else:
            # Fallback to using the original private mask if no criteria provided
            private_mask_sorted = test_private_mask[sort_idx]
        
        # Plot non-private data points
        ax.scatter(
            x_sorted[~private_mask_sorted], y_sorted[~private_mask_sorted], 
            color='black', marker='o', s=18, alpha=1.0, label='Non-private true data'
        )
        
        # Plot private data points
        ax.scatter(
            x_sorted[private_mask_sorted], y_sorted[private_mask_sorted], 
            color='red', marker='x', s=30, alpha=1.0, label='Private true data'
        )
        
        # Set x-label with correct feature name - MODIFIED to use more descriptive feature names
        if feature_name == 'AGEP':
            display_feature_name = 'Age'
        elif feature_name == 'PUMA':
            display_feature_name = 'Public Use Microdata Area (PUMA)'
        else:
            display_feature_name = feature_name
            
        ax.set_xlabel(f'{display_feature_name}', fontsize=14)
        
        # Set y-label with scale information - MODIFIED to only show on leftmost subplot
        if is_leftmost:
            # Format target_scale as power of 10 if it's a power of 10
            if np.log10(target_scale).is_integer():
                power = int(np.log10(target_scale))
                scale_str = rf'$10^{power}$'
            else:
                scale_str =  f'{target_scale}'
            
            ax.set_ylabel(f'Total Personal Income (*{scale_str})', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        return x_sorted, y_sorted, private_mask_sorted
    
    ############################################################
    # PLOT TYPE 1: Privacy-aware GP for each feature and alpha level
    ############################################################
    if 'privacy-aware' in algs:
        for feature_idx in plot_feature_indices:
            # Get feature name from modified feature info
            feature_name = modified_feature_info['feature_names'][feature_idx] if feature_idx < len(modified_feature_info['feature_names']) else f"Feature_{feature_idx}"
            
            # Create figure with subplots for each alpha
            fig, axes = plt.subplots(1, len(alphas), figsize=(5*len(alphas), 5))
            if len(alphas) == 1:
                axes = [axes]
            
            for i, alpha in enumerate(alphas):
                ax = axes[i]
                
                # Setup subplot with true data, flagging leftmost subplot
                x_sorted, y_sorted, private_mask_sorted = setup_subplot(ax, feature_idx, is_leftmost=(i==0))
                
                
                # Plot obfuscated data if available
                if 'obfuscated_data' in seed_results and alpha in seed_results['obfuscated_data']:
                    obf_data = seed_results['obfuscated_data'][alpha]
                    obf_x = obf_data['x'].numpy()
                    obf_y = obf_data['y'].numpy()
                    
                    # Sample obfuscated data points based on distance or interval
                    if isinstance(plot_interval, float):
                        # Sort obfuscated data by the feature we're plotting
                        obf_sort_idx = np.argsort(obf_x[:, feature_idx])
                        obf_x_sorted = obf_x[obf_sort_idx]
                        obf_y_sorted = obf_y[obf_sort_idx]
                        
                        # Distance-based sampling
                        obf_sample_idx = [0]  # Always include the first point
                        last_x_value = obf_x_sorted[0, feature_idx]
                        
                        for i in range(1, len(obf_x_sorted)):
                            if abs(obf_x_sorted[i, feature_idx] - last_x_value) >= plot_interval:
                                obf_sample_idx.append(i)
                                last_x_value = obf_x_sorted[i, feature_idx]
                        
                        # Always include the last point
                        if len(obf_x_sorted) > 1 and obf_sample_idx[-1] != len(obf_x_sorted) - 1:
                            obf_sample_idx.append(len(obf_x_sorted) - 1)
                        
                        obf_sample_idx = np.array(obf_sample_idx)
                        
                        # Use sorted and sampled data
                        obf_x_plot = obf_x_sorted[obf_sample_idx]
                        obf_y_plot = obf_y_sorted[obf_sample_idx]
                    else:
                        # Regular interval-based sampling
                        obf_sample_idx = np.arange(0, len(obf_x), plot_interval)
                        obf_x_plot = obf_x[obf_sample_idx]
                        obf_y_plot = obf_y[obf_sample_idx]
                    
                    ax.scatter(
                        obf_x_plot[:, feature_idx] if isinstance(plot_interval, float) else obf_x[obf_sample_idx, feature_idx],
                        obf_y_plot if isinstance(plot_interval, float) else obf_y[obf_sample_idx],
                        c='blue', s=15, alpha=1.0, 
                        label='Privacy-aware GP obfuscated data (Ours)'
                    )
                
                
                # Plot privacy-aware GP predictions
                add_predictions(ax, feature_idx, 'privacy-aware', alpha, 
                               f'Privacy-aware GP reconstructed prediction (Ours)', viz_settings['privacy-aware']['color'])
                
                # Plot exclude method if available
                if 'exclude' in algs:
                    add_predictions(ax, feature_idx, 'exclude', 'none', 
                                   viz_settings['exclude']['label'], viz_settings['exclude']['color'])
                
                # Plot attack method if available
                if 'attack' in algs and alpha in seed_results['preds'].get('attack', {}):
                    add_predictions(ax, feature_idx, 'attack', alpha, 
                                   viz_settings['attack']['label'], viz_settings['attack']['color'])
                
                
                # Set title with alpha value as per new requirements
                ax.set_title(f'H={alpha}K', color='blue', fontweight='bold', fontsize=18)
            
            # Create legend for all subplots with larger size and better position - FIXED
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
                ncol=min(3, len(handles)), fontsize=14.5,
            )
            
            # For the title, use more descriptive feature name
            if feature_name == 'AGEP':
                display_feature_name = 'Age'
            elif feature_name == 'PUMA':
                display_feature_name = 'Public Use Microdata Area (PUMA)'
            else:
                display_feature_name = feature_name
                
            # Add an overall title
            fig.suptitle(
                f'Census Income Prediction: Privacy-aware GP for {display_feature_name}', 
                color='black',
                fontweight='bold',
                fontsize=20, y=0.98,
            )
            
            # Save figure with adjusted layout to accommodate larger legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20)  # Increased bottom margin for legend
            fig_path = os.path.join(figs_dir, f'census_pa_{feature_name}_seed_{seed}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Privacy-aware GP figure for {feature_name} saved to: {fig_path}")
            plt.close(fig)
    
    ############################################################
    # PLOT TYPE 2: DP methods figures for all features
    ############################################################
    dp_algs = [alg for alg in ['dp-cloaking', 'dp-standard'] if alg in algs]
    
    if dp_algs:
        # Create figure with one subplot for each feature
        fig, axes = plt.subplots(1, len(plot_feature_indices), figsize=(5.5*len(plot_feature_indices), 5))
        if len(plot_feature_indices) == 1:
            axes = [axes]
        
        for i, feature_idx in enumerate(plot_feature_indices):
            ax = axes[i]
            # Get feature name from modified feature info
            feature_name = modified_feature_info['feature_names'][feature_idx] if feature_idx < len(modified_feature_info['feature_names']) else f"Feature_{feature_idx}"
            
            # Setup subplot with true data, flagging leftmost subplot
            x_sorted, y_sorted, private_mask_sorted = setup_subplot(ax, feature_idx, is_leftmost=(i==0))
            
            # Plot DP methods with different epsilon values
            for alg in dp_algs:
                for j, eps in enumerate(epsilons):
                    # Create label with epsilon and delta values
                    custom_label = rf'($\epsilon$={eps}, $\delta$=0.01)-{viz_settings[alg]["label"]}'
                    custom_color = eps_colors[j % len(eps_colors)]
                    
                    # Add predictions
                    add_predictions(ax, feature_idx, alg, eps, custom_label, custom_color)
            
            # Plot exclude method if available
            if 'exclude' in algs:
                add_predictions(ax, feature_idx, 'exclude', 'none', 
                               viz_settings['exclude']['label'], viz_settings['exclude']['color'])
        
        # Create legend for the first subplot with larger size - FIXED
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), 
                  ncol=min(3, len(handles)), fontsize=13)
        
        # Add an overall title
        fig.suptitle(
            'Census Income Prediction: Differential Privacy with GP',
            color='black',
            fontweight='bold',
            fontsize=20, y=1.0,
        )
        
        # Save figure with adjusted layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)  # Increased for larger legend
        fig_path = os.path.join(figs_dir, f'census_dp_seed_{seed}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"DP methods figure saved to: {fig_path}")
        plt.close(fig)
    
    return seed_results


def run(
        feature_indices=None,
        plot_feature_indices=None,
        seeds=range(5),
        algs=['privacy-aware', 'dp-cloaking', 'exclude', 'attack'],
        alphas=[0.1, 0.5, 0.9],
        epsilons=[0.5, 1],
        target_scale=1e5,
        verbose=True,
        plot_seed=0,
        run_again=False,
        privacy_criteria=None,
        plot_interval=1,  # Add plot_interval parameter
    ):
    """
    Run the full experimental pipeline: load data, run models with multiple seeds,
    compute metrics, save to CSV, and generate plots.
    
    :param feature_indices: Indices of features to use for modeling (None for all)
    :param plot_feature_indices: Indices of features to use for plotting (subset of feature_indices)
                                If None, uses features from privacy criteria
    :param seeds: Range of seeds to use
    :param algs: List of algorithms to run
    :param alphas: List of alpha values for privacy-aware GP
    :param epsilons: List of epsilon values for DP methods
    :param target_scale: Scale factor used for targets
    :param verbose: Whether to print verbose output
    :param plot_seed: Seed to use for plotting
    :param run_again: Whether to run experiments again even if results exist
    :param privacy_criteria: Dictionary with criteria for private data selection
    :param plot_interval: If an integer, represents the interval for sampling points to plot for algorithm curves and obfuscated data.
                        If a float, represents the minimum distance between consecutive points (in feature value units).
                        Note: All true data points will be plotted regardless of this setting.
    :return: Tuple containing aggregate metrics, CSV path, and seed results
    :rtype: tuple
    """
    # Paths for results
    results_path = os.path.join(results_dir, 'census_all_results.pkl')
    metrics_path = os.path.join(results_dir, 'census_all_metrics.pkl')
    
    # Run experiments or load existing results
    if not os.path.exists(results_path) or not os.path.exists(metrics_path) or run_again:
        aggregate_metrics, all_results = run_multiple_seeds(
            feature_indices=feature_indices,
            seeds=seeds,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons,
            target_scale=target_scale,
            verbose=verbose,
            run_again=run_again,
            privacy_criteria=privacy_criteria,
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
        with open(os.path.join(results_dir, 'census_aggregate_metrics.pkl'), 'rb') as f:
            aggregate_metrics = pickle.load(f)

        csv_path = os.path.join(results_dir, 'census_metrics.csv')
    
    # Plot results for the specified seed
    seed_results = plot_privacy_results(
        feature_indices=feature_indices,
        plot_feature_indices=plot_feature_indices,
        seed=plot_seed,
        algs=algs,
        alphas=alphas,
        epsilons=epsilons,
        target_scale=target_scale,
        run_again=False,  # We already checked run_again above
        privacy_criteria=privacy_criteria,
        plot_interval=plot_interval  # Pass the plot_interval parameter
    )
    
    return aggregate_metrics, csv_path, seed_results


if __name__ == "__main__":
    # Set parameters
    seeds = range(20)  # Run with 20 seeds
    #'feature_names': ['PUMA', 'POWPUMA', 'AGEP', 'ANC1P', 'ANC2P', 'YOEP', 'POBP', 'WKHP', 'WKWN']
    feature_indices = [0, 1, 2, 3, 6]  # Features to use for modeling data
    plot_feature_indices = None # Default plot feature indices to those used in privacy_criteria
    algs = ['privacy-aware', 'dp-cloaking', 'exclude', 'attack']
    alphas = [0.1, 0.5, 0.9]
    epsilons = [0.3, 0.5, 1.0]
    target_scale = 1e5  # Scale for target values
    verbose = True
    plot_seed = seeds[0]  # Seed to use for plotting
    run_again = False # Set to True to regenerate all results
    plot_interval = 1.0  # Minimum distance of 1.0 between consecutive points (in feature units)
    
    # Privacy criteria to use for identifying private data segments
    # Define criteria using feature names directly
    privacy_criteria = {
        'AGEP': (75, 100),  # Age between 75 and 100
        'PUMA': (0, 2000)  # PUMA between 0 and 2000
    }
    
    
    try:
        # Run the full experimental pipeline
        aggregate_metrics, csv_path, seed_results = run(
            feature_indices=feature_indices,
            plot_feature_indices=plot_feature_indices,
            seeds=seeds,
            algs=algs,
            alphas=alphas,
            epsilons=epsilons,
            target_scale=target_scale,
            verbose=verbose,
            plot_seed=plot_seed,
            run_again=run_again,
            privacy_criteria=privacy_criteria,
            plot_interval=plot_interval,  # Add plot_interval
        )
        
        print(f"Experiments completed. Results saved to: {csv_path}")
        
        # Print summary of results
        print("\nSummary of privacy-utility tradeoff:")
        for alg in algs:
            if alg in aggregate_metrics['mean_rmse']:
                print(f"\n{alg}:")
                for param, rmse in aggregate_metrics['mean_rmse'][alg].items():
                    privacy_loss = aggregate_metrics['mean_privacy_loss'][alg].get(param, 'N/A')
                    print(f"  Parameter: {param}, RMSE: {rmse:.8f}, Privacy Loss: {privacy_loss}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python dataset.py' first to prepare the census data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()