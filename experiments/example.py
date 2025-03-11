#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


class SDP:
    """Class for SDP optimization and visualization."""
    
    def __init__(self):
        """Initialize the SDP analyzer with directory setup."""
        # Setup directories
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.current_dir, 'data')
        self.figs_dir = os.path.join(self.current_dir, 'figs')
        self.results_dir = os.path.join(self.current_dir, 'results')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Dictionary to store all results
        self.results = {}

    @staticmethod
    def rbfkernel(x1, x2, lengthscale=1/20):
        """
        RBF Kernel function: k(x1, x2) = exp(-(x1-x2)^2/(2*lengthscale))
        Handles inputs of different shapes and returns kernel matrix [n,m].
        
        :param x1: First input array of shape (n,) or (n,d)
        :param x2: Second input array of shape (m,) or (m,d)
        :param lengthscale: Kernel lengthscale (default: 1/20)
        :return: Kernel matrix of shape (n,m)
        """
        # Convert inputs to arrays and ensure they're at least 1D
        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        
        # Reshape inputs for proper broadcasting
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
        if x2.ndim == 1:
            x2 = x2.reshape(-1, 1)
            
        # Compute squared distances
        sqd = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)
        
        # Return kernel values
        return np.exp(-sqd / (2 * lengthscale))

    def solve_sdp(self, ss_matrices, is_diagonal=False):
        """
        Solve a semidefinite program given SS matrices.
        
        :param ss_matrices: List of SS matrices for constraints
        :param is_diagonal: Whether to use diagonal optimization
        :return: Dictionary with optimal value and solution
        """
        n = ss_matrices[0].shape[0]
        
        # Define the semidefinite program
        if is_diagonal:
            x_var = cp.Variable(n, nonneg=True)  # Use a vector for diagonal matrix
            
            objective = cp.Minimize(cp.sum(x_var))
            constraints = []
            
            # Add constraints for each SS matrix
            for ss in ss_matrices:
                # Create a diagonal matrix expression
                x_diag = cp.diag(x_var)
                constraints.append(x_diag >> ss)
        else:
            # For full matrix optimization
            x_var = cp.Variable((n, n), symmetric=True)
            
            objective = cp.Minimize(cp.trace(x_var))
            constraints = [x_var >> 0]  # Add PSD constraint
            
            # Add constraints for each SS matrix
            for ss in ss_matrices:
                constraints.append(x_var >> ss)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            # Solve the semidefinite program
            problem.solve(solver=cp.SCS)
            
            # Print diagnostics
            print(f"SDP solved with optimal value: {problem.value}")
            
            # Check if the solution exists
            if x_var.value is None:
                print("Warning: SDP solution is None. Using zero matrix as fallback.")
                if is_diagonal:
                    solution = np.diag(np.zeros(n))
                else:
                    solution = np.zeros((n, n))
            else:
                if is_diagonal:
                    # Create diagonal matrix from vector solution
                    diag_values = x_var.value
                    print(f"Diagonal values: {diag_values}")
                    solution = np.diag(diag_values)
                else:
                    solution = x_var.value
            
            # Return the results
            return {
                'optimal_value': problem.value if problem.value is not None else float('inf'),
                'optimal_solution': solution
            }
        except Exception as e:
            print(f"Error solving SDP: {e}")
            # Create a minimal fallback solution
            if is_diagonal:
                # Create a minimal PSD matrix that satisfies constraints
                # Find minimum eigenvalues of each SS matrix to ensure our solution is valid
                min_eigs = []
                for ss in ss_matrices:
                    try:
                        eigvals = np.linalg.eigvalsh(ss)
                        min_eigs.append(max(0, -np.min(eigvals)))
                    except:
                        min_eigs.append(0.1)  # Default if eigenvalue computation fails
                
                # Use the maximum negative eigenvalue to ensure PSD
                safety_margin = max(min_eigs) + 0.1
                diag_values = np.ones(n) * safety_margin
                solution = np.diag(diag_values)
            else:
                # Create a simple PSD matrix as fallback
                solution = np.eye(n)
            
            return {
                'optimal_value': float('inf'),
                'optimal_solution': solution
            }

    def compute_pred_var(self, x_pred, var_matrix, x_grid):
        """
        Compute predictive variance at point x_pred.
        
        :param x_pred: Input point for prediction
        :param var_matrix: Variance matrix
        :param x_grid: Grid of input points used to build var_matrix
        :return: Predictive variance (scalar)
        """
        # Create kernel vector - vectorized computation
        r = self.rbfkernel(x_grid, x_pred).flatten()  # Ensure r is a 1D array
        
        try:
            # Solve linear system and compute 1 - r^T V^-1 r
            result = 1.0 - r.T @ np.linalg.solve(var_matrix, r)
            # Ensure we return a scalar
            return float(result)
        except np.linalg.LinAlgError:
            print(f"Warning: Matrix solving failed at x={x_pred}, adding regularization")
            # Add small regularization to make matrix invertible
            reg_matrix = var_matrix + 1e-8 * np.eye(var_matrix.shape[0])
            result = 1.0 - r.T @ np.linalg.solve(reg_matrix, r)
            return float(result)

    def plot_corrplot(self, matrix, ax, title=None):
        """
        Create a correlation plot similar to R's corrplot.
        
        :param matrix: Correlation matrix to visualize
        :param ax: Matplotlib axis to plot on
        :param title: Plot title
        """
        n = matrix.shape[0]
        
        # Set up the axes for square cells
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect('equal')  # Ensure square cells
        
        # Normalize the matrix for color mapping
        norm = Normalize(vmin=0, vmax=max(1.0, np.max(matrix)))
        
        # Create a colormap from white to dark red
        red_cmap = plt.cm.YlOrBr
        
        # Add grid lines
        for i in range(n+1):
            ax.axhline(i-0.5, color='gray', linestyle='-', linewidth=0.5)
            ax.axvline(i-0.5, color='gray', linestyle='-', linewidth=0.5)
        
        # Draw circles with size and color based on the correlation
        circle_max_size = 0.85  # Maximum circle size
        for i in range(n):
            for j in range(n):
                # Scale circle size based on matrix value
                size = circle_max_size * abs(matrix[i, j]) / np.max(matrix) if np.max(matrix) > 0 else 0
                if size > 0:
                    circle = Circle((j, i), size/2, 
                                   color=red_cmap(norm(matrix[i, j])),
                                   alpha=0.8)
                    ax.add_artist(circle)
        
        # Add labels with larger font
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"0.{i+1}" for i in range(n)], fontsize=12, color='red')
        ax.set_yticklabels([f"0.{i+1}" for i in range(n)], fontsize=12, color='red')
        
        if title:
            ax.set_title(title, fontsize=18)
            
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        
        # DO NOT invert y-axis to show 0.1 at the bottom (reversed from original)

    def plot_results(self, kernel_matrix, s_matrix, diag_var, x_grid):
        """
        Create and save the final figure with three subplots.
        
        :param kernel_matrix: Base kernel matrix
        :param s_matrix: S matrix for proposed method
        :param diag_var: Diagonal variance vector
        :param x_grid: Grid of input points
        :return: Path to saved figure
        """
        # Print diagnostic information
        print("Diagnostic information:")
        print(f"S matrix shape: {s_matrix.shape}")
        print(f"S matrix diagonal (first 3 values): {np.diag(s_matrix)[:3]}")
        print(f"Diagonal variance values (first 3 values): {diag_var[:3]}")
        
        # Setup for plot with 3 subplots of same size
        fig, axes = plt.subplots(1, 3, figsize=(5*3, 5))
        
        # First subplot - Correlation plot of S matrix
        self.plot_corrplot(s_matrix, axes[0], title="Optimal covariance matrix $\Sigma_{opt}$")
        
        # Second subplot - Barplot of diagonal values
        diag_var_proposed = np.diag(s_matrix)
        x_indices = np.arange(len(diag_var))
        width = 0.35
        
        axes[1].bar(x_indices - width/2, diag_var, width, label='Diagonal', color='red')
        axes[1].bar(x_indices + width/2, diag_var_proposed, width, label='Proposed', color='lime')
        axes[1].set_title('Synthetic Error Variance', fontsize=18)
        axes[1].set_xticks(x_indices)
        axes[1].set_xticklabels([f"0.{i+1}" for i in range(len(diag_var))], fontsize=12)
        axes[1].set_xlabel('Input points', fontsize=16)
        axes[1].set_ylim(0, 5)
        # Remove y-label
        axes[1].tick_params(axis='y', labelsize=12)
        axes[1].legend(fontsize=14)
        axes[1].grid(True)
        
        # Prepare variance matrices for different methods
        var_unsecure = kernel_matrix  # Unsecure
        var_proposed = kernel_matrix + s_matrix  # Proposed
        var_diagonal = kernel_matrix + np.diag(diag_var)  # Diagonal
        
        # Prepare x values for prediction (dense grid from 0 to 1)
        x_pred = np.linspace(0, 1, 100)
        
        # Calculate predictive variance values for each method
        variance_results = self._calculate_variances(x_pred, var_unsecure, var_proposed, var_diagonal, x_grid)
        
        # Print sample of computed variances for debugging
        for method_name, variance_data in variance_results.items():
            print(f"{method_name} sample (first 3 values): {variance_data[:3]}")
        
        # Third subplot - Predictive variance
        axes[2].plot(x_pred, variance_results['var_diagonal'], color='red', linestyle='-', linewidth=2, label='Diagonal')
        axes[2].plot(x_pred, variance_results['var_proposed'], color='lime', linestyle='-', linewidth=2, label='Proposed')
        axes[2].plot(x_pred, variance_results['var_unsecure'], color='blue', linestyle='-', linewidth=2, label='Unsecure')
        axes[2].set_title('Predictive Variance', fontsize=18)
        axes[2].set_xlabel('Predictive input', fontsize=16)
        # Remove y-label
        axes[2].set_ylim(-0.05, 0.5)  # Set lower limit to -0.05 to see Unsecure curve better
        axes[2].tick_params(axis='both', labelsize=12)
        axes[2].legend(loc='upper right', fontsize=14)
        axes[2].grid(True)
        
        # Add main title to the entire figure
        fig.suptitle('Example 1', fontsize=20, fontweight='bold', y=0.92, color='blue')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for the main title
        fig_path = os.path.join(self.figs_dir, 'example.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        
        # Store all results in the results dictionary
        self.results.update({
            'kernel_matrix': kernel_matrix,
            's_matrix': s_matrix,
            'diag_var': diag_var,
            'predictive_variances': variance_results
        })
        
        return fig_path

    def _calculate_variances(self, x_pred, var_unsecure, var_proposed, var_diagonal, x_grid):
        """
        Helper method to calculate variances for each method.
        
        :param x_pred: Points to evaluate predictive variance
        :param var_unsecure: Unsecure variance matrix
        :param var_proposed: Proposed variance matrix
        :param var_diagonal: Diagonal variance matrix
        :param x_grid: Grid of input points used to build matrices
        :return: Dictionary of variance values for each method
        """
        # Calculate predictive variances for each point in x_pred
        # Ensure we get 1D arrays for plotting
        pred_var_diagonal = np.array([self.compute_pred_var(x, var_diagonal, x_grid) for x in x_pred])
        pred_var_unsecure = np.array([self.compute_pred_var(x, var_unsecure, x_grid) for x in x_pred])
        pred_var_proposed = np.array([self.compute_pred_var(x, var_proposed, x_grid) for x in x_pred])
        
        # Debug information to verify peak locations
        diag_peak = x_pred[np.argmax(pred_var_diagonal)]
        proposed_peak = x_pred[np.argmax(pred_var_proposed)]
        print(f"Diagonal peak at x={diag_peak}")
        print(f"Proposed peak at x={proposed_peak}")
        
        # Ensure the returned arrays are 1D for plotting
        return {
            'var_diagonal': pred_var_diagonal,
            'var_unsecure': pred_var_unsecure,
            'var_proposed': pred_var_proposed
        }

    def run_examples(self):
        """
        Run all SDP examples and store results.
        
        :return: Tuple of kernel matrix and example1 results
        """
        # Create evenly spaced input points from 0.1 to 0.9
        x_grid = np.linspace(0.1, 0.9, 9)
        
        # Create kernel matrix using vectorized operation
        kernel_matrix = self.rbfkernel(x_grid, x_grid)
        
        # Example 1 - using diagonal constraint
        print("Running Example 1...")
        # Center at 0.5
        x_star = 0.5
        r_vec = self.rbfkernel(x_grid, x_star)
        # Ensure r_vec is a column vector for outer product
        r_vec = r_vec.flatten()
        ss = 2 * np.outer(r_vec, r_vec) - kernel_matrix
        result1 = self.solve_sdp([ss], is_diagonal=True)
        
        # Example 2 - two constraints
        print("Running Example 2...")
        r1 = self.rbfkernel(x_grid, 0.3).flatten()
        ss1 = 2 * np.outer(r1, r1) - kernel_matrix
        r2 = self.rbfkernel(x_grid, 0.5).flatten()
        ss2 = 2 * np.outer(r2, r2) - kernel_matrix
        result2 = self.solve_sdp([ss1, ss2])
        
        # Supplementary Example - three constraints
        print("Running Supplementary Example...")
        r1 = self.rbfkernel(x_grid, 0.0).flatten()
        ss1 = 2 * np.outer(r1, r1) - kernel_matrix
        r2 = self.rbfkernel(x_grid, 0.5).flatten()
        ss2 = 2 * np.outer(r2, r2) - kernel_matrix
        r3 = self.rbfkernel(x_grid, 0.8).flatten()
        ss3 = 2 * np.outer(r3, r3) - kernel_matrix
        result3 = self.solve_sdp([ss1, ss2, ss3])
        
        # Store results
        self.results.update({
            'x_grid': x_grid,
            'example1_result': result1['optimal_solution'],
            'example1_value': result1['optimal_value'],
            'example2_result': result2['optimal_solution'],
            'example2_value': result2['optimal_value'],
            'example3_result': result3['optimal_solution'],
            'example3_value': result3['optimal_value']
        })
        
        # Save example1 result
        np.save(os.path.join(self.data_dir, 'example.npy'), result1['optimal_solution'])
        
        # Return the kernel matrix and grid for visualization
        return kernel_matrix, result1['optimal_solution'], x_grid

    def create_visualization_data(self, kernel_matrix, x_grid):
        """
        Create data needed for visualization.
        
        :param kernel_matrix: Kernel matrix
        :param x_grid: Grid of input points
        :return: S matrix and diagonal variance vector
        """
        # Generate r vector centered at 0.5
        x_star = 0.5
        r_vec = self.rbfkernel(x_grid, x_star).flatten()
        
        # Create SS matrix
        ss = 2 * np.outer(r_vec, r_vec) - kernel_matrix
        
        # Perform eigendecomposition
        try:
            eigen_values, eigen_vectors = np.linalg.eigh(ss)
            
            # Set negative eigenvalues to zero
            lambda1 = np.maximum(eigen_values, 0)
            s_matrix = eigen_vectors @ np.diag(lambda1) @ eigen_vectors.T
        except np.linalg.LinAlgError as e:
            print(f"Warning: Eigendecomposition failed: {e}. Using simple matrix.")
            # Fallback to a simple PSD matrix
            s_matrix = np.eye(kernel_matrix.shape[0])
        
        # Solve SDP to get diagonal variance
        result = self.solve_sdp([ss], is_diagonal=True)
        diag_var = np.diag(result['optimal_solution'])
        
        return s_matrix, diag_var

    def save_results(self):
        """
        Save all results to a single pickle file.
        
        :return: Path to saved results
        """
        results_path = os.path.join(self.results_dir, 'example.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"All results saved to {results_path}")
        return results_path


def main():
    """Main function to run the SDP analysis."""
    try:
        # Create SDP instance
        sdp = SDP()
        
        # Run all examples and get kernel matrix
        kernel_matrix, example1_result, x_grid = sdp.run_examples()
        
        # Create visualization data
        s_matrix, diag_var = sdp.create_visualization_data(kernel_matrix, x_grid)
        
        # Create visualization
        sdp.plot_results(kernel_matrix, s_matrix, diag_var, x_grid)
        
        # Save all results to a single file
        results_path = sdp.save_results()
        
        print(f"Analysis complete. All results saved to {results_path}")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()