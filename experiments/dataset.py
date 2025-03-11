import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import csv
import argparse
import urllib.request
import zipfile
import shutil
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

###########################################
# Download and prepare Census PUMS data
###########################################
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                   reporthook=t.update_to)


def download_and_prepare_data(data_dir, sample_size=5000, seed=42):
    """
    Download Census PUMS data if it doesn't exist in the data directory.
    If sample_size is provided, randomly select that many rows from the original file.
    
    :param data_dir: Directory to save the data
    :param sample_size: Number of rows to randomly sample (None means use all data)
    :param seed: Random seed for reproducibility of sampling
    :return: Path to the prepared CSV file
    """
    base_filename = 'pums_ptx2023'
    
    # If sample_size is specified, include it in the filename
    target_file = os.path.join(data_dir, f'{base_filename}_sample_{sample_size}.csv')
    
    # If the file already exists, return its path
    if os.path.exists(target_file):
        print(f"Census data file already exists at {target_file}")
        return os.path.basename(target_file)
    
    # Create a temporary directory for downloading
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # URL of the census data
    url = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_ptx.zip"
    zip_path = os.path.join(temp_dir, 'census_data.zip')
    
    # Download the ZIP file
    print(f"Downloading census data from {url}...")
    try:
        download_url(url, zip_path)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Extract the ZIP file
    print("Extracting ZIP file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return None
    
    # Find the psam_p48.csv file
    source_file = None
    for root, _, files in os.walk(temp_dir):
        if 'psam_p48.csv' in files:
            source_file = os.path.join(root, 'psam_p48.csv')
            break
    
    if not source_file:
        print("Could not find psam_p48.csv in the extracted files")
        return None
    
    print(f"Found psam_p48.csv at {source_file}")
    
    # If no sampling is required, just copy the file
    if not sample_size:
        print(f"Copying entire file to {target_file}...")
        shutil.copy2(source_file, target_file)
    else:
        # Random sampling of rows
        print(f"Randomly sampling {sample_size} rows from the CSV file...")
        
        # Count total lines in the file
        with open(source_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Adjust sample_size if it exceeds total lines
        if sample_size >= total_lines - 1:  # -1 for header
            print(f"Sample size {sample_size} exceeds available data rows ({total_lines-1}). Using all rows.")
            shutil.copy2(source_file, target_file)
        else:
            # Generate random row indices (excluding header)
            selected_indices = set([0])  # Always include header (index 0)
            np.random.seed(seed)
            selected_indices.update(np.random.choice(range(1, total_lines), sample_size, replace=False))
            
            # Read and write selected rows
            with open(source_file, 'r') as source, open(target_file, 'w', newline='') as target:
                reader = csv.reader(source)
                writer = csv.writer(target)
                
                for i, row in enumerate(reader):
                    if i in selected_indices:
                        writer.writerow(row)
            
            print(f"Successfully sampled {sample_size} rows and saved to {target_file}")
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print(f"Census data prepared successfully at {target_file}")
    return os.path.basename(target_file)



###########################################
# Census Income Dataset Class
###########################################
class CensusIncomeDataset(Dataset):
    """
    PyTorch Dataset for Census Income data.
    
    This dataset handles the PUMS Census data and prepares it for 
    income prediction using PyTorch models.
    
    :param csv_file: Path to the csv file with the census data
    :param max_samples: Maximum number of samples to include in the dataset
    :param transform: Optional transform to be applied to features
    :param target_transform: Optional transform to be applied to the target
    :param test: If True, this is a test dataset with no targets
    """
    
    def __init__(
            self, csv_file, max_samples=None, 
            transform=None, target_transform=None, 
            test=False, seed=42, normalize_features=True,
        ):
        # Define selected columns based on requirements
        self.selected_columns = {
            'target': 'PINCP',
            'features': [
                'PUMA', 'POWPUMA', 'AGEP', 'ANC1P', 'ANC2P', 
                'YOEP', 'POBP', 'WKHP', 'WKWN'
            ]
        }
        
        # All features are treated as numerical
        self.num_indices = list(range(len(self.selected_columns['features'])))
        
        # Store transformations
        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.normalize_features = normalize_features
        
        # Load and preprocess data
        self._load_data(os.path.join(data_dir, csv_file), max_samples, seed=seed)
    
    def _load_data(self, csv_file, max_samples, seed=42):
        """
        Load and preprocess the raw data without using pandas.
        
        :param csv_file: Path to the CSV file
        :param max_samples: Maximum number of samples to include
        """
        # Read the CSV file
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # First row is the header
            
            # Find the indices of our target and feature columns
            col_indices = {}
            for col_name in self.selected_columns['features'] + [self.selected_columns['target']]:
                try:
                    col_indices[col_name] = header.index(col_name)
                except ValueError:
                    if col_name != self.selected_columns['target'] or not self.test:
                        raise ValueError(f"Column {col_name} not found in the dataset")
            
            # Read all data rows
            data_rows = list(reader)
            
            # Sample if necessary
            if max_samples is not None and max_samples < len(data_rows):
                np.random.seed(seed)
                sampled_indices = np.random.choice(len(data_rows), max_samples, replace=False)
                data_rows = [data_rows[i] for i in sampled_indices]
        
        # Extract features and target
        X = []
        y = []
        
        for row in data_rows:
            # Extract and convert features
            features = []
            valid_row = True
            
            for col_name in self.selected_columns['features']:
                col_idx = col_indices[col_name]
                value = row[col_idx] if col_idx < len(row) and row[col_idx] else ''
                
                # Convert to float, handle empty strings and non-numeric values
                try:
                    value = float(value) if value else float('nan')
                    features.append(value)
                except ValueError:
                    valid_row = False
                    break
            
            if not valid_row:
                continue
                
            X.append(features)
            
            # Extract and convert target if not in test mode
            if not self.test:
                target_idx = col_indices.get(self.selected_columns['target'])
                if target_idx is not None and target_idx < len(row):
                    target_value = row[target_idx]
                    try:
                        target_value = float(target_value) if target_value else float('nan')
                        if not np.isnan(target_value):  # Skip rows with missing target
                            y.append(target_value)
                        else:
                            X.pop()
                    except ValueError:
                        X.pop()
                else:
                    X.pop()
        
        # Convert to numpy arrays
        self.X = np.array(X, dtype=float)
        if not self.test:
            self.y = np.array(y, dtype=float)
            
        # Calculate statistics for normalization
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate statistics for normalization of numerical features."""
        # Replace any potential NaN or infinite values with 0
        numerical_X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate mean and std for each numerical feature
        self.feature_means = np.nanmean(numerical_X, axis=0)
        self.feature_stds = np.nanstd(numerical_X, axis=0)
        # Replace zeros in std with 1 to avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
    
    def preprocess_features(self, x):
        """
        Apply normalization to features.
        
        :param x: Raw feature values
        :return: Preprocessed features tensor
        """
        # Normalize numerical features
        x_processed = x.copy().astype(float)
        for i in range(len(self.num_indices)):
            if not np.isnan(x_processed[i]):
                if self.normalize_features:
                    x_processed[i] = (x_processed[i] - self.feature_means[i]) / self.feature_stds[i]
            else:
                x_processed[i] = 0.0
        
        return torch.tensor(x_processed, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = self.X[idx]
        
        # Apply preprocessing
        features = self.preprocess_features(features)
        
        # Apply custom transform if provided
        if self.transform:
            features = self.transform(features)
        
        if self.test:
            return features
        else:
            target = self.y[idx]
            
            # target = np.log1p(target)
            target = torch.tensor(target, dtype=torch.float32)
            
            # Apply custom target transform if provided
            if self.target_transform:
                target = self.target_transform(target)
            
            return features, target


###########################################
# Normalization transform class
###########################################
class Normalize(torch.nn.Module):
    """
    Normalization class that computes statistics along a specified dimension.
    
    :param dim: (int) Dimension along which to normalize. Default is 0.
    :param eps: (float) Small value to prevent division by zero
    :return: Normalized tensor
    """
    def __init__(self, dim=0, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
    def forward(self, tensor):
        # Always compute mean and std along the specified dimension
        mean = tensor.mean(dim=self.dim)
        std = tensor.std(dim=self.dim)
        
        # Prevent division by zero
        if isinstance(std, torch.Tensor):
            std = std.clamp(min=self.eps)
        else:
            std = max(std, self.eps)
        
        # Apply normalization
        return (tensor - mean) / std


###########################################
# Data preparation and loading functions
###########################################
def prepare_data_and_save(csv_file, output_dir, max_samples=10000, test_size=0.2, seed=42, normalize_features=True):
    """
    Preprocess census data and save train/test sets to disk.
    
    :param csv_file: Path to the input CSV file
    :param output_dir: Directory to save the preprocessed data
    :param max_samples: Maximum number of samples to process
    :param test_size: Proportion of data to use for testing
    :param seed: Random seed for reproducibility
    """
    
    # Create dataset with size limit
    dataset = CensusIncomeDataset(
        csv_file, 
        max_samples=max_samples,
        transform=None,  # Normalize(dim=0) # Normalize features
        target_transform=None, 
        test=False, 
        seed=seed, 
        normalize_features=normalize_features,
    )
    
    # Split into train and test sets
    test_length = int(len(dataset) * test_size)
    train_length = len(dataset) - test_length
    
    train_dataset, test_dataset = random_split(
        dataset, [train_length, test_length], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Process and collect data
    def process_split(split_dataset):
        inputs = []
        targets = []
        for idx in split_dataset.indices:
            input, target = dataset[idx]
            inputs.append(input)
            targets.append(target)
        return torch.stack(inputs), torch.stack(targets)
    
    train_inputs, train_targets = process_split(train_dataset)
    test_inputs, test_targets = process_split(test_dataset)
    
    # Save all data in one file
    data_dict = {
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'test_inputs': test_inputs,
        'test_targets': test_targets,
        'feature_info': {
            'feature_names': dataset.selected_columns['features'],
            'feature_means': dataset.feature_means.tolist(),
            'feature_stds': dataset.feature_stds.tolist()
        }
    }
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    torch.save(data_dict, os.path.join(output_dir, 'census_data.pt'))
    
    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test inputs shape: {test_inputs.shape}")
    print(f"Test targets shape: {test_targets.shape}")
    
    return {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'feature_dim': train_inputs.shape[1]
    }


def get_dataloaders(csv_file, batch_size=32, max_samples=10000, test_size=0.2, seed=42, normalize_features=True):
    """
    Create PyTorch DataLoaders for training and testing.
    
    :param csv_file: Path to the csv file with census data
    :param batch_size: Batch size for DataLoader
    :param max_samples: Maximum number of samples to include in the dataset
    :param test_size: Proportion of the dataset to include in the test split
    :param seed: Random seed for reproducibility
    :return: train_loader, test_loader
    """
    
    # Create dataset with size limit
    dataset = CensusIncomeDataset(
        csv_file, 
        max_samples=max_samples,
        transform=None,
        target_transform=None,
        test=False,
        seed=seed,
        normalize_features=normalize_features,
    )
    
    # Split into train and test sets
    test_length = int(len(dataset) * test_size)
    train_length = len(dataset) - test_length
    
    train_dataset, test_dataset = random_split(
        dataset, [train_length, test_length], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if os.name != 'nt' else 0,  # 0 for Windows compatibility
        generator=torch.Generator().manual_seed(seed), # Ensure reproducibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if os.name != 'nt' else 0,  # 0 for Windows compatibility
        generator=torch.Generator().manual_seed(seed), # Ensure reproducibility
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Preprocess census data for PyTorch')
    parser.add_argument('--input', type=str, default='pums_ptx2023_sample_5000.csv', 
                        help='Input CSV file path')
    parser.add_argument('--max_samples', type=int, default=2000, 
                        help='Maximum number of samples to process for model training')
    parser.add_argument('--csv_sample_size', type=int, default=5000, 
                        help='Number of rows to randomly sample from the original CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed for reproducibility')
    parser.add_argument('--download', action='store_true',
                        help='Download census data if not available')
    parser.add_argument('--normalize_features', action='store_true', default=False,
                        help='Normalize features in the dataset')
    args = parser.parse_args()
    
    # Check if we need to download the data
    input_file = args.input
    if args.download or not os.path.exists(os.path.join(data_dir, input_file)):
        print("Census data file not found or download requested")
        downloaded_file = download_and_prepare_data(
            data_dir, 
            sample_size=args.csv_sample_size,
            seed=args.seed
        )
        if downloaded_file:
            input_file = downloaded_file
        else:
            print("Failed to download census data. Please provide a valid input file.")
            exit(1)
    
    # Process and save the data
    print(f"Processing census data from {input_file}...")
    print(f"Using up to {args.max_samples} samples for model training")
    
    try:
        stats = prepare_data_and_save(
            csv_file=input_file,
            output_dir=data_dir,
            max_samples=args.max_samples,
            test_size=args.test_size,
            seed=args.seed,
            normalize_features=args.normalize_features,
        )
        
        print("\nSuccessfully preprocessed census data!")
        print(f"Preprocessed data saved to: {data_dir}")
        print(f"Train set: {stats['train_size']} samples")
        print(f"Test set: {stats['test_size']} samples")
        print(f"Feature dimensionality: {stats['feature_dim']}")
        print("\nTo use this data in your PyTorch models:")
        print("  data_dict = torch.load(os.path.join(data_dir, 'census_data.pt'))")
        print("  train_inputs = data_dict['train_inputs']")
        print("  train_targets = data_dict['train_targets']")
        print("  test_inputs = data_dict['test_inputs']")
        print("  test_targets = data_dict['test_targets']")
        
    except Exception as e:
        print(f"Error processing data: {e}")