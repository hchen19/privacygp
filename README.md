# Privacy-aware GP
This repository provides an implementation of privacy-aware Gaussian processes built on [PyTorch](https://pytorch.org).


# Requirements
- Python >= 3.10
- Dependencies are managed through `pyproject.toml`

# Usage
To reproduce the experiments, run the following commands step by step in your terminal:

1. Clone the repository or download the folder, ensure you are in the [privacygp](/) directory

2. Create and activate a virtual environment:
```bash
python -m venv .env
source .env/bin/activate
```

3. Upgrade pip and setuptools to ensure compatibility:
```bash
pip install --upgrade pip setuptools wheel
```

4. Install the package and all dependencies:
```bash
pip install .
```

5. Run the experiments

## Example
- To reproduce the example 1 in Figure 1, run the following command
    ```bash
    python experiments/example.py
    ```

## Experiments
### Satellite simulation
- To reproduce the satellite simulation in Figure 2, 3, 4 and Table 2, run the following command
    ```bash
    python experiments/satellite_simulation.py
    ```

- To reproduce the satellite simulation with zero-mean GP in Figure 5, run the following command
    ```bash
    python experiments/zeromean_satellite_simulation.py
    ```

### Real-world application (Census dataset)
- To reproduce the real-world application on the [PUMS Data](https://www.census.gov/programs-surveys/acs/microdata/access.html) provided by [the U.S. Census Bureau](https://www.census.gov/) in Figure 6, 7 and Table 3, run the following command
    ```bash
    python experiments/census.py
    ```