# Privacy-aware GP
This repository provides an implementation of privacy-aware Gaussian processes built on [PyTorch](https://pytorch.org).


# Usage
To reproduce the experiments, run the following commands step by step in your terminal:
1.  Clone the repository
```bash
git clone https://github.com/hchen19/privacygp.git
```

2. Create virtual environment and install necessary packages
```bash
cd privacygp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the experiments

- Ensure that you are in the [privacygp](/) directory and that it is included in your `PYTHONPATH` environment variable. You can do this by running:
    ```bash
    export PYTHONPATH="$PWD"
    ```

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
- To reproduce the real-world application on the [PUMS Data](https://www.census.gov/programs-surveys/acs/microdata/access.html) provided by [the U.S. Census Bureau](https://www.census.gov/) in Figure 6. 7 and Table 3, run the following command
    ```bash
    python experiments/census.py
    ```