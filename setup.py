from setuptools import setup

setup(
    name="privacy-aware-gp",
    version="0.1.0",
    packages=["privacygp", "experiments"],
    python_requires=">=3.9",
    install_requires=[
        "tqdm==4.66.3",
        "torch==2.4.0",
        "gpytorch==1.13",
        "torchdiffeq==0.2.5",
        "numpy==1.26.4",
        "matplotlib==3.8.0",
        "scipy==1.12.0",
        "cvxpy==1.6.4",
    ],
)