# Optimization Methods in Deep Learning

This repository contains the source code, mathematical simulations, and LaTeX manuscript for a comprehensive analysis of optimization algorithms used in deep learning. 

The objective of this project is to bridge theoretical geometry with practical algorithm dynamics. It derives the mathematical limits of first-order methods, the physics of momentum, the geometry of adaptive learning rates, and the linear algebra underlying quasi-Newton methods.

## Repository Architecture

The project is structured into modular Python scripts for mathematical simulation and modular LaTeX files for the manuscript.

```text
dl-optimization-theory/
├── pyproject.toml
├── src/
│   ├── utils/
│   │   ├── plot_config.py       # Global matplotlib academic styling
│   │   └── functions.py         # Mathematical test surfaces (e.g., QuadraticBowl, Rosenbrock)
│   └── scripts/                 # Simulation and visualization scripts
└── tex/
    ├── main.tex                 # LaTeX master document
    ├── references.bib           # Validated BibTeX citations
    ├── scribe.sty               # Academic stylesheet
    └── sections/                # Modular LaTeX content
```

## Mathematical Modules

1. **Mathematical Preliminaries:** Taylor expansions, Hessian condition numbers, and pathological curvature.
2. **First-Order Limits:** Exact dynamics of Gradient Descent on convex quadratics, spectral decay, stochastic noise balls, and Robbins-Monro conditions.
3. **Momentum Physics:** Polyak Heavy Ball as a discrete-time damped harmonic oscillator, phase space analysis, Nesterov Accelerated Gradient (NAG), and the quadratic speedup.
4. **Adaptive Matrices:** The sparse feature problem, dynamic diagonal preconditioning via Adagrad, the monotonicity flaw, and RMSProp.
5. **Adam and Bias Correction:** Fusing momentum and adaptive variance, with a rigorous mathematical proof of the exponential moving average (EMA) initialization bias and its exact correction.
6. **Second-Order and Secant Conditions:** Newton's Method, the necessity of Armijo backtracking line search, and the derivation of the BFGS update rule.
7. **Limited-Memory BFGS:** The two-loop recursion algorithm, memory budget stratification on high-dimensional correlated quadratics, and the fundamental limitations of quasi-Newton methods in stochastic regimes.

## Setup and Installation

This project uses `uv` for dependency management and `hatchling` as the build backend. 

1. Install `uv` if not already installed.
2. Initialize the virtual environment and install the package in editable mode:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Reproducing Visualizations

All figures in the manuscript are generated programmatically. The scripts simulate exact algorithmic dynamics and output publication-quality PDFs directly to the `tex/floats/` directory.

Run the scripts as Python modules from the project root:

```bash
uv run python -m src.scripts.plot_curvature
uv run python -m src.scripts.plot_spectral_decay
uv run python -m src.scripts.plot_sgd_noise
uv run python -m src.scripts.plot_momentum_phase
uv run python -m src.scripts.plot_momentum_speedup
uv run python -m src.scripts.plot_sparse_adagrad
uv run python -m src.scripts.plot_adaptive_lr
uv run python -m src.scripts.plot_adam_dynamics
uv run python -m src.scripts.plot_newton_bfgs
uv run python -m src.scripts.plot_lbfgs_memory
```

## Compiling the Manuscript

Ensure you have a standard TeX distribution installed (e.g., TeX Live or MacTeX). To compile the manuscript and correctly link the bibliography, execute the following build chain from inside the `tex/` directory:

```bash
cd tex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled output will be available as `main.pdf`.
```