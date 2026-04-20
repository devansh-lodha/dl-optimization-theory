import matplotlib.pyplot as plt

def set_academic_style():
    """Configures matplotlib for professional, publication-quality LaTeX plots."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "text.usetex": True,  # Requires a local LaTeX installation
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 11,
        "font.size": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf", # PDF is optimal for vector graphics in LaTeX
    })

def get_colors():
    """Returns a consistent color palette for optimization trajectories."""
    return {
        "sgd": "#e41a1c",      # Red
        "momentum": "#377eb8", # Blue
        "rmsprop": "#4daf4a",  # Green
        "adam": "#984ea3",     # Purple
        "lbfgs": "#ff7f00",    # Orange
        "contour": "#cccccc"   # Light Gray
    }
