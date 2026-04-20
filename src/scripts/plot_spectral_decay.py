import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def plot_spectral_decay():
    set_academic_style()
    
    # Define the eigenspectrum of the Hessian
    lambdas =[1.0, 5.0, 10.0]
    labels =[
        r'$\lambda_{min} = 1.0$ (Slowest)', 
        r'$\lambda_{mid} = 5.0$ (Fastest)', 
        r'$\lambda_{max} = 10.0$ (Slowest)'
    ]
    colors =['#377eb8', '#4daf4a', '#e41a1c']
    
    # Calculate the mathematically optimal step size
    alpha = 2.0 / (lambdas[0] + lambdas[-1])
    
    iterations = np.arange(0, 30)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for lam, label, color in zip(lambdas, labels, colors):
        # Component-wise error decay: x_t = (1 - alpha * lambda)^t * x_0
        # We assume initial error magnitude |x_0| = 1 for all components
        decay_factor = np.abs(1.0 - alpha * lam)
        error = decay_factor ** iterations
        ax.plot(iterations, error, marker='o', markersize=4, label=label, color=color, linewidth=1.5)
        
    ax.set_yscale('log')
    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel(r'Absolute Error $|x_i^{(t)}|$')
    ax.set_title(r'Spectral Error Decay under Optimal Step Size $\alpha^*$')
    
    # FIX: Moved legend to the bottom left where the plot is completely empty
    ax.legend(loc='lower left')
    
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/spectral_decay.pdf")
    plt.close()

if __name__ == "__main__":
    plot_spectral_decay()
