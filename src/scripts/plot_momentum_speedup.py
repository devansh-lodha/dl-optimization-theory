import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style, get_colors

def plot_momentum_speedup():
    set_academic_style()
    colors = get_colors()
    
    kappa = 100.0
    lam_max = kappa
    lam_min = 1.0
    num_steps = 150
    
    # Initialize state
    w0 = np.array([1.0, 1.0])
    A = np.array([[lam_max, 0], [0, lam_min]])
    
    def grad(w):
        return A @ w
    
    # 1. Gradient Descent (Optimal Alpha)
    alpha_gd = 2.0 / (lam_max + lam_min)
    w_gd = w0.copy()
    err_gd = np.zeros(num_steps)
    
    # 2. Polyak Momentum (Optimal Alpha and Beta)
    alpha_mom = 4.0 / (np.sqrt(lam_max) + np.sqrt(lam_min))**2
    beta_mom = ((np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0))**2
    w_mom = w0.copy()
    z_mom = np.zeros(2)
    err_mom = np.zeros(num_steps)
    
    # 3. Nesterov Accelerated Gradient (NAG - Optimal Parameters)
    # NAG has a stricter stability bound than Polyak (alpha <= 2/L)
    alpha_nag = 1.0 / lam_max
    beta_nag = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    w_nag = w0.copy()
    w_nag_prev = w0.copy()
    err_nag = np.zeros(num_steps)
    
    for t in range(num_steps):
        # Log L2 error (distance to origin)
        err_gd[t] = np.linalg.norm(w_gd)
        err_mom[t] = np.linalg.norm(w_mom)
        err_nag[t] = np.linalg.norm(w_nag)
        
        # GD Update
        w_gd = w_gd - alpha_gd * grad(w_gd)
        
        # Polyak Update
        z_mom = beta_mom * z_mom + grad(w_mom)
        w_mom = w_mom - alpha_mom * z_mom
        
        # NAG Update (Canonical Formulation)
        # y_t is the lookahead point
        y_nag = w_nag + beta_nag * (w_nag - w_nag_prev)
        w_nag_prev = w_nag.copy()
        w_nag = y_nag - alpha_nag * grad(y_nag)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(err_gd, color=colors["sgd"], label='Gradient Descent', linewidth=2)
    ax.plot(err_mom, color=colors["momentum"], label='Polyak Momentum', linewidth=2)
    ax.plot(err_nag, color=colors["adam"], label='Nesterov (NAG)', linewidth=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel(r'Distance to Optimum $\|w^{(t)} - w^*\|_2$')
    ax.set_title(r'The Quadratic Speedup ($\kappa = 100$)')
    ax.legend(loc='lower left')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/momentum_speedup.pdf")
    plt.close()

if __name__ == "__main__":
    plot_momentum_speedup()
