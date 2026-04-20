import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def simulate_sparse_data(alpha_sgd_small, alpha_sgd_large, alpha_ada, num_steps=250):
    np.random.seed(42) # For perfectly reproducible noise
    
    # Both features start at 1.0
    w_sgd_s = np.array([1.0, 1.0])
    w_sgd_l = np.array([1.0, 1.0])
    w_ada   = np.array([1.0, 1.0])
    
    G_ada = np.zeros(2)
    
    traj_sgd_s = [w_sgd_s.copy()]
    traj_sgd_l =[w_sgd_l.copy()]
    traj_ada   = [w_ada.copy()]
    
    for t in range(num_steps):
        # 1. Dense Feature: Appears every step, high noise
        noise = np.random.normal(0, 0.5)
        g0_s = w_sgd_s[0] + noise
        g0_l = w_sgd_l[0] + noise
        g0_a = w_ada[0] + noise
        
        # 2. Sparse Feature: Appears only every 25 steps (4% of the time), no noise
        is_active = (t % 25 == 0)
        g1_s = w_sgd_s[1] if is_active else 0.0
        g1_l = w_sgd_l[1] if is_active else 0.0
        g1_a = w_ada[1] if is_active else 0.0
        
        # SGD Updates
        w_sgd_s[0] -= alpha_sgd_small * g0_s
        w_sgd_s[1] -= alpha_sgd_small * g1_s
        traj_sgd_s.append(w_sgd_s.copy())
        
        w_sgd_l[0] -= alpha_sgd_large * g0_l
        w_sgd_l[1] -= alpha_sgd_large * g1_l
        traj_sgd_l.append(w_sgd_l.copy())
        
        # Adagrad Updates
        G_ada[0] += g0_a**2
        G_ada[1] += g1_a**2
        
        w_ada[0] -= (alpha_ada / (np.sqrt(G_ada[0]) + 1e-8)) * g0_a
        if is_active:
            w_ada[1] -= (alpha_ada / (np.sqrt(G_ada[1]) + 1e-8)) * g1_a
            
        traj_ada.append(w_ada.copy())
        
    return np.array(traj_sgd_s), np.array(traj_sgd_l), np.array(traj_ada)

def plot_sparse_adagrad():
    set_academic_style()
    num_steps = 250
    
    # Tuned hyperparameters for the perfect pedagogical visual
    traj_sgd_s, traj_sgd_l, traj_ada = simulate_sparse_data(
        alpha_sgd_small=0.05, 
        alpha_sgd_large=0.5, 
        alpha_ada=0.5, 
        num_steps=num_steps
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    iterations = np.arange(num_steps + 1)
    
    # --- PANEL A: Dense Feature ---
    # Z-order ensures the most important lines are on top
    ax1.plot(iterations, traj_sgd_l[:, 0], color="#e41a1c", linewidth=1.5, alpha=0.4, label=r'SGD (Large $\alpha$)', zorder=1)
    ax1.plot(iterations, traj_sgd_s[:, 0], color="#377eb8", linewidth=2.0, alpha=0.9, label=r'SGD (Small $\alpha$)', zorder=2)
    ax1.plot(iterations, traj_ada[:, 0], color="#4daf4a", linewidth=2.5, label='Adagrad', zorder=3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5, zorder=0)
    
    ax1.set_xlabel('Iteration $t$')
    ax1.set_ylabel(r'Feature Value $w_1$')
    ax1.set_title(r'(a) Dense, Noisy Feature (Appears 100\%)')
    ax1.legend(loc='upper right')
    ax1.set_ylim([-1.2, 1.5])
    ax1.grid(True, ls='--', alpha=0.5)
    
    # --- PANEL B: Sparse Feature ---
    ax2.plot(iterations, traj_sgd_l[:, 1], color="#e41a1c", linewidth=1.5, alpha=0.4, label=r'SGD (Large $\alpha$)', zorder=1)
    ax2.plot(iterations, traj_sgd_s[:, 1], color="#377eb8", linewidth=2.0, alpha=0.9, label=r'SGD (Small $\alpha$)', zorder=2)
    
    # We use a step-plot feel for Adagrad here to emphasize the discrete nature of the updates
    ax2.plot(iterations, traj_ada[:, 1], color="#4daf4a", linewidth=2.5, label='Adagrad', zorder=3)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5, zorder=0)
    
    ax2.set_xlabel('Iteration $t$')
    ax2.set_ylabel(r'Feature Value $w_2$')
    ax2.set_title(r'(b) Rare, Sparse Feature (Appears 4\%)')
    ax2.legend(loc='upper right')
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/sparse_adagrad.pdf")
    plt.close()

if __name__ == "__main__":
    plot_sparse_adagrad()
