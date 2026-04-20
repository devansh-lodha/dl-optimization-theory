import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def simulate_noisy_sgd(func, w0, alpha_schedule, num_steps, noise_std=0.5, seed=42):
    np.random.seed(seed)
    w = np.array(w0, dtype=float)
    traj = np.zeros((num_steps + 1, len(w)))
    traj[0] = w.copy()
    
    for t in range(num_steps):
        g_true = func.gradient(w)
        noise = np.random.normal(0, noise_std, size=w.shape)
        g_stochastic = g_true + noise
        
        alpha = alpha_schedule(t)
        w = w - alpha * g_stochastic
        traj[t+1] = w.copy()
        
    return traj

class IsotropicQuadratic:
    def __init__(self, a=1.0):
        self.a = a
    def __call__(self, x):
        return 0.5 * self.a * np.sum(x**2)
    def gradient(self, x):
        return self.a * x

def plot_sgd_noise():
    set_academic_style()
    
    func = IsotropicQuadratic(a=1.0)
    w0 =[2.0, 2.0]
    num_steps = 1000
    noise_std = 0.8
    
    # 1. Constant Learning Rate
    alpha_const = 0.05
    sched_const = lambda t: alpha_const
    traj_const = simulate_noisy_sgd(func, w0, sched_const, num_steps, noise_std, seed=42)
    
    # 2. Decaying Learning Rate (Starts at the EXACT same alpha to ensure fair comparison)
    alpha_0 = 0.05
    decay = 0.01
    sched_decay = lambda t: alpha_0 / (1.0 + decay * t)
    traj_decay = simulate_noisy_sgd(func, w0, sched_decay, num_steps, noise_std, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    # --- PANEL A: Trajectory ---
    x1 = np.linspace(-1.0, 2.5, 400)
    x2 = np.linspace(-1.0, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * (X1**2 + X2**2)
            
    levels = np.logspace(-2, 1.5, 10)
    ax1.contour(X1, X2, Z, levels=levels, colors="#cccccc", linewidths=0.8, zorder=1)
    
    ax1.plot(traj_const[:, 0], traj_const[:, 1], '-', color="#e41a1c", linewidth=1.0, alpha=0.5, label='Constant LR', zorder=2)
    ax1.plot(traj_decay[:, 0], traj_decay[:, 1], '-', color="#377eb8", linewidth=1.5, alpha=0.8, label='Decaying LR', zorder=3)
    
    ax1.plot(w0[0], w0[1], 'ko', markersize=6, label='Init', zorder=4)
    ax1.plot(0, 0, 'k*', markersize=10, label='Minimum', zorder=4)
    
    ax1.set_xlabel(r'$w_1$')
    ax1.set_ylabel(r'$w_2$')
    ax1.set_title(r'(a) Stochastic Trajectories')
    ax1.legend(loc='lower left')
    ax1.set_xlim([-1.0, 2.5])
    ax1.set_ylim([-1.0, 2.5])
    ax1.set_aspect('equal')
    
    # --- PANEL B: Convergence Plot ---
    dist_sq_const = np.sum(traj_const**2, axis=1)
    dist_sq_decay = np.sum(traj_decay**2, axis=1)
    iterations = np.arange(num_steps + 1)
    
    # Moving average to expose the expected value (variance floor)
    window = 40
    def moving_average(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate((a[:n-1], ret[n - 1:] / n))

    ax2.plot(iterations, moving_average(dist_sq_const, window), color="#e41a1c", linewidth=2.0, label='Constant LR')
    ax2.plot(iterations, moving_average(dist_sq_decay, window), color="#377eb8", linewidth=2.0, label='Decaying LR')
    
    # Theoretical variance floor: alpha * d * sigma^2 / 2
    # Here: a=1, d=2, sigma=0.8, alpha=0.05
    theoretical_floor = (alpha_const * 2 * (noise_std**2)) / 2.0
    ax2.axhline(theoretical_floor, color='k', linestyle='--', alpha=0.6, label=r'Theoretical Noise Floor')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration $t$')
    ax2.set_ylabel(r'Squared Distance $\|w^{(t)} - w^*\|_2^2$')
    ax2.set_title(r'(b) Convergence of Expected Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/sgd_noise.pdf")
    plt.close()

if __name__ == "__main__":
    plot_sgd_noise()
