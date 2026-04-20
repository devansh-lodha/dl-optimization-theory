import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def simulate_momentum(func, w0, alpha, beta, num_steps):
    w = np.array(w0, dtype=float)
    z = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        z = beta * z + g
        w = w - alpha * z
        traj.append(w.copy())
    return np.array(traj)

def simulate_rmsprop(func, w0, alpha, beta2, num_steps, epsilon=1e-8):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        v = beta2 * v + (1 - beta2) * (g**2)
        w = w - (alpha / (np.sqrt(v) + epsilon)) * g
        traj.append(w.copy())
    return np.array(traj)

def simulate_adam(func, w0, alpha, beta1, beta2, num_steps, epsilon=1e-8):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    traj = [w.copy()]
    for t in range(1, num_steps + 1):
        g = func.gradient(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        w = w - (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
        traj.append(w.copy())
    return np.array(traj)

class QuadraticBowl:
    def __init__(self, a=1.0, b=10.0):
        self.a = a
        self.b = b
    def __call__(self, x):
        return 0.5 * (self.a * x[0]**2 + self.b * x[1]**2)
    def gradient(self, x):
        return np.array([self.a * x[0], self.b * x[1]])

def plot_adam_dynamics():
    set_academic_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    # --- PANEL A: Bias Correction ---
    # Simulate a stationary gradient distribution E[g] = 1.0 with some noise
    np.random.seed(42)
    num_bias_steps = 100
    true_expected_g = 1.0
    g_t = np.random.normal(true_expected_g, 0.5, num_bias_steps)
    
    beta = 0.9
    m_raw = np.zeros(num_bias_steps)
    m_corrected = np.zeros(num_bias_steps)
    
    m_prev = 0.0
    for t in range(num_bias_steps):
        # Raw EMA update
        m_prev = beta * m_prev + (1 - beta) * g_t[t]
        m_raw[t] = m_prev
        # Bias-corrected EMA
        m_corrected[t] = m_prev / (1 - beta**(t+1))
        
    iterations = np.arange(1, num_bias_steps + 1)
    
    # FIX: Used standard \mathbf{E} instead of the custom \E macro
    ax1.axhline(true_expected_g, color='k', linestyle='--', alpha=0.6, label=r'True Expectation $\mathbf{E}[g]$')
    ax1.plot(iterations, m_raw, color="#e41a1c", linewidth=2.0, label=r'Raw EMA $m_t$')
    ax1.plot(iterations, m_corrected, color="#4daf4a", linewidth=2.0, label=r'Bias-Corrected $\hat{m}_t$')
    
    ax1.set_xlabel('Iteration $t$')
    ax1.set_ylabel('Moment Estimate')
    ax1.set_title(r'(a) The Initialization Bias ($\beta=0.9$)')
    ax1.legend(loc='lower right')
    ax1.grid(True, ls='--', alpha=0.5)
    
    # --- PANEL B: Adam Trajectory ---
    func = QuadraticBowl(a=1.0, b=10.0)
    w0 =[-2.0, 2.0]
    num_steps = 150
    
    # Standard hyperparameters for comparison
    traj_mom = simulate_momentum(func, w0, alpha=0.03, beta=0.9, num_steps=num_steps)
    traj_rms = simulate_rmsprop(func, w0, alpha=0.1, beta2=0.999, num_steps=num_steps, epsilon=0.1)
    traj_adam = simulate_adam(func, w0, alpha=0.1, beta1=0.9, beta2=0.999, num_steps=num_steps, epsilon=0.1)
    
    x1 = np.linspace(-2.5, 2.5, 400)
    x2 = np.linspace(-2.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func([X1, X2])
            
    levels = np.logspace(-1, 2, 12)
    ax2.contour(X1, X2, Z, levels=levels, colors="#cccccc", linewidths=0.8, zorder=1)
    
    # Plot Trajectories
    ax2.plot(traj_mom[:, 0], traj_mom[:, 1], '-', color="#377eb8", linewidth=1.5, alpha=0.8, label='Momentum', zorder=2)
    ax2.plot(traj_rms[:, 0], traj_rms[:, 1], '-', color="#984ea3", linewidth=1.5, alpha=0.9, label='RMSProp', zorder=3)
    ax2.plot(traj_adam[:, 0], traj_adam[:, 1], '-', color="#ff7f00", linewidth=2.5, alpha=0.9, label='Adam', zorder=4)
    
    ax2.plot(traj_mom[-1, 0], traj_mom[-1, 1], 'o', color="#377eb8", markersize=5, zorder=5)
    ax2.plot(traj_rms[-1, 0], traj_rms[-1, 1], 'd', color="#984ea3", markersize=5, zorder=5)
    ax2.plot(traj_adam[-1, 0], traj_adam[-1, 1], 's', color="#ff7f00", markersize=5, zorder=5)
    
    ax2.plot(w0[0], w0[1], 'ko', markersize=6, label='Init', zorder=6)
    ax2.plot(0, 0, 'b*', markersize=10, label='Minimum', zorder=6)
    
    ax2.set_xlabel(r'$w_1$ (Low Curvature)')
    ax2.set_ylabel(r'$w_2$ (High Curvature)')
    ax2.set_title(r'(b) Parameter Trajectories')
    ax2.legend(loc='lower right')
    ax2.set_xlim([-2.5, 2.5])
    ax2.set_ylim([-2.5, 2.5])
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/adam_dynamics.pdf")
    plt.close()

if __name__ == "__main__":
    plot_adam_dynamics()
