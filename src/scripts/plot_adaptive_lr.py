import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def simulate_adagrad(func, w0, alpha, num_steps, epsilon=1e-8):
    w = np.array(w0, dtype=float)
    G = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        G += g**2
        w = w - (alpha / (np.sqrt(G) + epsilon)) * g
        traj.append(w.copy())
    return np.array(traj)

def simulate_rmsprop(func, w0, alpha, gamma, num_steps, epsilon=1e-8):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    traj =[w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        v = gamma * v + (1 - gamma) * (g**2)
        w = w - (alpha / (np.sqrt(v) + epsilon)) * g
        traj.append(w.copy())
    return np.array(traj)

def simulate_sgd(func, w0, alpha, num_steps):
    w = np.array(w0, dtype=float)
    traj =[w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        w = w - alpha * g
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

def plot_adaptive_lr():
    set_academic_style()
    
    # The absolute standard: flat on w1, steep on w2
    func = QuadraticBowl(a=1.0, b=10.0)
    w0 =[-2.0, 2.0]
    num_steps = 150
    
    # 1. SGD: Optimal-ish alpha. Causes heavy zigzag on w2, slow crawl on w1.
    traj_sgd = simulate_sgd(func, w0, alpha=0.15, num_steps=num_steps)
    
    # 2. Adagrad: Standard alpha. Visibly stalls out due to accumulating G.
    traj_ada = simulate_adagrad(func, w0, alpha=0.3, num_steps=num_steps)
    
    # 3. RMSProp: Standard alpha. Epsilon naturally caps the terminal step size to prevent jitter.
    traj_rms = simulate_rmsprop(func, w0, alpha=0.05, gamma=0.9, num_steps=num_steps, epsilon=0.25)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    # --- PANEL A: Trajectory ---
    x1 = np.linspace(-2.5, 2.5, 400)
    x2 = np.linspace(-2.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func([X1, X2])
            
    levels = np.logspace(-1, 2, 12)
    ax1.contour(X1, X2, Z, levels=levels, colors="#cccccc", linewidths=0.8, zorder=1)
    
    # Plot SGD
    ax1.plot(traj_sgd[:, 0], traj_sgd[:, 1], '-', color="#e41a1c", linewidth=1.5, alpha=0.7, label='SGD', zorder=2)
    
    # Plot RMSProp (Thick background line)
    ax1.plot(traj_rms[:, 0], traj_rms[:, 1], '-', color="#984ea3", linewidth=3.0, alpha=0.7, label='RMSProp', zorder=3)
    
    # Plot Adagrad (Thin line with markers resting perfectly on top of the RMSProp path to show the stall)
    ax1.plot(traj_ada[:, 0], traj_ada[:, 1], 's-', color="#4daf4a", linewidth=1.5, markersize=3, label='Adagrad', zorder=4)
    
    ax1.plot(w0[0], w0[1], 'ko', markersize=6, label='Init', zorder=5)
    ax1.plot(0, 0, 'b*', markersize=10, label='Minimum', zorder=5)
    
    ax1.set_xlabel(r'$w_1$ (Low Curvature)')
    ax1.set_ylabel(r'$w_2$ (High Curvature)')
    ax1.set_title(r'(a) Parameter Trajectories')
    ax1.legend(loc='lower right')
    ax1.set_xlim([-2.5, 2.5])
    ax1.set_ylim([-2.5, 2.5])
    ax1.set_aspect('equal')
    
    # --- PANEL B: Convergence Plot ---
    dist_sgd = np.linalg.norm(traj_sgd, axis=1)
    dist_ada = np.linalg.norm(traj_ada, axis=1)
    dist_rms = np.linalg.norm(traj_rms, axis=1)
    iterations = np.arange(num_steps + 1)
    
    ax2.plot(iterations, dist_sgd, color="#e41a1c", linewidth=2.0, label='SGD')
    ax2.plot(iterations, dist_rms, color="#984ea3", linewidth=2.0, label='RMSProp')
    ax2.plot(iterations, dist_ada, color="#4daf4a", linewidth=2.0, label='Adagrad (Stalled)')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration $t$')
    ax2.set_ylabel(r'Distance to Optimum $\|w^{(t)} - w^*\|_2$')
    ax2.set_title(r'(b) Convergence Rate')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/adaptive_lr.pdf")
    plt.close()

if __name__ == "__main__":
    plot_adaptive_lr()
