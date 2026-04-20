import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style, get_colors
from src.utils.functions import QuadraticBowl

def simulate_gd(func, w0, alpha, num_steps):
    """Simulates standard Gradient Descent."""
    w = np.array(w0, dtype=float)
    trajectory =[w.copy()]
    for _ in range(num_steps):
        grad = func.gradient(w)
        w = w - alpha * grad
        trajectory.append(w.copy())
    return np.array(trajectory)

def plot_pathological_curvature():
    set_academic_style()
    colors = get_colors()
    
    # Ill-conditioned quadratic (kappa = 10)
    func = QuadraticBowl(a=10.0, b=1.0)
    
    # Generate grid for contours
    x1 = np.linspace(-1.5, 1.5, 400)
    x2 = np.linspace(-1.5, 1.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * (func.a * X1**2 + func.b * X2**2)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot contours
    levels = np.logspace(-2, 1.5, 15)
    ax.contour(X1, X2, Z, levels=levels, colors=colors["contour"], linewidths=1.0)
    
    # Simulate Gradient Descent
    # Max learning rate before divergence is 2 / lambda_max = 0.2
    # We use 0.18 to clearly show the oscillation
    w0 = [-1.2, 1.0]
    trajectory = simulate_gd(func, w0, alpha=0.18, num_steps=15)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors["sgd"], 
            markersize=4, linewidth=1.5, label='Gradient Descent')
    
    # Plot start and end markers
    ax.plot(w0[0], w0[1], 'ko', markersize=6, label='Initialization')
    ax.plot(0, 0, 'b*', markersize=10, label="Global Minimum")
    
    ax.set_xlabel(r'$w_1$ (High Curvature)')
    ax.set_ylabel(r'$w_2$ (Low Curvature)')
    ax.set_title(r'Gradient Descent in an Ill-Conditioned Ravine ($\kappa=10$)')
    ax.legend(loc='upper right')
    
    # Lock aspect ratio so the geometric distortion is mathematically accurate
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal')
    
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/pathological_curvature.pdf")
    plt.close()

if __name__ == "__main__":
    plot_pathological_curvature()
