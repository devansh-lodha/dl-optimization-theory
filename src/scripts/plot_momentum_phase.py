import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

def plot_momentum_phase():
    set_academic_style()
    
    alpha = 0.1
    lam = 1.0
    
    # Calculate critical beta where discriminant is strictly 0
    beta_critical = (1.0 - np.sqrt(alpha * lam))**2
    
    # Define three regimes
    betas = [0.1, beta_critical, 0.9]
    labels =[
        r'Overdamped ($\beta=0.1$)', 
        r'Critically Damped ($\beta \approx 0.468$)', 
        r'Underdamped ($\beta=0.9$)'
    ]
    colors =['#377eb8', '#4daf4a', '#e41a1c']
    
    # Increased steps from 40 to 150 to make the spiral mathematically smooth
    num_steps = 150 
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    
    for ax, beta, label, color in zip(axes, betas, labels, colors):
        x = np.zeros(num_steps)
        y = np.zeros(num_steps)
        
        # Initialization: start far from optimum (x=1.0) with zero velocity (y=0.0)
        x[0] = 1.0
        y[0] = 0.0
        
        # Simulate the state-space recursion
        for k in range(num_steps - 1):
            y[k+1] = beta * y[k] + lam * x[k]
            x[k+1] = x[k] - alpha * y[k+1]
            
        # Draw origin axes to ground the phase space into quadrants
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
        
        # Refined aesthetics: thinner lines, smaller dots, slight transparency
        ax.plot(x, y, 'o-', markersize=1.5, color=color, linewidth=1.0, alpha=0.85)
        
        # Clean start and end markers
        ax.plot(x[0], y[0], 'ko', markersize=4, label='Start' if ax == axes[0] else "")
        ax.plot(0, 0, 'k*', markersize=7, label='Equilibrium' if ax == axes[0] else "")
        
        ax.set_title(label)
        ax.set_xlabel(r'Position ($x_i$)')
        
        if ax == axes[0]:
            ax.set_ylabel(r'Velocity Accumulator ($y_i$)')
            ax.legend(loc='upper right')
        
        ax.grid(True, ls='--', alpha=0.3)
        
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/momentum_phase.pdf")
    plt.close()

if __name__ == "__main__":
    plot_momentum_phase()
