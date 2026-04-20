import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style

class Rosenbrock:
    def __call__(self, w):
        x, y = w[0], w[1]
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2
        
    def gradient(self, w):
        x, y = w[0], w[1]
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])
        
    def hessian(self, w):
        x, y = w[0], w[1]
        dxx = 2.0 - 400.0 * (y - x**2) + 800.0 * x**2
        dxy = -400.0 * x
        dyy = 200.0
        return np.array([[dxx, dxy], [dxy, dyy]])

def backtracking_line_search(func, w, p, g, alpha_init=1.0, rho=0.5, c=1e-4):
    """Armijo Backtracking Line Search to guarantee sufficient decrease."""
    alpha = alpha_init
    f_curr = func(w)
    m = np.dot(g, p)
    # Shrink alpha until the loss sufficiently decreases
    while func(w + alpha * p) > f_curr + c * alpha * m:
        alpha *= rho
        if alpha < 1e-8:  # Safety failsafe
            break
    return alpha

def simulate_newton(func, w0, num_steps):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        if np.linalg.norm(g) < 1e-5:
            break
            
        H = func.hessian(w)
        
        # Regularize Hessian if it's not positive definite (Modified Newton)
        eigvals = np.linalg.eigvals(H)
        if np.min(eigvals) <= 1e-5:
            H = H + np.eye(len(w)) * (-np.min(eigvals) + 1e-3)
            
        p = np.linalg.solve(H, -g)
        
        # Use Line Search to prevent massive overshooting
        alpha = backtracking_line_search(func, w, p, g)
        w = w + alpha * p
        traj.append(w.copy())
    return np.array(traj)

def simulate_bfgs(func, w0, num_steps):
    w = np.array(w0, dtype=float)
    I = np.eye(len(w))
    H_inv = I.copy()
    traj =[w.copy()]
    
    for _ in range(num_steps):
        g = func.gradient(w)
        if np.linalg.norm(g) < 1e-5:
            break
            
        # Step direction
        p = -H_inv @ g
        
        # Line search guarantees y^T s > 0 curvature condition is met
        alpha = backtracking_line_search(func, w, p, g)
        s = alpha * p
        w_new = w + s
        
        g_new = func.gradient(w_new)
        y = g_new - g
        
        # BFGS Secant Update
        ys = np.dot(y, s)
        if ys > 1e-10:
            rho = 1.0 / ys
            A = I - rho * np.outer(s, y)
            B = I - rho * np.outer(y, s)
            H_inv = A @ H_inv @ B + rho * np.outer(s, s)
            
        w = w_new
        traj.append(w.copy())
        
    return np.array(traj)

def simulate_gd(func, w0, num_steps):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    for _ in range(num_steps):
        g = func.gradient(w)
        p = -g
        # GD also requires line search on Rosenbrock to prevent divergence
        alpha = backtracking_line_search(func, w, p, g, alpha_init=0.01)
        w = w + alpha * p
        traj.append(w.copy())
    return np.array(traj)

def plot_newton_bfgs():
    set_academic_style()
    func = Rosenbrock()
    w0 =[-1.2, 1.0]
    
    # Simulate
    traj_gd = simulate_gd(func, w0, num_steps=2000)
    traj_bfgs = simulate_bfgs(func, w0, num_steps=100)
    traj_newton = simulate_newton(func, w0, num_steps=100)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    x1 = np.linspace(-1.5, 1.5, 400)
    x2 = np.linspace(-0.5, 1.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func([X1, X2])
            
    levels = np.logspace(-1, 3, 20)
    ax.contour(X1, X2, Z, levels=levels, colors="#cccccc", linewidths=0.8, zorder=1)
    
    # Subsample GD trajectory so it isn't a solid block of red ink
    gd_plot = traj_gd[::20]
    ax.plot(gd_plot[:, 0], gd_plot[:, 1], '-', color="#e41a1c", linewidth=2.0, alpha=0.8, label='Gradient Descent', zorder=2)
    
    ax.plot(traj_bfgs[:, 0], traj_bfgs[:, 1], 'o-', color="#984ea3", linewidth=2.0, markersize=4, alpha=0.9, label='BFGS', zorder=3)
    ax.plot(traj_newton[:, 0], traj_newton[:, 1], 's-', color="#377eb8", linewidth=2.0, markersize=5, label="Newton's Method", zorder=4)
    
    ax.plot(traj_gd[-1, 0], traj_gd[-1, 1], 'o', color="#e41a1c", markersize=6, zorder=5)
    ax.plot(traj_bfgs[-1, 0], traj_bfgs[-1, 1], 'd', color="#984ea3", markersize=6, zorder=5)
    ax.plot(traj_newton[-1, 0], traj_newton[-1, 1], 's', color="#377eb8", markersize=6, zorder=5)
    
    ax.plot(w0[0], w0[1], 'ko', markersize=6, label='Init', zorder=6)
    ax.plot(1, 1, 'b*', markersize=10, label='Minimum', zorder=6)
    
    ax.set_xlabel(r'$w_1$')
    ax.set_ylabel(r'$w_2$')
    ax.set_title(r'Second-Order vs Quasi-Newton Methods')
    ax.legend(loc='lower right')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/newton_bfgs.pdf")
    plt.close()

if __name__ == "__main__":
    plot_newton_bfgs()
