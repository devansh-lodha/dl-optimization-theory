import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_config import set_academic_style, get_colors

def two_loop_recursion(g, S, Y):
    q = len(S)
    if q == 0:
        return -g.copy()
    
    alpha = np.zeros(q)
    rho = np.zeros(q)
    p = g.copy()
    
    # Backward pass
    for i in reversed(range(q)):
        rho[i] = 1.0 / (np.dot(Y[i], S[i]) + 1e-10)
        alpha[i] = rho[i] * np.dot(S[i], p)
        p = p - alpha[i] * Y[i]
        
    # Initial Hessian scaling
    gamma = np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-10)
    p = p * gamma
    
    # Forward pass
    for i in range(q):
        beta = rho[i] * np.dot(Y[i], p)
        p = p + S[i] * (alpha[i] - beta)
        
    return -p

def backtracking_line_search(A, w, p, g, alpha_init=1.0, rho=0.5, c=1e-4):
    """Real-world Armijo Backtracking Line Search"""
    alpha = alpha_init
    f_curr = 0.5 * np.dot(w, A @ w)
    m = np.dot(g, p)
    
    for _ in range(30):
        w_new = w + alpha * p
        f_new = 0.5 * np.dot(w_new, A @ w_new)
        # Sufficient decrease condition
        if f_new <= f_curr + c * alpha * m:
            return alpha
        alpha *= rho # Backtrack
        
    return alpha

def simulate_lbfgs(A, w0, m, num_steps):
    w = np.array(w0, dtype=float)
    S, Y = [],[]
    traj_f = [0.5 * np.dot(w, A @ w)]
    
    g = A @ w
    for _ in range(num_steps):
        if np.linalg.norm(g) < 1e-7:
            break
            
        p = two_loop_recursion(g, S, Y)
        
        # The key difference: using standard backtracking line search
        alpha = backtracking_line_search(A, w, p, g)
        
        s = alpha * p
        w_new = w + s
        g_new = A @ w_new
        y = g_new - g
        
        S.append(s)
        Y.append(y)
        if len(S) > m:
            S.pop(0)
            Y.pop(0)
            
        w = w_new
        g = g_new
        traj_f.append(0.5 * np.dot(w, A @ w))
        
    while len(traj_f) < num_steps + 1:
        traj_f.append(traj_f[-1])
        
    return np.array(traj_f)

def simulate_gd_exact(A, w0, num_steps):
    w = np.array(w0, dtype=float)
    traj_f = [0.5 * np.dot(w, A @ w)]
    g = A @ w
    
    for _ in range(num_steps):
        if np.linalg.norm(g) < 1e-7:
            break
        p = -g
        # We give GD exact line search so it serves as the absolute best-case theoretical baseline
        Ap = A @ p
        alpha = -np.dot(g, p) / (np.dot(p, Ap) + 1e-10)
        w = w + alpha * p
        g = A @ w
        traj_f.append(0.5 * np.dot(w, A @ w))
        
    while len(traj_f) < num_steps + 1:
        traj_f.append(traj_f[-1])
        
    return np.array(traj_f)

def plot_lbfgs_memory():
    set_academic_style()
    colors = get_colors()
    
    d = 100
    kappa = 1000.0
    
    # linspace creates a uniformly dense spectrum, maximizing the need for larger memory budgets
    lambdas = np.linspace(1.0, kappa, d)
    
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    A = Q @ np.diag(lambdas) @ Q.T
    
    w0 = Q @ np.ones(d)
    num_steps = 150
    
    dist_gd = simulate_gd_exact(A, w0, num_steps)
    dist_lbfgs_1 = simulate_lbfgs(A, w0, m=1, num_steps=num_steps)
    dist_lbfgs_5 = simulate_lbfgs(A, w0, m=5, num_steps=num_steps)
    dist_lbfgs_20 = simulate_lbfgs(A, w0, m=20, num_steps=num_steps)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    iterations = np.arange(num_steps + 1)
    
    ax.plot(iterations, dist_gd, color="#e41a1c", linewidth=2.0, label='GD (Exact Line Search)')
    ax.plot(iterations, dist_lbfgs_1, color="#4daf4a", linewidth=2.0, label=r'L-BFGS ($m=1$)')
    ax.plot(iterations, dist_lbfgs_5, color="#377eb8", linewidth=2.0, label=r'L-BFGS ($m=5$)')
    ax.plot(iterations, dist_lbfgs_20, color="#984ea3", linewidth=2.5, label=r'L-BFGS ($m=20$)')
    
    ax.set_yscale('log')
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel(r'Objective Value $f(w^{(t)})$')
    ax.set_title(r'L-BFGS Memory Budget on a Dense 100-D Quadratic ($\kappa=1000$)')
    ax.legend(loc='upper right')
    ax.grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs("tex/floats", exist_ok=True)
    plt.savefig("tex/floats/lbfgs_memory.pdf")
    plt.close()

if __name__ == "__main__":
    plot_lbfgs_memory()
    