import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def optimize_psi_sqp(t, x_d, y_d, x_dd, y_dd, psi0=None, max_iters=5):
    # Importación local para evitar dependencias circulares
    from c2_flatness_reconstruct import _psi_dot_ode
    
    N = len(t)
    dt = t[1] - t[0] # Asume pasos de tiempo uniformes del planificador
    
    if psi0 is None:
        speed0 = np.hypot(x_d[0], y_d[0])
        psi0 = float(np.arctan2(y_d[0], x_d[0])) if speed0 > 0.001 else 0.0

    # Función auxiliar para vectorizar la evaluación de la EDO
    def eval_ode(psi_vec):
        return np.array([_psi_dot_ode(psi_vec[k], x_d[k], y_d[k], x_dd[k], y_dd[k]) for k in range(N)])
        
    # 1. Semilla Inicial: Modelo plano idealizado (m11 = m22)
    psi_guess = np.unwrap(np.arctan2(y_d, x_d))
    psi_guess[0] = psi0
    
    # 2. Bucle SQP (Iteraciones Algebraicas)
    for iteration in range(max_iters):
        f_val = eval_ode(psi_guess)
        
        # Cálculo del Jacobiano Numérico (Serie de Taylor de 1er orden)
        eps = 1e-4
        f_plus = eval_ode(psi_guess + eps)
        f_minus = eval_ode(psi_guess - eps)
        J_val = (f_plus - f_minus) / (2.0 * eps)
        
        # 3. Ensamblaje del Sistema Lineal (Matriz H / A del QP)
        # Representa la derivada polinomial a tramos: (psi_k - psi_{k-1})/dt
        main_diag = np.zeros(N)
        lower_diag = np.zeros(N-1)
        b = np.zeros(N)
        
        # Condición de frontera inicial
        main_diag[0] = 1.0
        b[0] = psi0
        
        for k in range(1, N):
            main_diag[k] = 1.0 / dt - J_val[k]
            lower_diag[k-1] = -1.0 / dt
            b[k] = f_val[k] - J_val[k] * psi_guess[k]
            
        A = diags([lower_diag, main_diag], [-1, 0], format='csr')
        
        # 4. Solución Lineal Pura (sin RK4, sin NLP)
        psi_new = spsolve(A, b)
        
        # 5. Criterio de convergencia temprana
        max_diff = np.max(np.abs(psi_new - psi_guess))
        psi_guess = psi_new
        
        if max_diff < 1e-5:
            break
            
    r_opt = eval_ode(psi_guess)
    return psi_guess, r_opt