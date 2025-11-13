# framework_utils.py

import numpy as np
from scipy.integrate import solve_ivp

# --- 1. CORE DYNAMICS (framework_utils.py) ---
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)
positivity_floor = 1e-8
DEFAULT_INIT_STATE = (25.0, 25.0, 25.0) 
DEFAULT_LAMBDA = 0.5 

def awareness_dynamics(t, y, lambda_A0):
    """
    Defines the core Ordinary Differential Equations (ODEs) for the framework.
    """
    O, M, P = y
    # dO: Observation
    dO = lambda_A0 * (1 - O) * (1 + 0.1 * P) - 0.1 * O * M
    # dM: Memory
    dM = 0.05 * O - 0.02 * M
    # dP: Pattern
    dP = 0.01 * M - 0.01 * P
    
    # Enforce positivity 
    dO = max(dO, -O + positivity_floor)
    dM = max(dM, -M + positivity_floor)
    dP = max(dP, -P + positivity_floor)
    return [dO, dM, dP]

def solve_core_dynamics(init_state, lambda_A0):
    """
    Solves the awareness_dynamics ODEs for a given initial state and lambda_A0.
    """
    sol = solve_ivp(
        awareness_dynamics, t_span, init_state, 
        args=(lambda_A0,), t_eval=t_eval, method='RK45'
    )
    O, M, P = sol.y
    R = O * M * P # R = Awareness
    return sol.t, O, M, P, R