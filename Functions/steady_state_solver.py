import numpy as np
from scipy.optimize import fsolve
from math import exp
import warnings
from scipy.optimize import OptimizeWarning

def steady_state_smc(params, vari_init_vals):
    """
    Solve the steady-state Ca_in_SMC0, Ca_SR, and y using fsolve.
    
    params    : dictionary containing all numeric parameter values
    vari_init_vals : dictionary with initial guesses for Ca_in_SMC and Ca_SR
    Returns   : Ca_in_SMC, Ca_SR, y (floats)
    """

    # Extract parameters
    p = {k.split('/')[-1]: v for k, v in params.items()}
    # Normalize vari_init_vals keys to short names
    vari_init_vals = {k.split('/')[-1]: v for k, v in vari_init_vals.items()}


    # 1) Solve Eq1 for Ca_in_SMC0
    A = -p['alpha1'] * p['gca'] * p['V0'] * p['Vp'] / (2 * p['F'])
    B = (
        p['alpha0'] * (1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) *
        (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2
        + (p['alpha1']*p['gca']*p['V0']*p['Vp']*p['Ca_E']*exp(-p['V0']*p['F']/(p['R']*p['T']))/(2*p['F']))
    )
    D =p['alpha0']*(1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2 * p['Kp']**4

    def f_eq(c):
        # c is array-like, fsolve passes it as [c]
        Ca_in_SMC0 = c[0]
        val = (
            A * Ca_in_SMC0**5
            + B * Ca_in_SMC0**4
            + D
        )
        return val
    print (f"Parameters: A={A}, B={B}, D={D}")
    # Compute a lower bound estimate for Ca_in_SMC0
    C0 = max(vari_init_vals['Ca_in_SMC0']+1, 4*B/(5*A))  # the lower bound for Ca_in_SMC0


    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
    try:
        # Solve Eq1 for Ca_in_SMC0 using the improved initial guess
        Ca_in_SMC0_val = fsolve(f_eq, np.array([C0]))[0]
        print(f"Ca_in_SMC0 is {Ca_in_SMC0_val} ")
    except OptimizeWarning:
        print("fsolve did not converge for this parameter set.")
        return None, None, None  # skip this run
    # Define term1 and term2 here for g_eq
    Ca_in_SMC0 = Ca_in_SMC0_val
    term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) / (p['k_ryr2']**3 + Ca_in_SMC0**3))
    term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
    E = (term1 + p['Jer'])
    H = - (term1 + p['Jer']*Ca_in_SMC0 + term2)
    K = p['Jer']*(p['k_ryr3']**4)
    M = - (p['k_ryr3']**4)*(p['Jer'] + term2)

    # 2) Solve Eq2 for Ca_SR
    def g_eq(s):
        # s is array-like, fsolve passes it as [s]
        Ca_SR = s[0]
        val = (E*Ca_SR**5 + H*Ca_SR**4 + K*Ca_SR + M)
        return val
        print (f"Parameters for g_eq: E={E}, H={H}, K={K}, M={M}")

    # If applies, find all roots of g_eq ---
    s_vals = np.linspace(0, 2, 1000)  # search range, can adjust upper limit if needed
    g_vals = np.array([g_eq([s]) for s in s_vals])
    crossings = np.where(np.diff(np.sign(g_vals)))[0]

    roots = []
    for idx in crossings:
        root = fsolve(g_eq, [s_vals[idx]])[0]
        if not any(np.isclose(root, r, atol=1e-6) for r in roots):
            roots.append(root)

    print(f"Ca_SR has {len(roots)} solution(s): " + ", ".join([f"{r:.4f}" for r in roots]))
    # ---------------------------------------------
    Ca_SR0_val = fsolve(g_eq, np.array([vari_init_vals['Ca_SR0']]))[0]

    # 3) Compute y
    y0_val = (p['l_4'] * Ca_in_SMC0_val) / (p['l_4'] * Ca_in_SMC0_val + p['l_m4'])
    print(f"y0 is {y0_val} ")

    return float(Ca_in_SMC0_val), float(Ca_SR0_val), float(y0_val)
