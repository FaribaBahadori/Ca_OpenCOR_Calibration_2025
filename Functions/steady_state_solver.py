import numpy as np
from scipy.optimize import fsolve
from math import exp
import warnings
from scipy.optimize import OptimizeWarning

def steady_state_smc(params, vari_init_vals, return_extra=False):
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
    B1 = p['alpha0'] * (1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2   
    B2 = (p['alpha1']*p['gca']*p['V0']*p['Vp']*p['Ca_E']*exp(-p['V0']*p['F']/(p['R']*p['T']))/(2*p['F']))
    e1 = 1 - exp(-p['V0']*p['F']/(p['R']*p['T']))
    e2 = (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2

    D =p['alpha0']*(1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2 * p['Kp']**4
    c00 =(-4*B/(5*A))
    def f_eq(c):
        # c is array-like, fsolve passes it as [c]
        Ca_in_SMC = c[0]
        val = (
            A * Ca_in_SMC**5
            + B * Ca_in_SMC**4
            + D
    
        )
        return val
    print (f"Coefficients in Eq1: A={A}, B={B}, D={D}")
 #   C0 = max(vari_init_vals['Ca_in_SMC0']+1, c00)  # the lower bound for Ca_in_SMC0
  #  C0 = vari_init_vals['Ca_in_SMC0']
    c00
    Ca_in_SMC0_val = float("nan")  # default, ensures variable always exists
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            c_scan = np.linspace(c00, c00*10, 1000)  # scan beyond local min
            f_vals = [f_eq([c]) for c in c_scan]

            # Find where function crosses zero
            crossings1 = np.where(np.diff(np.sign(f_vals)))[0]
            if len(crossings1) > 0:
                idx = crossings1[0]
                Ca_in_SMC0_val = fsolve(f_eq, [c_scan[idx]])[0]
                print(f"Correct root beyond c_min({c00}):", Ca_in_SMC0_val)
            else:
                print(f"No root found larger than c_min({c00})")

            print(f"Ca_in_SMC0 is {Ca_in_SMC0_val} ")
        except OptimizeWarning:
            print("fsolve did not converge for this parameter set.")
            return None, None, None  # skip this run
        except Exception as e:
            print(f"fsolve failed with error: {e}")
            return None, None, None  # skip this run

    Ca_in_SMC0 = Ca_in_SMC0_val

    # --- 2) Solve Eq2 for Ca_SR ---
    term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) / (p['k_ryr2']**3 + Ca_in_SMC0**3))
    term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
    E = (term1 + p['Jer'])
    H = - (term1 + p['Jer']*Ca_in_SMC0 + term2)
    K = p['Jer']*(p['k_ryr3']**4)
    M = - (p['k_ryr3']**4)*(p['Jer'] + term2)
 
    def g_eq(s):
        # s is array-like, fsolve passes it as [s]
        Ca_SR = s[0]
        val = (E*Ca_SR**5 + H*Ca_SR**4 + K*Ca_SR + M)
        return val

    # If applies, find all roots of g_eq ---
    s_vals = np.linspace(0, int(vari_init_vals['Ca_SR0']), 1000)  # search range, can adjust upper limit if needed
    g_vals = np.array([g_eq([s]) for s in s_vals])
    crossings = np.where(np.diff(np.sign(g_vals)))[0]

    roots = []
    for idx in crossings:
        root = fsolve(g_eq, [s_vals[idx]])[0]
        if not any(np.isclose(root, r, atol=1e-6) for r in roots):
            roots.append(root)

    print(f"Ca_SR has {len(roots)} solution(s): " + ", ".join([f"{r:.4f}" for r in roots]))
    Ca_SR0_val = fsolve(g_eq, np.array([vari_init_vals['Ca_SR0']]))[0]

    # 3) Compute y
    y0_val = (p['l_4'] * Ca_in_SMC0_val) / (p['l_4'] * Ca_in_SMC0_val + p['l_m4'])
    print(f"y0 is {y0_val} ")

    # --- Decide return ---
    if return_extra:
        return float(Ca_in_SMC0_val), float(Ca_SR0_val), float(y0_val), c00, A, B, B1, B2, e1, e2, D, E, H, K, M, term1, term2
    else:
        return float(Ca_in_SMC0_val), float(Ca_SR0_val), float(y0_val)
