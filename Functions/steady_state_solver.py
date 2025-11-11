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
    clb = 0.001  # Physiological lower bound for Ca_in_SMC0
    cub = 100.001  # Physiological upper bound for Ca_in_SMC0
    # 1) Solve Eq1 for Ca_in_SMC0
    #  Coefficients for Eq1
    
    B0 = p['Ca_E']*exp(-2*p['V0']*p['F']/(p['R']*p['T']))
    B1 = (p['alpha1'] * p['gca'] * p['V0']) / (2 * p['F']*(1 - exp(-2*p['V0']*p['F']/(p['R']*p['T']))) * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2)
    B2 = p['Vp']-p['alpha0']-B0*B1
    B3 = B1 *p['Kp']**4
    B4 = -(p['alpha0']+ B0*B1)*p['Kp']**4
    e1 = 1 - exp(-p['V0']*p['F']/(p['R']*p['T']))
    e2 = (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2

    def f_eq(c):
        Ca_in_SMC0 = c[0]
        val = (
            p['alpha0']
            - (p['alpha1'] * p['V0'] * p['gca'] / (2 * p['F']))
            * ((Ca_in_SMC0 - p['Ca_E'] * np.exp(-2 * p['V0'] * p['F'] / (p['R'] * p['T'])))
            / ((1 + np.exp(-(p['V0'] - p['Vm']) / p['km']))**2
                * (1 - np.exp(-2 * p['V0'] * p['F'] / (p['R'] * p['T'])))))
            - (p['Vp'] * Ca_in_SMC0**4) / (p['Kp']**4 + Ca_in_SMC0**4)
        )
        return val


    ## c_vals = np.linspace(int(-100*(vari_init_vals['Ca_in_SMC0']+1)),int(100*(vari_init_vals['Ca_in_SMC0']+1)),400)

    print (f"Coefficients in Eq1: B1={B1}, B2={B2}, B3={B3}, B4={B4}")
    C0 = vari_init_vals['Ca_in_SMC0']
     
    Ca_in_SMC0_val = float("nan")  # default, ensures variable always exists
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            c_scan = np.linspace(clb, cub, 100000)  # scan beyond local min
            f_vals = [f_eq([c]) for c in c_scan]

            # Find where function crosses zero
            crossings1 = np.where(np.diff(np.sign(f_vals)))[0]
            num_roots1 = len(np.where(np.diff(np.sign(f_vals)))[0])
            print(f" Ca_in_SMC0 has {num_roots1} root(s) between {clb} and {cub}.")
            
            if len(crossings1) > 0:
                idx1 = crossings1[0]
                Ca_in_SMC0_val = fsolve(f_eq, [c_scan[idx1]])[0]
                print(f"the root between {c_scan[idx1]} and {c_scan[idx1+1]} is:", Ca_in_SMC0_val)
            else:
                print(f"No root found between {clb} and  {cub}.")
                
        except OptimizeWarning:
            print("fsolve did not converge for this parameter set.")
            return None, None, None  # skip this run
        except Exception as e:
            print(f"fsolve failed with error: {e}")
            return None, None, None  # skip this run

    Ca_in_SMC0 = Ca_in_SMC0_val

    slb = 0.01  # physiological lower bound for Ca_SR
    sub = 100.01  # physiological upper bound for Ca_SR

    # --- 2) Solve Eq2 for Ca_SR ---
    term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) / (p['k_ryr2']**3 + Ca_in_SMC0**3))
    term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
    E = (term1 + p['Jer'])
    H = - (term1 + p['Jer']*Ca_in_SMC0 + term2)
    K = p['Jer']*(p['k_ryr3']**4)
    M = - (p['k_ryr3']**4)*(p['Jer'] + term2)
    print (f"Coefficients in Eq2: E={E}, H={H}, K={K}, M={M}")
    def g_eq(s):
        Ca_SR0 = s[0]
        val = (
            (p['k_RyR']
            * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3)
                / (p['k_ryr2']**3 + Ca_in_SMC0**3))
            * (Ca_SR0**4 / (p['k_ryr3']**4 + Ca_SR0**4))
            + p['Jer'])
            * (Ca_SR0 - Ca_in_SMC0)
            - (p['Ve'] * Ca_in_SMC0**2) / (p['Ke'] + Ca_in_SMC0**2)
        )
        return val


    # If applies, find all roots of g_eq ---
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning) # treat as error
        try:
            s_vals = np.linspace(slb, sub, 100000)  # search range, can adjust upper limit if needed
            g_vals = np.array([g_eq([s]) for s in s_vals])
            crossings = np.where(np.diff(np.sign(g_vals)))[0]
            num_roots = len(np.where(np.diff(np.sign(g_vals)))[0])
            print(f" Ca_SR has {num_roots} root(s) between {slb} and {sub}.")
            roots = []
            for idx in crossings:
                root = fsolve(g_eq, [s_vals[idx]])[0]
                #avoid duplicates s and check physiological range
                if not any(np.isclose(root, r, atol=1e-6) for r in roots):
                    if slb < root < sub:  # physiological range check
                        roots.append(root)
                       
            if roots:
                print(f"Between {slb} and {sub}, Ca_SR has {len(roots)} solution(s): " + "," .join([f"{r:.4f}" for r in roots]))
                # pick a root close to initial guess
                Ca_SR0_val = fsolve (g_eq, [vari_init_vals['Ca_SR0']])[0]
                print( f"selected {Ca_SR0_val}")
            else:
                print(f"No root found for Ca_SR between {slb} and {sub}.")
                Ca_SR0_val= None
        
        except OptimizeWarning:
            print("fsolve did not converge for Ca_SR with this parameter set.")
            Ca_SR0_val= None
            roots = None
        except Exception as e:
            print(f"fsolve for Ca_SR failed with error: {e}")
            Ca_SR0_val= None
            roots = None

    # 3) Compute y
    y0_val = (p['l_4'] * Ca_in_SMC0_val) / (p['l_4'] * Ca_in_SMC0_val + p['l_m4'])
    print(f"y0 is {y0_val} ")
    # Check if valid roots were found
    if Ca_in_SMC0_val is None:
        print(f"Skipping run: No valid root found for Ca_in_SMC0 between {clb} and {cub}.")
        if return_extra:
            return (None, None, None, B0, B1, B2, B3, B4, e1, e2, E, H, K, M, term1, term2)
        else:
            return None, None, None

    if Ca_SR0_val is None:
        print(f"Skipping run: No valid root found for Ca_SR between {slb} and {sub}.")
        if return_extra:
            return (None, None, None, B0, B1, B2, B3, B4, e1, e2, E, H, K, M, term1, term2)
        else:
            return None, None, None
    
    if return_extra:
        return float(Ca_in_SMC0_val), float(Ca_SR0_val), float(y0_val), B0, B1, B2, B3, B4, e1, e2, E, H, K, M, term1, term2
    else:
        return float(Ca_in_SMC0_val), float(Ca_SR0_val), float(y0_val)

