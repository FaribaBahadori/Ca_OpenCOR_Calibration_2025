import numpy as np
from scipy.optimize import fsolve
from math import exp

def steady_state_smc(params, vari_init_vals):
    """
    Solve the steady-state Ca_in_SMC0, Ca_SR, and y using fsolve.
    
    params    : dictionary containing all numeric parameter values
    vari_init_vals : dictionary with initial guesses for Ca_in_SMC and Ca_SR
    Returns   : Ca_in_SMC, Ca_SR, y (floats)
    """

    # Extract parameters
    p = {k.split('/')[-1]: v for k, v in params.items()}

    # 1) Solve Eq1 for Ca_in_SMC0
    def f_eq(c):
        # c is array-like, fsolve passes it as [c]
        Ca_in_SMC0 = c[0]
        val = (
            (-p['alpha1'] * p['gca'] * p['V0'] * p['Vp'] / (2 * p['F'])) * Ca_in_SMC0**5
            + (
                p['alpha0'] * (1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) *
                (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2
                + (p['alpha1']*p['gca']*p['V0']*p['Vp']*p['Ca_E']*exp(-p['V0']*p['F']/(p['R']*p['T']))/(2*p['F']))
            ) * Ca_in_SMC0**4
            + p['alpha0']*(1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) *
              (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2 * p['Kp']**4
        )
        return val

    Ca_in_SMC0_val = fsolve(f_eq, np.array([vari_init_vals['Ca_in_SMC0']]))[0]

    # 2) Solve Eq2 for Ca_SR
    def g_eq(s):
        # s is array-like, fsolve passes it as [s]
        Ca_SR = s[0]
        Ca_in_SMC0 = Ca_in_SMC0_val
        term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) / (p['k_ryr2']**3 + Ca_in_SMC0**3))
        term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
        val = (
            (term1 + p['Jer'])*Ca_SR**5
            - (term1 + p['Jer']*Ca_in_SMC0 + term2)*Ca_SR**4
            + p['Jer']*(p['k_ryr3']**4)*Ca_SR
            - (p['k_ryr3']**4)*(p['Jer'] + term2)
        )
        return val

    Ca_SR_val = fsolve(g_eq, np.array([vari_init_vals['Ca_SR0']]))[0]

    # 3) Compute y
    y_val = (p['l_4'] * Ca_in_SMC0_val) / (p['l_4'] * Ca_in_SMC0_val + p['l_m4'])

    return float(Ca_in_SMC0_val), float(Ca_SR_val), float(y_val)
