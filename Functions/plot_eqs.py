import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Functions.steady_state_solver import steady_state_smc

from math import exp

def plot_eqs(params, vari_init_vals, run_dir, row=None):
    """
    Generate 3 plots (f_eq, g_eq, y_func) for the optimized solution.

    params        : dict of parameters (from optimizer or CSV)
    vari_init_vals: dict of initial variable values from model
    run_dir       : folder path to save the plots
    """
    os.makedirs(run_dir, exist_ok=True)

    # === Solve steady state values ===
    ## Ca_in_SMC0_val, Ca_SR0_val, y0_val = steady_state_smc(params, vari_init_vals, return_extra=False)
    Ca_in_SMC0_val, Ca_SR0_val, y0_val = steady_state_smc(params, vari_init_vals, return_extra=False)
    if Ca_in_SMC0_val is None:
        raise RuntimeError("Steady-state solver did not converge for given parameters.")
    
     # Extract parameters
    p = {k.split('/')[-1]: v for k, v in params.items()}

    # -------------------------------
    # Plot 1: f_eq(Ca_in_SMC0)
    # -------------------------------
    A = -p['alpha1'] * p['gca'] * p['V0'] * p['Vp'] / (2 * p['F'])
    B = (p['alpha0'] * (1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) *
            (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2
            + (p['alpha1']*p['gca']*p['V0']*p['Vp']*p['Ca_E']
               * exp(-p['V0']*p['F']/(p['R']*p['T']))/(2*p['F'])))
    D = (p['alpha0']*(1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2 * p['Kp']**4)
    c00 =(-4*B/(5*A))
    def f_eq(c):
        return (A * c**5 + B * c**4 + D)
    
    c_vals = np.linspace(int(-0.5*Ca_in_SMC0_val),int(1.1*Ca_in_SMC0_val),400)
    ## c_vals = np.linspace(min(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), max(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), 400)

    f_vals = [f_eq(c) for c in c_vals]

    plt.figure()
    plt.plot(c_vals, f_vals, label="f(c)")
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(Ca_in_SMC0_val, color="r", linestyle=":", label=f"Ca_Root={Ca_in_SMC0_val:.4f}, f({Ca_in_SMC0_val:.4f})={f_eq(Ca_in_SMC0_val):.4f}, c_min={c00:.4f}")
    plt.title(f"f(C)={A:.2f}*C^5 + {B:.2f}*C^4 + {D:.2f}")
    plt.xlabel("Ca_in_SMC")
    plt.ylabel("f(c)")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if row is not None:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot1_f_eq_row{row}.png"))
    else:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot1_f_eq.png"))
    plt.close()

    # -------------------------------
    # Plot 2: g_eq(Ca_SR)
    # -------------------------------
    Ca_in_SMC0 = Ca_in_SMC0_val
    term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) /
                            (p['k_ryr2']**3 + Ca_in_SMC0**3))
    term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
    E = (term1 + p['Jer'])
    H = - (term1 + p['Jer']*Ca_in_SMC0 + term2)
    K = p['Jer']*(p['k_ryr3']**4)
    M = - (p['k_ryr3']**4)*(p['Jer'] + term2)
    def g_eq(s):
        return ((term1 + p['Jer'])*s**5
                - (term1 + p['Jer']*Ca_in_SMC0 + term2)*s**4
                + p['Jer']*(p['k_ryr3']**4)*s
                - (p['k_ryr3']**4)*(p['Jer'] + term2))
    s_vals = np.linspace(min(int(-2*Ca_SR0_val),int(2*Ca_SR0_val)), max(int(-2*Ca_SR0_val),int(2*Ca_SR0_val)), 400)
    ## s_vals = np.linspace(min(-1*Ca_SR0_val, 2*Ca_SR0_val), max(-1*Ca_SR0_val, 2*Ca_SR0_val), 400)
    g_vals = [g_eq(s) for s in s_vals]

    plt.figure()
    plt.plot(s_vals, g_vals, label="g(s)")
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(Ca_SR0_val, color="r", linestyle=":", label=f"Root={Ca_SR0_val:.4f}, g({Ca_SR0_val:.4f})={g_eq(Ca_SR0_val):.4f}")
    plt.title(f"g(s)={E:.2f}*s^5 + {H:.2f}*s^4 + {K:.2f}*s + {M:.2f}")
    plt.xlabel("Ca_SR")
    plt.ylabel("g(s)")
    plt.legend()
    if row is not None:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot2_g_eq_row{row}.png"))
    else:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot2_g_eq.png"))
    plt.close()

    # -------------------------------
    # Plot 3: y(c0)
    # -------------------------------
    def y_func(c0):
        return (p['l_4'] * c0) / (p['l_4'] * c0 + p['l_m4'])
    c_vals = np.linspace(-y0_val, 2*y0_val, 400)
    ## c_vals = np.linspace(min(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), max(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), 400)
    y_vals = [y_func(c) for c in c_vals]

    plt.figure()
    plt.plot(c_vals, y_vals, label="y(c0)")
    plt.axvline(Ca_in_SMC0_val, color="r", linestyle=":",
                label=f"c0={Ca_in_SMC0_val:.4f}, y={y0_val:.4f}, y(c={Ca_in_SMC0_val:.4f})={y_func(Ca_in_SMC0_val):.4f}")
    plt.title(f"y(c)={p['l_4']:.2f}*c)/({p['l_4']:.2f}*c + {p['l_m4']:.2f}")
    plt.xlabel("Ca_in_SMC0")
    plt.ylabel("y")
    plt.legend()
    if row is not None:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot3_y_func_row{row}.png"))
    else:
        plt.savefig(os.path.join(run_dir, f"{timestamp}_plot3_y_func.png"))
    plt.close()

    print(f"Plots saved in {run_dir}")
