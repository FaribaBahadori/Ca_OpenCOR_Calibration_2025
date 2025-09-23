import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Functions.steady_state_solver import steady_state_smc

from math import exp

def plot_eqs(params, vari_init_vals, run_dir):
    """
    Generate 3 plots (f_eq, g_eq, y_func) for the optimized solution.

    params        : dict of parameters (from optimizer or CSV)
    vari_init_vals: dict of initial variable values from model
    run_dir       : folder path to save the plots
    """
    os.makedirs(run_dir, exist_ok=True)

    # === Solve steady state values ===
    Ca_in_SMC0_val, Ca_SR0_val, y0_val = steady_state_smc(params, vari_init_vals)
    if Ca_in_SMC0_val is None:
        raise RuntimeError("Steady-state solver did not converge for given parameters.")
    
     # Extract parameters
    p = {k.split('/')[-1]: v for k, v in params.items()}

    # -------------------------------
    # Plot 1: f_eq(Ca_in_SMC0)
    # -------------------------------
    
    def f_eq(c):
        A = -p['alpha1'] * p['gca'] * p['V0'] * p['Vp'] / (2 * p['F'])
        B = (
            p['alpha0'] * (1 - exp(-p['V0']*p['F']/(p['R']*p['T']))) *
            (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2
            + (p['alpha1']*p['gca']*p['V0']*p['Vp']*p['Ca_E']
               * exp(-p['V0']*p['F']/(p['R']*p['T']))/(2*p['F']))
        )
        return (A * c**5
                + B * c**4
                + p['alpha0']*(1 - exp(-p['V0']*p['F']/(p['R']*p['T'])))
                * (1 + exp(-(p['V0'] - p['Vm'])/p['km']))**2
                * p['Kp']**4)
    c_vals = np.linspace(-10000,20000, 400)
    ## c_vals = np.linspace(min(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), max(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), 400)

    f_vals = [f_eq(c) for c in c_vals]

    plt.figure()
    plt.plot(c_vals, f_vals, label="f(c)")
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(Ca_in_SMC0_val, color="r", linestyle=":", label=f"Root={Ca_in_SMC0_val:.4f}")
    plt.title("f_eq(Ca_in_SMC0)")
    plt.xlabel("Ca_in_SMC0")
    plt.ylabel("f(c)")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(run_dir, f"plot1_f_eq_{timestamp}.png"))
    plt.close()

    # -------------------------------
    # Plot 2: g_eq(Ca_SR)
    # -------------------------------
    def g_eq(s):
        Ca_in_SMC0 = Ca_in_SMC0_val
        term1 = p['k_RyR'] * (p['k_ryr0'] + (p['k_ryr1'] * Ca_in_SMC0**3) /
                              (p['k_ryr2']**3 + Ca_in_SMC0**3))
        term2 = p['Ve'] * Ca_in_SMC0**2 / (p['Ke']**2 + Ca_in_SMC0**2)
        return ((term1 + p['Jer'])*s**5
                - (term1 + p['Jer']*Ca_in_SMC0 + term2)*s**4
                + p['Jer']*(p['k_ryr3']**4)*s
                - (p['k_ryr3']**4)*(p['Jer'] + term2))
    s_vals = np.linspace(-500,500, 400)
    ## s_vals = np.linspace(min(-1*Ca_SR0_val, 2*Ca_SR0_val), max(-1*Ca_SR0_val, 2*Ca_SR0_val), 400)
    g_vals = [g_eq(s) for s in s_vals]

    plt.figure()
    plt.plot(s_vals, g_vals, label="g(s)")
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(Ca_SR0_val, color="r", linestyle=":", label=f"Root={Ca_SR0_val:.4f}")
    plt.title("g_eq(Ca_SR)")
    plt.xlabel("Ca_SR")
    plt.ylabel("g(s)")
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"plot2_g_eq_{timestamp}.png"))
    plt.close()

    # -------------------------------
    # Plot 3: y(c0)
    # -------------------------------
    def y_func(c0):
        return (p['l_4'] * c0) / (p['l_4'] * c0 + p['l_m4'])
    c_vals = np.linspace(-10,10, 400)
    ## c_vals = np.linspace(min(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), max(-1*Ca_in_SMC0_val, 2*Ca_in_SMC0_val), 400)
    y_vals = [y_func(c) for c in c_vals]

    plt.figure()
    plt.plot(c_vals, y_vals, label="y(c0)")
    plt.axvline(Ca_in_SMC0_val, color="r", linestyle=":",
                label=f"c0={Ca_in_SMC0_val:.4f}, y={y0_val:.4f}")
    plt.title("y(c0)")
    plt.xlabel("Ca_in_SMC0")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"plot3_y_func_{timestamp}.png"))
    plt.close()

    print(f"Plots saved in {run_dir}")
