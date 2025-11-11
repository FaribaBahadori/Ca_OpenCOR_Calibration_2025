import numpy as np

def ODE_La_Ls(t, y, lc, Amp, Am, ls0, kx1, kx2, beta, lopt):
    """
    Equivalent to MATLAB's ODE_La_Ls.m
    la = y[0], ls = y[1]
    """
    dy = np.zeros(2)

    # Parameters (same as in MATLAB)
    alpha_s = 4.5
    vx = 5000
    fAMp = 0.0013
    fAM = 0.0855
    mu_s = 0.00001
    ks = 0.2
    epsilon = 1e-10
    Amp = max(Amp, 1e-12)
    Am = max(Am, 1e-12)

    # Differential equations
    dy[0] = (kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1] - fAMp * Amp * vx) / (fAM * Am + fAMp * Amp + epsilon)
    dy[1] = ((kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1])
             * np.exp(-beta * ((y[0] - lopt) / lopt) ** 2)
             - ks * (np.exp(alpha_s * (y[1] - ls0) / ls0) - 1)) / mu_s

    return dy
