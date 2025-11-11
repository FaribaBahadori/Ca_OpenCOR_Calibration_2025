from datetime import datetime

import os, time, numpy as np, pandas as pd, csv, matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots only
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Import ODE system and model interface
#from Eq14_Force_Model import Eq14
from ODE_La_Ls import ODE_La_Ls
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
#---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Run the CellML model and extract AM, AMp, and Time
# ---------------------------------------------------------------------
# Paths and simulation setup
cellml_path = r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025/Models/Main_Coupled_SMC_Model.cellml"
# --- Run CellML model through SimulationHelper (from opencor_helper.py) ---
sim = SimulationHelper(cellml_path=cellml_path, dt=0.1, sim_time=2000, pre_time=0)
# Optional: specify parameters if you have any to vary
param_names = ['Environment/tau']

# Run the simulation (only once)
success = sim.run()
if not success:
    print("Simulation failed to converge.")
    sys.exit()

# Extract outputs, AM (state) and AMp (algebraic), from CellML model
results = sim.get_results([['AM/AM'], ['AMp/AMp']])
AM = results[0][0]   # AM time series
AMP = results[1][0]  # AMp time series
Time = sim.tSim      # time vector
# ---------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------

# Save to CSV
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_folder = os.path.join(os.path.dirname(__file__), "output_artery")
os.makedirs(output_folder, exist_ok=True)

csv_file = os.path.join(output_folder, f"AM_AMp_outputs_{timestamp}.csv")
df = pd.DataFrame({'time': Time, 'AM': AM, 'AMP': AMP})
df.to_csv(csv_file, index=False)

# Read CSV for the rest of your code
data = pd.read_csv(csv_file).values

# Global parameters
lc = None
AMp = None
AM = None
ls0 = None
kx1 = None
kx2 = None
beta = None
lopt = None

# ---------------------------------------------------------------------
# Parameters for Eq14 and ODE system
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Defining parameters
# ---------------------------------------------------------------------
N = 5000
kpp = 0.1
h = 15
n = 6
ls0 = 30
l0 = 40
kx1 = 12.5
kx2 = 8.8
beta = 7.5
lopt = 150
alpha_pp = 0.0002
PPi = 0.0021

# Initial conditions
la000 = 103.8708634  #89.60
ls000 = 30

la00 = la000 
ls00 = ls000 

# Redundant parameters
#alpha_s = 4.5
#vx = 5000
#fAMp = 0.0013
#fAM = 0.0855
#mu_s = 0.00001
#ks = 0.2
#epsilon = 1e-10

A = n * h
# ---------------------------------------------------------------------
# Extracting columns
# ---------------------------------------------------------------------
Time = data[:, 0]
AM = data[:, 1]
AMP = data[:, 2]

# Initialize arrays
LC = np.zeros_like(Time)
LA = np.zeros_like(Time)
LS = np.zeros_like(Time)

# set initial (t=0) values
LC[0] = la00 + ls00
LA[0] = la00
LS[0] = ls00

# ---------------------------------------------------------------------
# Time loop to solve Eq14 and ODE system
# ---------------------------------------------------------------------
for j in range(1, len(Time)-1):
#test1

    AM_val = float(AM[j])
    AMP_val = float(AMP[j])
    lc00 = la00 + ls00
    
    def Eq14(lc_val):
        return ((N * kpp) / A) * (np.exp(alpha_pp * (lc_val - l0) / l0) - 1) \
             + (N / A) * (kx1 * AMP[j] + kx2 * AM[j]) * (lc_val - la00 - ls00) \
             * np.exp(-beta * ((la00 - lopt) / lopt) ** 2) \
             - (PPi / 2) * (((n * lc_val) / (np.pi * h)) - 1)

    ##lc00 = la00 + ls00  # initial guess
    lc_solution = fsolve(Eq14, lc00)[0]

    # safety check in case fsolve fails
    if not np.isfinite(lc_solution):
        print(f"Warning: fsolve returned non-finite value at step {j}, using previous lc")
        lc_solution = LC[j-1] if j > 0 else lc00

        # Update globals
    ##    lc = lc_solution
    ##    Amp = AMP[j]
    ##    Am = AM[j]
        # Solve ODE system for la and ls
    sol = solve_ivp(
        lambda t, y: ODE_La_Ls(t, y, lc_solution, AMP_val, AM_val, ls0, kx1, kx2, beta, lopt),
        [Time[j], Time[j+1]],
        np.array([la00, ls00], dtype=float),
        method='LSODA',
        rtol=1e-7,
        atol=1e-9
    )

    la00 = sol.y[0, -1]
    ls00 = sol.y[1, -1]

    LA[j] = la00
    LS[j] = ls00
    LC[j] = lc_solution
# Copy the last computed values to the final time point
LC[-1] = LC[-2]
LA[-1] = LA[-2]
LS[-1] = LS[-2]
# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(Time, LC, label='LC', linewidth=3, color='blue')
plt.plot(Time, LA, label='LA', linewidth=3, color='black')
plt.plot(Time, LS, label='LS', linewidth=3, color='green')

LX = LC - LA - LS
plt.plot(Time, LX, label='LX', linewidth=3, color='yellow')

Ri = 0.5 * (((n * LC) / np.pi) - h)
plt.plot(Time, Ri, label='Ri', linewidth=3, color='red')

plt.xlabel('Time', fontweight='bold', fontsize=14)
plt.ylabel('LC and Ri', fontweight='bold', fontsize=14)
plt.title(f'LC and Ri Over Time (PPi = {PPi})', fontweight='bold', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()

# ---------------------------------------------------------------------
# Save plot to the "output" folder (with date and time in the file name)
# ---------------------------------------------------------------------

# Create output folder (if it doesn't exist)

# Define the output file name (includes PPi and timestamp)
output_file = os.path.join(output_folder, f"Single_Artery_Plot_PPi_{PPi}_{timestamp}.png")

# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {output_file}")

# Optionally display the plot as well
##plt.show()
# ---------------------------------------------------------------------
# Save computed results (LC, LA, LS, LX, Ri) to CSV
# ---------------------------------------------------------------------

csv_file_plot_data = os.path.join(output_folder, f"Single_Artery_Data_{timestamp}.csv")

df_plot = pd.DataFrame({
    'Time': Time,
    'LC': LC,
    'LA': LA,
    'LS': LS,
    'LX': LC - LA - LS,
    'Ri': 0.5 * (((n * LC) / np.pi) - h)
})

df_plot.to_csv(csv_file_plot_data, index=False)
print(f"Plot data saved to: {csv_file_plot_data}")
# ---------------------------------------------------------------------
# Save summary data (final values or constants) to a separate CSV
# ---------------------------------------------------------------------
summary_data = {
    'Parameter': ['PPi', 'N', 'kpp', 'h', 'n', 'l0', 'ls0', 'la000', 'ls000', 'kx1', 'kx2', 'beta', 'lopt'],
    'Value': [PPi, N, kpp, h, n, l0, ls0, la000, ls000, kx1, kx2, beta, lopt]
}

csv_file_summary = os.path.join(output_folder, f"Eq14_Summary_{timestamp}.csv")
pd.DataFrame(summary_data).to_csv(csv_file_summary, index=False)

print(f"Summary data saved to: {csv_file_summary}")
