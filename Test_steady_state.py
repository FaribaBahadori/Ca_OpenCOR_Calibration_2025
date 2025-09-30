import csv # for reading CSV files
import sys
import os
import sys
#  import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # renders plots to files without opening a window
import datetime  # add this at the top with other imports

from Functions.plot_eqs import plot_eqs
# Importing helper modules
from Functions.steady_state_solver import steady_state_smc
print("Python executable:", sys.executable)

os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())

print("Steady state script started running!")

def load_parameter_data(csv_par_file, n_init=3):
    """
    Read a CSV file with:
    - first row = column names
    - each subsequent row = values
    - first `n_init` columns = initial guesses
    - remaining columns = parameters

    Returns a list of tuples: (params, init_vals)
    """
    data = []
    with open(csv_par_file, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)  # read first row (names of all columns)
        headers = [h.strip() for h in headers]   # remove spaces from column names
        for row in reader:
            if not row:
                continue
            values = [float(x) for x in row]

            # split into initial guesses and parameters
            init_vals = dict(zip(headers[:n_init], values[:n_init]))
            params = dict(zip(headers[n_init:], values[n_init:]))

            data.append((params, init_vals))
    return data
# file path
csv_par_file = r"C:\\Fariba_2025\\Ca_OpenCOR_Calibration_2025\\Test\\Par_data_Steady_Test_1.csv"

# load all parameter sets
parameter_sets = load_parameter_data(csv_par_file, n_init=3)

# Setting the output directory
output_file_path = "Test/Test_Output"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# --- Prepare folder for plots ---
plot_folder = os.path.join(os.path.dirname(output_file_path), "run_steady")
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# CSV path for saving results
output_folder = r"C:\\Fariba_2025\\Ca_OpenCOR_Calibration_2025\\Test\\Test_Output"
os.makedirs(output_folder, exist_ok=True)  # create folder if not exists
output_file = os.path.join(output_folder, "steady_state_results.csv")

# Prepare a list to store results
results_list = []

# Loop over parameter sets
for i, (params, init_vals) in enumerate(parameter_sets, start=1):
    print(f"\n--- Running steady state for row {i} ---")
    Ca_in_SMC0, Ca_SR0, y0, Eq1_B0, Eq1_B1, Eq1_B2, Eq1_B3, Eq1_B4, Eq1_e1, Eq1_e2, Eq2_E, Eq2_H, Eq2_K, Eq2_M, term1, term2 = steady_state_smc(params, init_vals, return_extra=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    result_row = {
        "Timestamp": timestamp, 
        "Row": i,
        "SS_Ca_in_SMC0": Ca_in_SMC0, "SS_Ca_SR0": Ca_SR0,"SS_y0": y0,
        "Eq1_B0": Eq1_B0, "Eq1_B1": Eq1_B1, "Eq1_B2": Eq1_B2, "Eq1_B3": Eq1_B3, "Eq1_B4": Eq1_B4, "Eq1_e1": Eq1_e1, "Eq1_e2": Eq1_e2, 
        "Eq2_E": Eq2_E, "Eq2_H": Eq2_H, "Eq2_K": Eq2_K, "Eq2_M": Eq2_M,
        "term1": term1, "term2": term2}

    # Append initial guesses next
    result_row.update(init_vals)

    # Append parameters next
    result_row.update(params)
    results_list.append(result_row)

    plot_eqs(params, init_vals, plot_folder, row=i)

    '''# --- Call plot_eqs for last parameter set ---
    params_dict = params     
    init_vals_dict = init_vals  
    # Inside the loop
    plot_eqs(params_dict, init_vals_dict, plot_folder, row=i)'''


# Convert current results to DataFrame
df_new = pd.DataFrame(results_list)

# Convert current results to DataFrame
df_new = pd.DataFrame(results_list)

if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)

    # Find empty rows (all NaN in 'Row' column)
    empty_idx = df_existing[df_existing['Row'].isna()].index.tolist()

    # Fill empty rows with new data
    for i, idx in enumerate(empty_idx):
        if i < len(df_new):
            df_existing.loc[idx] = df_new.iloc[i]

    # Append remaining new rows if any
    if len(df_new) > len(empty_idx):
        df_existing = pd.concat([df_existing, df_new.iloc[len(empty_idx):]], ignore_index=True)

    df_results = df_existing
else:
    df_results = df_new

# Save back to CSV
df_results.to_csv(output_file, index=False)
print(f"\nAll steady-state results saved to:\n{output_file}")

