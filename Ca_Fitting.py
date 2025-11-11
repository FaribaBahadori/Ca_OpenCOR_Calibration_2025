import datetime, os, sys, pandas as pd
os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())
from Functions.init_set_test import init_set_test
##import Calibration as calib  # Calibration.py script

# 1. Create initial parameter sets CSV
csv_path = init_set_test()

# 2. Load all parameter sets
param_sets = pd.read_csv(csv_path)

# 3. Prepare output directory
output_dir = r"outputs/Ca_Fitting/run222"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M") # session-level timestamp
output_base_path = os.path.join("outputs", "Ca_Fitting", f"session_{timestamp}_SS_test_Ke")
# 4. Loop over each row
for i, row in param_sets.iterrows():
    print(f"\n=== Running optimisation {i+1}/{len(param_sets)} ===")

    # Pass the row values to the calibration script
    param_init_vals = row.to_list()  # this will override default in my script
    # Now run the calibration script (it will use this row as initial params)
    # exec the script in its own namespace
    with open("No_Calibration.py") as f:
        code = f.read()
    exec(code, {
    "__file__": os.path.abspath("No_Calibration.py"),
    "output_base_path": output_base_path,
    "param_init_vals": param_init_vals,
})

    print(f"Completed run {i+1}")
