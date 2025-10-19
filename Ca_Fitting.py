import pandas as pd
import datetime, os, sys
os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())
from Functions.init_set import init_set
import Calibration as calib  # your existing script

# 1. Create initial parameter sets CSV
csv_path = init_set()

# 2. Load all parameter sets
param_sets = pd.read_csv(csv_path)

# 3. Prepare output directory
output_dir = r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025/outputs/Ca_Fitting/run222"
os.makedirs(output_dir, exist_ok=True)

# 4. Loop over each row
for i, row in param_sets.iterrows():
    print(f"\n=== Running optimization {i+1}/{len(param_sets)} ===")

    # Pass the row values to the calibration script
    param_init_vals = row.to_list()  # this will override default in your script
    # Now run the calibration script (it will use this row as initial params)
    # Option 1: import and run main() if you wrap your script in a function
    # Option 2: exec the script (simplest, no changes inside script)
    with open(calib.__file__) as f:
        code = f.read()
    exec(code, globals())

    print(f"Completed run {i+1}")
