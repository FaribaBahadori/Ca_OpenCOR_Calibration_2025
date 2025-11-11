
import csv, itertools, numpy as np, sys, os, math
from datetime import datetime

# Shows which Python program is running this script
print("Python executable:", sys.executable)
os.chdir(r"C:/Fariba_2025/Ca_OpenCOR_Calibration_2025")
sys.path.insert(0, os.getcwd())
def init_set():

# define parameters: name : (lower, upper, n)
    params = {
        'l_1': (2000, 2000, 0),       # 2000 //
        'l_2': (1, 1, 0),             # 1 //
        'l_3': (2000, 2000, 0),       # 2000 //
        'l_4': (1, 1, 0),             # 1 \\
        'l_5': (100, 100, 0),         # 100 //
        'l_m1': (260, 260, 0),        # 260 //
        'l_m2': (1.05, 1.05, 0),      # 1.05 //
        'l_m3': (1886, 1886, 0),      # 1886 //
        'l_m4': (0.145, 0.145, 0),    # 0.145 //
        'l_m5': (8.2, 8.2, 0),        # 8.2 //
        'p_agonist': (0, 0, 0),    # 0 //
        'Ca_E': (1600, 1600, 0),      # 1600 \\
        't1_KCL': (2, 2, 0),          # 2 \\
        't2_KCL': (1802, 1802, 0),    # 1802 \\
        'alpha0': (0.05, 0.1, 0),     # 0.05
        'alpha1': (0.25, 0.5, 0),        # 0.25
        'alpha2': (1, 1, 0),          # 1 //
        'V0': (-60, -60, 0),          # -60  \\
        'V1': (-30, -30, 0),          # -30
        'k_ryr0': (0.0072, 1, 1),    # 0.0072        vm
        'k_ryr1': (0.3, 10, 1),      # 0.334           km
        'k_ryr2': (0.1, 10, 1),        # 0.5         gca
        'k_ryr3': (38, 50, 1),        # 38            F
        'Vm': (-50, -50, 0),          # -50           R
        'km': (12, 12, 0),           # 12             T
        'gca': (16, 30, 1),            # 16           k_ryr0
        'F': (96485, 96485, 0),       # 96485 \\      k_ryr1
        'R': (8345, 8345, 0),         # 8345 \\       k_ryr2
        'T': (310, 310, 0),           # 310  \\       k_ryr3
        'Jer': (0.1, 0.1, 0),        # 0.1            Jer
        'Ve': (4.5, 4.5, 1),              # 4.5       Ve
        'Ke': (0.1, 0.15, 0),            # 0.1         Ke
        'Vp': (4.5, 4.5, 0),              # 4.5       Vp
        'Kp': (0.4, 0.4, 0),            # 0.4         Kp
        'gamma': (5.5, 5.5, 0),       # 5.5
        'delta_SMC': (0.05, 0.05, 0),  # 0.05
        'k_RyR': (5, 5, 0),          # 5
        'k_ipr': (5.55, 5.55, 0)        # 5.55 //
    }

    # Setting the output directory
    base_path = "outputs/Ca_Fitting/Sets/"
    # Add timestamp for unique filenames
    timestamp1 = datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print("not os.path.exists(base_path)")
    # make value sets
    sets = {p: [round(lo,8)] if (lo == hi or n == 0) else np.linspace(lo, hi, n+1).round(6).tolist() 
            for p,(lo,hi,n) in params.items()}
    # write Parameter_Division_Set.csv
    with open(os.path.join(base_path, f'Par_Division_Set_{timestamp1}.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([[p]+v for p,v in sets.items()])
    n_rows = math.prod(len(v) for v in sets.values())
    print(f"Expected number of rows = {n_rows}")
    # make Cartesian product and write Initial_Par_Set.csv
    names = list(sets)
    with open(os.path.join(base_path, f'Initial_Par_Set_{timestamp1}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(itertools.product(*sets.values()))
    print("Parameter_Division_Set.csv and Initial_Par_Set.csv created.")
    print(f"Rows written = {sum(1 for _ in open(os.path.join(base_path, f'Initial_Par_Set_{timestamp1}.csv')))-1}")
    print("Files will be saved in:", os.getcwd())
    return os.path.join(base_path, f'Initial_Par_Set_{timestamp1}.csv')
