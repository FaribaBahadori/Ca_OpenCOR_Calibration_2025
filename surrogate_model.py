from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from Functions.model_utils import SimulationManager
import numpy as np
import os
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import random
from itertools import combinations
from sklearn.metrics import mean_squared_error, r2_score

def top_n_argmax(array, n):
    idx = np.argpartition(array, -n)[-n:]
    return idx[np.argsort(array[idx])][::-1]

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = top_n_argmax(std, n_batch)
    return query_idx, X[query_idx]


param_names = ['SMC_Par/delta_SMC', 'SMC_Par/k_RyR']      # This list can be appended
output_names = ['model_parameters1/Ca_in_SMC', 'SMC_Par/Ca_in_SMC_dig']
output_type = 'max_Ca'
output_labels = ['max_Ca', 'time_to_max', 'relax_time']  # Customize as needed

# ---- Build 3 datasets for each parameter ----
n_samples = 200
datasets = []
for i, param in enumerate(param_names):

    data_path = f"outputs/model_dataset/run3/dataset_OAT_param{i}_qmc_200.csv"

    # Load data
    data = pd.read_csv(data_path)
    X = data[param_names].values
    Y = data[output_labels].values

    # Random subset
    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    Y = Y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=42)

    datasets.append({
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test
    })

working_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(working_dir, "Models/presynaptic_varicosity.cellml")

# Setting the output directory
dataset_idx = 1
output_file_path = f"outputs/surrogate/train/v16/single_param/"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

sim_manager = SimulationManager(cal_param_names=param_names)
ground_truth = sim_manager.get_ground_truth()

# Load data from CSV file
data_file_path = "outputs/model_dataset/run6/model_dataset_FULL_qmc_3000.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(data_file_path)

# Assuming the CSV has columns for inputs (X) and outputs (Y)
X = data[param_names].values  # Extract input features based on param_names
Y = data[['max_Ca', 'time_to_max', 'relax_time']].values  # Extract output features based on output_names
# Set the number of samples to use
n_samples = 3000 
# Randomly select n_samples from the dataset
indices = np.random.choice(len(X), n_samples, replace=False)
X = X[indices]
Y = Y[indices]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#######        There is also an option to just use OAT datasets for both train and test process     ##########
# X_train, Y_train, X_test, Y_test = datasets[dataset_idx]["X_train"], datasets[dataset_idx]["Y_train"], datasets[dataset_idx]["X_test"], datasets[dataset_idx]["Y_test"]

# Choosing random samples for training at first step
init_train_sample_num = int(0.6 * X_train.shape[0])
rnd_train_idx = [random.randint(0, X_train.shape[0]-1) for _ in range(init_train_sample_num)]

# ---- Build 3 Regressors ----
n_outputs = 3
regressors = []

for i in range(n_outputs):
    kernel = RBF()
    # kernel = Matern(nu=4)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)

    learner = ActiveLearner(estimator=gpr, query_strategy=GP_regression_std,
        X_training=X_train[rnd_train_idx, :], y_training=Y_train[rnd_train_idx, i]  # train on the i-th output
    )

    regressors.append(learner)

n_batch = 60
std_threshold = 0.05 #efine the threshold for stopping criteria
batch_idx = 0

while True:

    for i, reg in enumerate(regressors):
        query_idx, _ = reg.query(X_train)
        print(f"Querying {len(query_idx)} samples for output {i+1} in batch {batch_idx + 1}")
        reg.teach(X_train[query_idx], Y_train[query_idx, i])

    # Predict for the test dataset
    pred_means = []
    pred_stds = []
    for i, reg in enumerate(regressors):
        mean, std = reg.predict(X_train, return_std=True)
        pred_means.append(mean)
        pred_stds.append(std)

    print("Total samples: {}".format(init_train_sample_num + n_batch * (batch_idx + 1)))

    # Check stopping criteria
    max_std = max([np.max(std) for std in pred_stds])
    print(f"Max std in batch {batch_idx}: {max_std}")
    if max_std < std_threshold:
        print("Stopping criteria met. Exiting loop.")
        break

    batch_idx += 1

for i in range(len(regressors)):
    y_mean, y_std = regressors[i].predict(X_test, return_std=True)  

    # Compute Accuracy Metrics
    mse = mean_squared_error(Y_test[:, i], y_mean)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(Y_test[:, i], y_mean)

    print(mse, rmse, r2)

for j, input_lable in enumerate(param_names):
    for i, label in enumerate(output_labels):

        X_test = datasets[j]["X_test"]
        Y_test = datasets[j]["Y_test"]

        
        y_mean, y_std = regressors[i].predict(X_test, return_std=True)  

        # Compute Accuracy Metrics
        mse = mean_squared_error(Y_test[:, i], y_mean)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(Y_test[:, i], y_mean)

        # print(mse, rmse, r2)

        plt.figure(figsize=(8,6))
        x = X_test[:, j]
        lower = y_mean - 1.96 * y_std
        upper = y_mean + 1.96 * y_std

        # Sort by x for better plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        mean_sorted = y_mean[sort_idx]
        lower_sorted = lower[sort_idx]
        upper_sorted = upper[sort_idx]
        y_true_sorted = Y_test[sort_idx, i]

        plt.plot(x_sorted, mean_sorted, 'b-', label='Surrogate model Prediction')
        plt.fill_between(x_sorted, lower_sorted, upper_sorted, color='blue', alpha=0.2, label='95% CI')
        plt.scatter(x_sorted, y_true_sorted, color='red', alpha=0.5, label='Original model outputs')

        y_range = [np.min(y_true_sorted), np.max(y_true_sorted)]
        plt.ylim([0.9*y_range[0], 1.1*y_range[1]])
        plt.xlim([0, np.max(x_sorted)*1.01])
        plt.xlabel(param_names[j])
        if label == "max_Ca":
            label += " (uM)"
        else:
            label += " (s)"
        plt.ylabel(label)
        plt.title(f"{label} vs {param_names[j]}")
        plt.grid(True)
                
        # **Add Accuracy as a Text Annotation**
        plt.annotate(f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.6f}", 
                    xy=(0.80, 0.85), xycoords="axes fraction", fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
        
        plt.tight_layout()
        plt.legend(loc='lower center', ncol=3)
        plt.savefig(os.path.join(output_file_path, f"test_dataset_{label}_vs_{input_lable.replace("/", "_")}.png"))
        plt.close()