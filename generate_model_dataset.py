from Functions.model_utils import SimulationManager
import os
import pandas as pd
import numpy as np

param_names = ['ACh_release/kappa_f', 'ACh_release/R_ACh', 'membrane/R_Ca_dis']      # This list can be appended
output_names = ['max_Ca', 'time_to_max', 'relax_time']  # Output names for the outputs

working_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(working_dir, "Models/presynaptic_varicosity.cellml")

# Setting the output directory
output_file_path = "outputs/model_dataset/run6/"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

sim_manager = SimulationManager(cal_param_names=param_names)

# Generate dataset for surrogate
sample_method = 'qmc'  # or 'latin'
num_samples = 3000  # Number of samples to generate

# Y, X = sim_manager.generate_samples(sample_method=sample_method, num_samples=num_samples, verbose=True)

# # Save the dataset
# # Create a DataFrame with parameter names and output names
# data = pd.DataFrame(X, columns=param_names)
# for i, output_name in enumerate(output_names):
#     data[output_name] = Y[:, i]

# # Save the DataFrame to a CSV file
# data.to_csv(os.path.join(output_file_path, f"model_dataset_{sample_method}_{num_samples}.csv"), index=False)

# all_datasets = sim_manager.generate_samples_OAT_Full(sample_method=sample_method, num_samples=num_samples, verbose=True)

# for dataset in all_datasets:
#     samples = dataset["samples"]
#     results = dataset["results"]
#     variation = dataset["variation"]
#     param_index = dataset["param_index"]

#     # Build DataFrame
#     df = pd.DataFrame(samples, columns=param_names)
#     for i, output_name in enumerate(output_names):
#         df[output_name] = results[:, i]

#     # Build filename
#     if variation == "OAT":
#         filename = f"dataset_OAT_param{param_index}_{sample_method}_{num_samples}.csv"
#     elif variation == "FULL":
#         filename = f"dataset_FULL_{sample_method}_{num_samples}.csv"

#     # Save CSV
#     filepath = os.path.join(output_file_path, filename)
#     df.to_csv(filepath, index=False)


# all_datasets = sim_manager.generate_samples_two_param_combinations(num_samples=num_samples, sample_method=sample_method, verbose=True)

# for dataset in all_datasets:
#     samples = dataset["samples"]
#     results = dataset["results"]
#     variation = dataset["variation"]
#     param_index = dataset["param_indices"]

#     # Build DataFrame
#     df = pd.DataFrame(samples, columns=param_names)
#     for i, output_name in enumerate(output_names):
#         df[output_name] = results[:, i]

#     # Build filename
#     if variation == "OAT":
#         filename = f"dataset_OAT_param{param_index}_{sample_method}_{num_samples}.csv"
#     elif variation == "FULL":
#         filename = f"dataset_FULL_{sample_method}_{num_samples}.csv"
#     elif variation == "2D_COMBO":
#         filename = f"dataset_two_param{param_index[0]}_{param_index[1]}_{sample_method}_{num_samples}.csv"

#     # Save CSV
#     filepath = os.path.join(output_file_path, filename)
#     df.to_csv(filepath, index=False)


Y, X = sim_manager.generate_samples(sample_method=sample_method, num_samples=num_samples, verbose=True)

# Save the dataset
# Create a DataFrame with parameter names and output names
data = pd.DataFrame(X, columns=param_names)
for i, output_name in enumerate(output_names):
    data[output_name] = Y[:, i]

# Save the DataFrame to a CSV file
data.to_csv(os.path.join(output_file_path, f"model_dataset_FULL_{sample_method}_{num_samples}.csv"), index=False)