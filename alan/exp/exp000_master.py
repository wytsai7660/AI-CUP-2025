# # master.py
# import subprocess
# import sys
# import yaml
# import os
# import time
# import tempfile


# batch_size = 32,
# n_epochs = 10,
# lr = 1e-5,
# weight_decay = 0.05,
# num_warmup_steps = 10,
# main_loss_weight = 0.6,
# loss_type = 'BCE',
# model_type = 'gru',
# seq_len = 500,
# shift = 250,

# # input_dim = 24,
# # n_enc = 2,
# # nhead = 8,
# # d_model = 64,
# # max_seq_len = 500,
# # dropout = 0.2,
# # mean_pooling = True,
# # pos_emb = False,

# input_dim = 24,
# hidden_dim = 64,
# d_model = 128,
# dropout = 0.2,
# mean_pooling = True,

# # ablations: encoder vs gru
# # learning rate, batch size, weight decay
# # main loss weight
# # loss type
# # mean pooling
# # 


# ablation_configs = [
#     {
#         "name": "gru baseline CE len1000",
#         "batch_size": 32,
#         "n_epochs": 20,
#         "lr": 1e-5,
#         "weight_decay": 0.05,
#         "num_warmup_steps": 10,
#         "main_loss_weight": 0.2,
#         "loss_type": "CE",
#         "model_type": "gru",
#         "seq_len": 1000,
#         "shift": 500,
#         "input_dim": 24,
#         "hidden_dim": 64,
#         "d_model": 128,
#         "dropout": 0.2,
#         "mean_pooling": True,
#     },
#     {
#         "name": "gru baseline CE len1000",
#         "batch_size": [8, 16, 32],
#         "n_epochs": [5, 10, 20],
#         "lr": [1e-3, 1e-4, 1e-5],
#         "weight_decay": [5e-2, 5e-3, 5e-4],
#         "num_warmup_steps": 10,
#         "main_loss_weight": [0.05, 0.2, 0.6, 0.8],
#         "loss_type": "CE",
#         "model_type": "gru",
#         "seq_len": 1000,
#         "shift": 500,
#         "input_dim": 24,
#         "hidden_dim": 64,
#         "d_model": 128,
#         "dropout": [0.2, 0.5],
#         "mean_pooling": True,
#         "use_aug_features": "basic",
#         "use_mode_as_target": False,
#         "split_type": "level",
#         "output_dim": 11,
#     },

#     {
#         "name": "gru low main weight CE len1000",
#         "batch_size": 32,
#         "n_epochs": 20,
#         "lr": 1e-5,
#         "weight_decay": 0.05,
#         "num_warmup_steps": 10,
#         "main_loss_weight": 0.05,
#         "loss_type": "CE",
#         "model_type": "gru",
#         "seq_len": 1000,
#         "shift": 500,
#         "input_dim": 24,
#         "hidden_dim": 64,
#         "d_model": 128,
#         "dropout": 0.2,
#         "mean_pooling": True,
#     },

#     {
#         "name": "gru heavy dropout CE len1000",
#         "batch_size": 32,
#         "n_epochs": 20,
#         "lr": 1e-4,
#         "weight_decay": 0.05,
#         "num_warmup_steps": 10,
#         "main_loss_weight": 0.2,
#         "loss_type": "CE",
#         "model_type": "gru",
#         "seq_len": 1000,
#         "shift": 500,
#         "input_dim": 24,
#         "hidden_dim": 64,
#         "d_model": 128,
#         "dropout": 0.5,
#         "mean_pooling": True,
#     },
#     {
#         "name": "encoder baseline CE len1000",
#         "batch_size": [8, 16, 32],
#         "n_epochs": [5, 10, 20],
#         "lr": [1e-3, 1e-4, 1e-5],
#         "weight_decay": [5e-2, 5e-3, 5e-4],
#         "num_warmup_steps": 10,
#         "main_loss_weight": [0.05, 0.2, 0.6, 0.8],
#         "loss_type": "CE",
#         "model_type": "encoder",
#         "seq_len": 1000,
#         "shift": 500,
#         "n_enc": 6,
#         "nhead": 8,
#         "d_model": 128,
#         "max_seq_len": 1000,
#         "dropout": [0.2, 0.5],
#         "pos_emb": True,
#         "mean_pooling": True,
#         "input_dim": 24,
#     },
# ]

# script_to_run = "exp004_ablation.py"
# temp_config_dir = "temp_configs"

# os.makedirs(temp_config_dir, exist_ok=True)

# print(f"Starting ablation study with {len(ablation_configs)} configurations.")

# for i, config in enumerate(ablation_configs):
#     print(f"\n--- Running Configuration {i+1}/{len(ablation_configs)} ---")
#     print("Configuration details:")
#     print(yaml.dump(config, indent=4))

#     temp_config_filename = os.path.join(temp_config_dir, f"config_run_{i+1}.yaml")
#     try:
#         with open(temp_config_filename, 'w') as f:
#             yaml.dump(config, f)
#         print(f"Saved configuration to {temp_config_filename}")

#         command = [sys.executable, script_to_run, '--config', temp_config_filename]

#         print(f"\nExecuting command: {' '.join(command)}")

#         result = subprocess.run(command, capture_output=True, text=True, check=True)

#         print("\n--- Subprocess Output (stdout) ---")
#         print(result.stdout)

#         if result.stderr:
#             print("\n--- Subprocess Output (stderr) ---")
#             print(result.stderr)

#         print(f"\n--- Configuration {i+1} Finished Successfully ---")

#     except FileNotFoundError:
#         print(f"\nError: The script '{script_to_run}' was not found.")
#         print("Please make sure your_script_name.py is in the correct path.")
#         break 
#     except yaml.YAMLError as e:
#          print(f"\nError: Could not write YAML configuration to {temp_config_filename} - {e}")
#     except subprocess.CalledProcessError as e:
#         print(f"\nError: Configuration {i+1} Failed!")
#         print(f"Command: {' '.join(e.cmd)}")
#         print(f"Return Code: {e.returncode}")
#         print("\n--- Subprocess Error Output (stderr) ---")
#         print(e.stderr)
#         print("\n--- Subprocess Output (stdout) ---")
#         print(e.stdout)
#         # 可以選擇是否繼續下一組
#     except Exception as e:
#          print(f"\nAn unexpected error occurred during configuration {i+1}: {e}")
#          # 可以選擇是否繼續下一組

#     # 在兩次運行之間稍作間隔
#     if i < len(ablation_configs) - 1:
#         time.sleep(1) # 間隔 1 秒

# print("\n--- Ablation Study Complete ---")
# print(f"Temporary configuration files saved in '{temp_config_dir}'. You can review or clean them up.")

import os
import sys
import subprocess
import yaml
import time
import itertools
import copy

ablation_configs = [
    # {
    #     "name": "encoder_CE",
    #     "batch_size": [8, 16, 32],
    #     "n_epochs": [10, 20],
    #     "lr": [1e-3, 1e-4, 1e-5],
    #     "weight_decay": [5e-2, 5e-3, 5e-4],
    #     "num_warmup_steps": 10,
    #     "main_loss_weight": [0.05, 0.2, 0.6, 0.8],
    #     "loss_type": "CE",
    #     "model_type": "encoder",
    #     "seq_len": 1000,
    #     "shift": 500,
    #     "n_enc": 6,
    #     "nhead": 8,
    #     "d_model": 128,
    #     "max_seq_len": 1000,
    #     "dropout": 0.5,
    #     "pos_emb": [True, False],
    #     "mean_pooling": True,
    #     "input_dim": 24,
    #     "use_aug_features": "basic", # Fixed
    #     "use_mode_as_target": False, # Fixed
    #     "split_type": "level", # Fixed
    #     "output_dim": 11, # Fixed

    # },

    # {
    #     "name": "encoder_baseline_CE", # Changed to be more of a base name
    #     "batch_size": [8, 16, 32],
    #     "n_epochs": [10, 20],
    #     "lr": [1e-3, 1e-4, 1e-5],
    #     "weight_decay": [5e-2, 5e-3, 5e-4],
    #     "num_warmup_steps": 10, # Fixed
    #     "main_loss_weight": [0.05, 0.2, 0.6, 0.8],
    #     "loss_type": "CE", # Fixed
    #     "model_type": "encoder", # Fixed
    #     "seq_len": 1000, # Fixed
    #     "shift": 500, # Fixed
    #     "input_dim": 24, # Fixed
    #     "hidden_dim": 64, # Fixed
    #     "d_model": 128, # Fixed
    #     "dropout": [0.2, 0.5],
    #     "mean_pooling": True, # Fixed
    #     "use_aug_features": "basic", # Fixed
    #     "use_mode_as_target": False, # Fixed
    #     "split_type": "level", # Fixed
    #     "output_dim": 11, # Fixed
    # },
    # {
    #     "name": "gru_best_macro_roc", # Changed to be more of a base name
    #     "batch_size": 8,
    #     "n_epochs": 20,
    #     "lr": 1e-3,
    #     "weight_decay": 5e-2,
    #     "num_warmup_steps": 10, # Fixed
    #     "main_loss_weight": 0.2,
    #     "loss_type": "CE", # Fixed
    #     "model_type": "gru", # Fixed
    #     "seq_len": 1000, # Fixed
    #     "shift": 500, # Fixed
    #     "input_dim": 24, # Fixed
    #     "hidden_dim": 64, # Fixed
    #     "d_model": 128, # Fixed
    #     "dropout": 0.2,
    #     "mean_pooling": True, # Fixed
    #     "use_aug_features": "basic", # Fixed
    #     "use_mode_as_target": False, # Fixed
    #     "split_type": "level", # Fixed
    #     "output_dim": 11, # Fixed
    # },
    # {
    #     "name": "gru_best_most", # Changed to be more of a base name
    #     "batch_size": 8,
    #     "n_epochs": 20,
    #     "lr": 1e-3,
    #     "weight_decay": 5e-3,
    #     "num_warmup_steps": 10, # Fixed
    #     "main_loss_weight": 0.8,
    #     "loss_type": "CE", # Fixed
    #     "model_type": "gru", # Fixed
    #     "seq_len": 1000, # Fixed
    #     "shift": 500, # Fixed
    #     "input_dim": 24, # Fixed
    #     "hidden_dim": 64, # Fixed
    #     "d_model": 128, # Fixed
    #     "dropout": 0.5,
    #     "mean_pooling": True, # Fixed
    #     "use_aug_features": "basic", # Fixed
    #     "use_mode_as_target": False, # Fixed
    #     "split_type": "level", # Fixed
    #     "output_dim": 11, # Fixed
    # },
    {
        "name": "gru_best_macro_f1", # Changed to be more of a base name
        "batch_size": 8,
        "n_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 5e-2,
        "num_warmup_steps": 10, # Fixed
        "main_loss_weight": 0.05,
        "loss_type": "CE", # Fixed
        "model_type": "gru", # Fixed
        "seq_len": 1000, # Fixed
        "shift": 500, # Fixed
        "input_dim": 24, # Fixed
        "hidden_dim": 64, # Fixed
        "d_model": 128, # Fixed
        "dropout": 0.5,
        "mean_pooling": True, # Fixed
        "use_aug_features": "basic", # Fixed
        "use_mode_as_target": False, # Fixed
        "split_type": "level", # Fixed
        "output_dim": 11, # Fixed
    },
]

script_to_run = "exp007_sweep.py" # Make sure this script exists and is executable
temp_config_dir = "temp_configs"

os.makedirs(temp_config_dir, exist_ok=True)

def generate_run_configs(base_config_template):
    """
    Generates individual run configurations from a base template
    by creating a Cartesian product of list-valued parameters.
    """
    sweep_params = {}
    fixed_params = {}
    original_name = base_config_template.get("name", "run")

    for key, value in base_config_template.items():
        if key == "name": # Keep original name separate for base
            continue
        if isinstance(value, list):
            sweep_params[key] = value
        else:
            fixed_params[key] = value

    if not sweep_params:
        # If no lists to sweep, just return the original config
        # but ensure it's a list of one for consistency
        config = copy.deepcopy(fixed_params)
        config["name"] = original_name
        return [config]

    sweep_param_names = list(sweep_params.keys())
    sweep_param_values_list = list(sweep_params.values())

    generated_configs = []
    for value_combination in itertools.product(*sweep_param_values_list):
        current_run_config = copy.deepcopy(fixed_params)
        run_specific_name_parts = [original_name]
        for i, param_name in enumerate(sweep_param_names):
            param_value = value_combination[i]
            current_run_config[param_name] = param_value
            # Create a short representation for the name
            # Handle scientific notation for learning rates gracefully in names
            if isinstance(param_value, float) and param_value < 1e-2 :
                 val_str = f"{param_value:.0e}" # e.g., 1e-3
            else:
                 val_str = str(param_value)
            run_specific_name_parts.append(f"{param_name.replace('_','')}-{val_str}")

        current_run_config["name"] = "_".join(run_specific_name_parts)
        generated_configs.append(current_run_config)

    return generated_configs

all_individual_runs = []
for base_config in ablation_configs:
    all_individual_runs.extend(generate_run_configs(base_config))

print(f"Starting ablation study with {len(ablation_configs)} base configurations, "
      f"generating a total of {len(all_individual_runs)} individual experiment runs.")

run_counter = 0
for i, run_config in enumerate(all_individual_runs):
    run_counter += 1
    print(f"\n--- Running Configuration {run_counter}/{len(all_individual_runs)} ---")
    print("Configuration details:")
    # Use sort_keys=False to maintain order for readability if desired, though not critical
    print(yaml.dump(run_config, indent=4, sort_keys=False))

    temp_config_filename = os.path.join(temp_config_dir, f"config_run_{run_counter}.yaml")
    try:
        with open(temp_config_filename, 'w') as f:
            yaml.dump(run_config, f, sort_keys=False)
        print(f"Saved configuration to {temp_config_filename}")

        # Ensure script_to_run is in the path or provide full path
        # Example: command = [sys.executable, os.path.join(os.getcwd(), script_to_run), '--config', temp_config_filename]
        command = [sys.executable, script_to_run, '--config', temp_config_filename]  #'--device', 'cuda:1', '--wandb'

        print(f"\nExecuting command: {' '.join(command)}")
        
        # Consider adding a timeout to subprocess.run if runs can hang
        result = subprocess.run(command, capture_output=True, text=True, check=False) # check=False to handle errors manually

        if result.returncode == 0:
            print("\n--- Subprocess Output (stdout) ---")
            print(result.stdout)
            # if result.stderr: # Sometimes successful runs still print to stderr (e.g., warnings)
            #     print("\n--- Subprocess Output (stderr) ---")
            #     print(result.stderr)
            print(f"\n--- Configuration {run_counter} Finished Successfully ---")
        else:
            # This block is now effectively the same as CalledProcessError handling
            print(f"\nError: Configuration {run_counter} Failed!")
            print(f"Command: {' '.join(command)}") # Using command list directly
            print(f"Return Code: {result.returncode}")
            print("\n--- Subprocess Error Output (stderr) ---")
            print(result.stderr)
            print("\n--- Subprocess Output (stdout) ---")
            print(result.stdout)
            # Decide whether to continue or break
            # break # Uncomment to stop on first failure

    except FileNotFoundError:
        print(f"\nError: The script '{script_to_run}' was not found.")
        print("Please make sure your_script_name.py is in the correct path or provide an absolute path.")
        break
    except yaml.YAMLError as e:
         print(f"\nError: Could not write YAML configuration to {temp_config_filename} - {e}")
         break # Likely a critical error
    except subprocess.CalledProcessError as e: # This might not be hit if check=False
        print(f"\nError: Configuration {run_counter} Failed (CalledProcessError)!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("\n--- Subprocess Error Output (stderr) ---")
        print(e.stderr)
        print("\n--- Subprocess Output (stdout) ---")
        print(e.stdout)
        # break # Uncomment to stop on first failure
    except Exception as e:
         print(f"\nAn unexpected error occurred during configuration {run_counter}: {e}")
         # break # Uncomment to stop on unexpected errors

    if run_counter < len(all_individual_runs):
        print(f"Waiting for 1 second before next run...")
        time.sleep(1)

print("\n--- Ablation Study Complete ---")
print(f"Temporary configuration files saved in '{temp_config_dir}'. You can review or clean them up.")