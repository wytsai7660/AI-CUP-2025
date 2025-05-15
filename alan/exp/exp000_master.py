# master.py
import subprocess
import sys
import yaml
import os
import time
import tempfile


batch_size = 32,
n_epochs = 10,
lr = 1e-5,
weight_decay = 0.05,
num_warmup_steps = 10,
main_loss_weight = 0.6,
loss_type = 'BCE',
model_type = 'gru',
seq_len = 500,
shift = 250,

# input_dim = 24,
# n_enc = 2,
# nhead = 8,
# d_model = 64,
# max_seq_len = 500,
# dropout = 0.2,
# mean_pooling = True,
# pos_emb = False,

input_dim = 24,
hidden_dim = 64,
d_model = 128,
dropout = 0.2,
mean_pooling = True,

# ablations: encoder vs gru
# learning rate, batch size, weight decay
# main loss weight
# loss type
# mean pooling
# 


ablation_configs = [
    {
        "name": "gru baseline CE len1000",
        "batch_size": 32,
        "n_epochs": 20,
        "lr": 1e-5,
        "weight_decay": 0.05,
        "num_warmup_steps": 10,
        "main_loss_weight": 0.2,
        "loss_type": "CE",
        "model_type": "gru",
        "seq_len": 1000,
        "shift": 500,
        "input_dim": 24,
        "hidden_dim": 64,
        "d_model": 128,
        "dropout": 0.2,
        "mean_pooling": True,
    },
    {
        "name": "gru low main weight CE len1000",
        "batch_size": 32,
        "n_epochs": 20,
        "lr": 1e-5,
        "weight_decay": 0.05,
        "num_warmup_steps": 10,
        "main_loss_weight": 0.05,
        "loss_type": "CE",
        "model_type": "gru",
        "seq_len": 1000,
        "shift": 500,
        "input_dim": 24,
        "hidden_dim": 64,
        "d_model": 128,
        "dropout": 0.2,
        "mean_pooling": True,
    },

    {
        "name": "gru heavy dropout CE len1000",
        "batch_size": 32,
        "n_epochs": 20,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "num_warmup_steps": 10,
        "main_loss_weight": 0.2,
        "loss_type": "CE",
        "model_type": "gru",
        "seq_len": 1000,
        "shift": 500,
        "input_dim": 24,
        "hidden_dim": 64,
        "d_model": 128,
        "dropout": 0.5,
        "mean_pooling": True,
    },
    {
        "name": "encoder high lr CE len1000",
        "batch_size": 32,
        "n_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "num_warmup_steps": 10,
        "main_loss_weight": 0.2,
        "loss_type": "CE",
        "model_type": "encoder",
        "seq_len": 1000,
        "shift": 500,
        "n_enc": 6,
        "nhead": 8,
        "d_model": 128,
        "max_seq_len": 1000,
        "dropout": 0.2,
        "pos_emb": True,
        "mean_pooling": True,
        "input_dim": 24,
    },
]

script_to_run = "exp004_ablation.py"
temp_config_dir = "temp_configs"

os.makedirs(temp_config_dir, exist_ok=True)

print(f"Starting ablation study with {len(ablation_configs)} configurations.")

for i, config in enumerate(ablation_configs):
    print(f"\n--- Running Configuration {i+1}/{len(ablation_configs)} ---")
    print("Configuration details:")
    print(yaml.dump(config, indent=4))

    temp_config_filename = os.path.join(temp_config_dir, f"config_run_{i+1}.yaml")
    try:
        with open(temp_config_filename, 'w') as f:
            yaml.dump(config, f)
        print(f"Saved configuration to {temp_config_filename}")

        command = [sys.executable, script_to_run, '--config', temp_config_filename]

        print(f"\nExecuting command: {' '.join(command)}")

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("\n--- Subprocess Output (stdout) ---")
        print(result.stdout)

        if result.stderr:
            print("\n--- Subprocess Output (stderr) ---")
            print(result.stderr)

        print(f"\n--- Configuration {i+1} Finished Successfully ---")

    except FileNotFoundError:
        print(f"\nError: The script '{script_to_run}' was not found.")
        print("Please make sure your_script_name.py is in the correct path.")
        break 
    except yaml.YAMLError as e:
         print(f"\nError: Could not write YAML configuration to {temp_config_filename} - {e}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Configuration {i+1} Failed!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("\n--- Subprocess Error Output (stderr) ---")
        print(e.stderr)
        print("\n--- Subprocess Output (stdout) ---")
        print(e.stdout)
        # 可以選擇是否繼續下一組
    except Exception as e:
         print(f"\nAn unexpected error occurred during configuration {i+1}: {e}")
         # 可以選擇是否繼續下一組

    # 在兩次運行之間稍作間隔
    if i < len(ablation_configs) - 1:
        time.sleep(1) # 間隔 1 秒

print("\n--- Ablation Study Complete ---")
print(f"Temporary configuration files saved in '{temp_config_dir}'. You can review or clean them up.")

# 清理臨時文件 (可選)
# print(f"\nCleaning up temporary config files in '{temp_config_dir}'...")
# try:
#     for entry in os.listdir(temp_config_dir):
#         os.remove(os.path.join(temp_config_dir, entry))
#     os.rmdir(temp_config_dir)
#     print("Cleanup complete.")
# except Exception as e:
#     print(f"Error during cleanup: {e}")