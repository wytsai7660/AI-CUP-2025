# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import time
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt # <--- 導入 matplotlib
from torch.cuda.amp import autocast,grad_scaler # <--- 導入自動混合精度

# Import from your model file
# Make sure model.py contains:
# - TimeSeriesDataset, build_transformer
# - collate_fn (if not using universal_collate_fn from dataloader.py)
#   If you created dataloader.py, you might import TimeSeriesDataset and universal_collate_fn from there
try:
    from model import  build_transformer
    from dataloader import TimeSeriesDataset, collate_fn
except ImportError:
    print("Warning: Could not import from model.py. Trying dataloader.py...")
    try:
        # If you created dataloader.py with TimeSeriesDataset and universal_collate_fn
        from dataloader import TimeSeriesDataset, universal_collate_fn as collate_fn
        from model import build_transformer # Still need model builder from model.py
    except ImportError:
        print("Error: Could not import necessary components from model.py or dataloader.py.")
        print("Please ensure TimeSeriesDataset, collate_fn/universal_collate_fn, and build_transformer are available.")
        exit(1)


# --- Helper Functions ---

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_masks(src_batch, tgt_batch, pad_idx=0.0): #<-- Default pad_idx to float
    """
    Creates the necessary masks for the Transformer model.
    Args:
        src_batch: Source sequence batch (batch_size, src_seq_len, features).
        tgt_batch: Target sequence batch for decoder input (batch_size, tgt_seq_len, features).
        pad_idx: The padding index/value (assuming 0.0 for padded time series).
    Returns:
        src_mask: Mask for source sequence padding.
        tgt_mask: Combined mask for target sequence padding and look-ahead.
    """
    src_mask = (src_batch[:, :, 0] != pad_idx).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, src_len)
    tgt_seq_len = tgt_batch.size(1)
    # Check if tgt_seq_len is 0, return None or empty masks if so
    if tgt_seq_len == 0:
        return src_mask, None # Or handle as needed

    tgt_pad_mask = (tgt_batch[:, :, 0] != pad_idx).unsqueeze(1).unsqueeze(-1) # (batch, 1, tgt_len, 1)
    look_ahead_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt_batch.device)).bool() # (tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & look_ahead_mask # (batch, 1, tgt_len, tgt_len)
    return src_mask, tgt_mask


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model checkpoint."""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

# --- Plotting Function ---
def plot_losses(train_losses, val_losses, save_path):
    """Plots training and validation losses and saves the figure."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close the figure to free memory


# --- Training and Validation Functions (Keep as before) ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, grad_clip_value=1.0, scaler=None): # Added scaler
    """Trains the model for one epoch to predict static metadata."""
    model.train()
    total_loss = 0.0
    processed_batches = 0
    num_total_batches = len(dataloader)
    start_time = time.time()
    pbar = tqdm(enumerate(dataloader), total=num_total_batches, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch_data in pbar:
        if batch_data is None or batch_data[0] is None:
            continue

        padded_time_series, stacked_metadata, lengths, _ = batch_data # Unpack 4 items

        # --- Move data to device ---
        src = padded_time_series.to(device)
        # !!! Target is now the static metadata !!!
        target = stacked_metadata.to(device) # Shape: (batch, 9)

        # --- Prepare Decoder Input (tgt_input) ---
        # For predicting a single vector, the standard approach is to feed
        # only a start-of-sequence (SOS) token to the decoder.
        # Let's create a dummy SOS input of shape (batch, 1, d_input)
        # We can use zeros or a dedicated SOS embedding if available. Using zeros here.
        batch_size = src.size(0)
        d_input = src.size(2) # Get d_input from src
        # Create a single time step input for the decoder
        tgt_input = torch.zeros((batch_size, 1, d_input), device=device, dtype=src.dtype)

        # --- Create Masks ---
        # Source mask depends only on src padding
        src_mask = (src[:, :, 0] != 0.0).unsqueeze(1).unsqueeze(2)
        # Target mask for decoder self-attention needs to handle only the single SOS input step
        # It should prevent the SOS token from attending to anything else (trivial case here)
        tgt_mask = torch.ones((1, 1), device=device).bool() # A (1, 1) mask allowing attention to itself

        # --- Forward and Loss ---
        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            # Pass src, the single SOS token tgt_input, and masks
            output = model(src, tgt_input, src_mask, tgt_mask) # Expected shape: (batch, 9)

            # Ensure shapes match before loss calculation
            if output.shape != target.shape:
                 print(f"Shape mismatch ERROR in batch {batch_idx}: Output {output.shape}, Target {target.shape}")
                 continue # Skip this batch

            # Calculate loss directly between model output (batch, 9) and target metadata (batch, 9)
            # Consider if MSE is the best loss for all 9 features (esp. one-hot parts)
            loss = criterion(output, target)

        if torch.isnan(loss) or torch.isinf(loss):
             print(f"Warning: NaN or Inf loss encountered in batch {batch_idx}. Skipping batch update.")
             continue

        # --- Backward Pass ---
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1} [Train] - Avg Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")
    return avg_loss


def validate_model(model, dataloader, criterion, device, epoch):
    """Validates the model predicting static metadata."""
    model.eval()
    total_loss = 0.0
    processed_batches = 0
    num_total_batches = len(dataloader)
    start_time = time.time()
    pbar = tqdm(enumerate(dataloader), total=num_total_batches, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch_idx, batch_data in pbar:
            if batch_data is None or batch_data[0] is None: continue
            padded_time_series, stacked_metadata, lengths, _ = batch_data

            src = padded_time_series.to(device)
            target = stacked_metadata.to(device) # Target is metadata

            # --- Prepare Decoder Input (SOS token) ---
            batch_size = src.size(0)
            d_input = src.size(2)
            tgt_input = torch.zeros((batch_size, 1, d_input), device=device, dtype=src.dtype)

            # --- Create Masks ---
            src_mask = (src[:, :, 0] != 0.0).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.ones((1, 1), device=device).bool()

            # --- Forward Pass ---
            with autocast(enabled=(device.type == 'cuda')): # Use autocast for validation too
                output = model(src, tgt_input, src_mask, tgt_mask) # Expected shape: (batch, 9)

                if output.shape != target.shape:
                     print(f"Shape mismatch ERROR in validation batch {batch_idx}: Output {output.shape}, Target {target.shape}")
                     continue

                # Calculate loss
                loss = criterion(output, target)

            if torch.isnan(loss) or torch.isinf(loss):
                 loss_item = 0.0
            else:
                 loss_item = loss.item()

            total_loss += loss_item
            processed_batches += 1
            pbar.set_postfix({"loss": loss_item})

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1} [Val]   - Avg Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")
    return avg_loss


def validate_model(model, dataloader, criterion, device, epoch):
    """Validates the model."""
    model.eval()
    total_loss = 0.0
    processed_batches = 0
    num_total_batches = len(dataloader)
    start_time = time.time()
    pbar = tqdm(enumerate(dataloader), total=num_total_batches, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch_idx, batch_data in pbar:
            if batch_data is None or batch_data[0] is None:
                print(f"Warning: Skipping empty or problematic validation batch {batch_idx}")
                continue

            padded_time_series, stacked_metadata, lengths, _ = batch_data
            padded_time_series = padded_time_series.to(device)

            src = padded_time_series
            tgt_input = padded_time_series[:, :-1, :]
            tgt_output = padded_time_series[:, 1:, :]

            if tgt_input.size(1) == 0:
                continue

            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=0.0)
            if tgt_mask is None:
                continue

            output = model(src, tgt_input, src_mask, tgt_mask)
            output_dim = output.shape[-1]
            current_tgt_output = tgt_output[..., :output_dim]

            loss_calc_mask = (tgt_input[:, :, 0] != 0.0).unsqueeze(-1).repeat(1, 1, output_dim)

            if output.shape != current_tgt_output.shape or output.shape != loss_calc_mask.shape:
                print(f"Shape mismatch in validation batch {batch_idx}.")
                continue

            loss = criterion(output * loss_calc_mask, current_tgt_output * loss_calc_mask)

            num_actual_elements = loss_calc_mask.sum()
            if num_actual_elements > 0:
                 loss = loss * (loss_calc_mask.numel() / num_actual_elements)
            else:
                 loss = torch.tensor(0.0, device=device)


            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered in validation batch {batch_idx}. Treating as 0 for average.")
                loss_item = 0.0 # Don't let NaN skew average
            else:
                 loss_item = loss.item()

            total_loss += loss_item
            processed_batches += 1
            pbar.set_postfix({"loss": loss_item})

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1} [Val]   - Avg Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")
    return avg_loss


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer for Time Series Data to Predict Metadata")
    # ... (Argument parsing: ENSURE --output_dim defaults to or is set to 9) ...
    parser.add_argument('--data_dir', type=str, default='./39_Training_Dataset', help='Directory containing the dataset')
    parser.add_argument('--info_filename', type=str, default='train_info.csv', help='Name of the info CSV file')
    parser.add_argument('--ts_dir_name', type=str, default='train_data', help='Name of the subdirectory containing .txt files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_metadata', help='Directory to save checkpoints and plots') # Maybe different dir
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--d_input', type=int, default=6, help='Input feature dimension')
    parser.add_argument('--output_dim', type=int, default=9, help='Output feature dimension (MUST BE 9 for metadata)') # Set default to 9
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=128, help='Feed forward dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of encoder/decoder layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=4000, help='Max sequence length for PE') # Use the increased value
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (reduce if OOM)') # Start lower maybe
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction for validation split')
    # parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps') # Add if needed

    args = parser.parse_args()

    # --- Assert output_dim ---
    if args.output_dim != 9:
        print("ERROR: output_dim MUST be 9 for predicting the 9 metadata features.")
        exit(1)

    # --- Setup ---
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data (using TimeSeriesDataset and universal_collate_fn) ---
    # ... (Keep the data loading and splitting logic from previous version) ...
    print("Loading data...")
    info_path = os.path.join(args.data_dir, args.info_filename)
    try:
        full_dataset = TimeSeriesDataset(data_dir=args.data_dir, info_path=info_path, time_series_dir_name=args.ts_dir_name)
        # Split data
        if args.val_split > 0 and args.val_split < 1:
             total_size = len(full_dataset)
             val_size = int(args.val_split * total_size)
             train_size = total_size - val_size
             if train_size <= 0 or val_size <= 0:
                  print(f"Warning: Invalid split. Using full dataset for both.")
                  train_dataset, val_dataset = full_dataset, full_dataset
             else:
                  print(f"Splitting data: Train={train_size}, Validation={val_size}")
                  train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        else:
             print("Warning: No validation split. Using full dataset for both.")
             train_dataset, val_dataset = full_dataset, full_dataset

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # Use universal_collate_fn from dataloader.py
        from dataloader import universal_collate_fn as collate_fn

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    except FileNotFoundError as e: print(f"Error loading data: {e}"); exit(1)
    except Exception as e: print(f"An error occurred during data loading: {e}"); import traceback; traceback.print_exc(); exit(1)
    print("Data loaded successfully.")


    # --- Build Model ---
    print("Building model...")
    model = build_transformer(
        d_input=args.d_input,
        output_dim=args.output_dim, # Should be 9 now
        N=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.n_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len # Use seq_len here
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Loss and Optimizer ---
    # !! Consider if MSELoss is appropriate for one-hot encoded parts !!
    # Alternatives: BCEWithLogitsLoss (if model outputs logits) or a custom combined loss.
    # Using MSELoss for now as requested by the shape matching.
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    print("Starting training...")
    for epoch in range(args.epochs):
        # Pass scaler to train function
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch, args.grad_clip)
        # Validation doesn't use scaler for backward/step
        val_loss = validate_model(model, val_dataloader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint logic (using best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # ... (save checkpoint code remains the same) ...
            checkpoint_state = {
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss, 'args': args
            }
            ckpt_filename = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}_val_{val_loss:.6f}.pth.tar")
            save_checkpoint(checkpoint_state, filename=ckpt_filename)
            print(f"** Best validation loss improved to {best_val_loss:.6f}. Checkpoint saved. **")
        else:
             print(f"Validation loss did not improve from {best_val_loss:.6f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # --- Plotting ---
    plot_filename = os.path.join(args.checkpoint_dir, "loss_curves_metadata.png") # Different name maybe
    plot_losses(train_losses, val_losses, plot_filename)