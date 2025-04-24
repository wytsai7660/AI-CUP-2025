# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import re
import glob
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# --- Helper function to parse cut_point string ---
def parse_cut_points(cut_point_str):
    """Parses the '[ 0 61 ...]' string into a list of integers."""
    if not isinstance(cut_point_str, str):
        # print(f"Warning: cut_point is not a string ({type(cut_point_str)}), cannot parse.")
        return None
    try:
        # Remove brackets, strip whitespace, split by space, filter empty strings, convert to int
        points = [int(p) for p in cut_point_str.strip('[] ').split() if p]
        if not points: return None
        return points
    except Exception as e:
        print(f"Error parsing cut_point string '{cut_point_str}': {e}")
        return None

# --- Training/Validation Dataset with Segmentation ---
class TimeSeriesDataset(Dataset):
    """
    Dataset for loading training and validation data.
    Reads time series files, segments them based on cut_point in info_df,
    and processes metadata. Ignores the last segment.
    """
    def __init__(self, data_dir, info_path, time_series_dir_name="train_data"):
        super().__init__()
        self.time_series_base_path = os.path.join(data_dir, time_series_dir_name)
        if not os.path.isdir(self.time_series_base_path):
            raise FileNotFoundError(f"Error: Time series data directory not found: {self.time_series_base_path}")

        try:
            self.info_df = pd.read_csv(info_path)
            # Ensure necessary columns exist
            required_cols = ['unique_id', 'cut_point', 'gender', 'hold racket handed', 'play years', 'level']
            if not all(col in self.info_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in self.info_df.columns]
                raise ValueError(f"Error: Info CSV file '{info_path}' is missing required columns: {missing}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Info CSV file not found: {info_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading or validating info CSV file '{info_path}': {e}")

        # Define expected categories for one-hot encoding
        self.play_years_categories = [0, 1, 2]
        self.level_categories = [2, 3, 4, 5]
        self.metadata_output_dim = 1 + 1 + len(self.play_years_categories) + len(self.level_categories) # Should be 9

        # Pre-process to create segment information
        self.segment_info = []
        print("Preprocessing dataset: Creating segments from cut points (ignoring last segment)...")
        skipped_files_count = 0
        skipped_segments_count = 0

        for metadata_index, row in tqdm(self.info_df.iterrows(), total=len(self.info_df), desc="Processing CSV rows"):
            try:
                unique_id = row['unique_id']
                cut_points = parse_cut_points(row['cut_point'])
                txt_filename = f"{unique_id}.txt"
                txt_path = os.path.join(self.time_series_base_path, txt_filename)

                if not os.path.exists(txt_path):
                    if skipped_files_count < 5: # Limit warnings
                         print(f"Warning: File {txt_path} (listed in CSV) not found. Skipping all segments for unique_id {unique_id}.")
                    elif skipped_files_count == 5:
                         print("Further file not found warnings will be suppressed...")
                    skipped_files_count += 1
                    continue

                # Need at least 2 cut points to define a segment to keep (start and end)
                if cut_points is None or len(cut_points) < 2:
                    skipped_segments_count += (len(cut_points) if cut_points else 1) # Count potential segments skipped
                    continue

                cut_points = sorted(list(set(cut_points))) # Ensure sorted and unique

                # Iterate up to len(cut_points) - 1 to ignore the last cut point as a start
                num_segments_in_file = 0
                for i in range(len(cut_points) - 1):
                    start_index = cut_points[i]
                    end_index = cut_points[i+1] # Exclusive end index

                    # Basic validation: start must be before end
                    if start_index < 0 or start_index >= end_index:
                        # print(f"Warning: Invalid segment range [{start_index}:{end_index}] for unique_id {unique_id}. Skipping segment.")
                        skipped_segments_count += 1
                        continue

                    self.segment_info.append({
                        'file_path': txt_path,
                        'start': start_index,
                        'end': end_index,
                        'metadata_index': metadata_index
                    })
                    num_segments_in_file += 1
                if num_segments_in_file == 0 and len(cut_points) >= 2:
                     skipped_segments_count += 1 # Count the file where all segments were invalid

            except Exception as e:
                print(f"Error processing row {metadata_index} (unique_id: {row.get('unique_id', 'N/A')}): {e}")
                continue # Skip to next row on error

        if skipped_files_count > 0:
            print(f"Warning: Skipped {skipped_files_count} unique_ids due to missing .txt files.")
        if skipped_segments_count > 0:
            print(f"Warning: Skipped {skipped_segments_count} potential segments due to invalid cut points or ranges.")
        print(f"Preprocessing complete. Total valid segments created: {len(self.segment_info)}")
        if len(self.segment_info) == 0:
             print("CRITICAL WARNING: Dataset preprocessing yielded zero valid segments! Check data integrity and cut points.")

    def __len__(self):
        return len(self.segment_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if not (0 <= idx < len(self.segment_info)):
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.segment_info)}")

        segment_data_info = self.segment_info[idx]
        file_path = segment_data_info['file_path']
        start_index = segment_data_info['start']
        end_index = segment_data_info['end']
        metadata_index = segment_data_info['metadata_index']

        # Load the original time series data file
        try:
            # Consider caching loaded files if I/O becomes a bottleneck and memory allows
            full_time_series_data = np.loadtxt(file_path, dtype=np.float32)

            # Handle potential edge cases for the loaded array
            if full_time_series_data.ndim == 0: # File contains only one number?
                return None # Invalid segment data
            elif full_time_series_data.ndim == 1:
                if len(full_time_series_data) == 6: full_time_series_data = np.expand_dims(full_time_series_data, axis=0)
                else: return None # Incorrect feature count in 1D file
            elif full_time_series_data.shape[1] != 6: # Check features if > 1D
                return None # Incorrect feature count
            elif len(full_time_series_data) == 0: # Empty file after loading
                 return None

        except Exception as e:
            print(f"Error loading file {file_path} in __getitem__: {e}")
            return None # Signal collate_fn to skip

        # Slice the segment
        file_length = len(full_time_series_data)
        # Adjust indices safely
        safe_start = max(0, start_index)
        safe_end = min(end_index, file_length)

        # Check if indices are valid *after* adjustment
        if safe_start >= safe_end:
            # print(f"Warning: Segment indices [{start_index}:{end_index}] resulted in empty slice for {os.path.basename(file_path)}. Skipping.")
            return None # Segment is empty or invalid

        segment_array = full_time_series_data[safe_start:safe_end, :]

        # Final check if segment is somehow empty
        if segment_array.shape[0] == 0:
            return None

        time_series_tensor = torch.from_numpy(segment_array.copy()) # Use copy to avoid potential numpy issues

        # Get and process metadata
        try:
            metadata_row = self.info_df.iloc[metadata_index]
            gender = float(metadata_row['gender'])
            handed = float(metadata_row['hold racket handed'])
            play_years_val = metadata_row['play years']
            level_val = metadata_row['level']

            # Handle potential NaN or non-numeric values safely
            play_years = 0 # Default
            if pd.notna(play_years_val):
                try: play_years = int(play_years_val)
                except ValueError: pass # Keep default if conversion fails

            level = 2 # Default
            if pd.notna(level_val):
                try: level = int(level_val)
                except ValueError: pass # Keep default

            # One-hot encode
            play_years_onehot = [0.0] * len(self.play_years_categories)
            if play_years in self.play_years_categories: play_years_onehot[self.play_years_categories.index(play_years)] = 1.0

            level_onehot = [0.0] * len(self.level_categories)
            if level in self.level_categories: level_onehot[self.level_categories.index(level)] = 1.0

            metadata_features = [gender, handed] + play_years_onehot + level_onehot
            metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32)
            if metadata_tensor.shape[0] != self.metadata_output_dim:
                 raise ValueError("Internal Error: Metadata tensor dimension mismatch.")

        except Exception as e:
             print(f"Error processing metadata for row index {metadata_index} (file: {os.path.basename(file_path)}): {e}")
             return None # Cannot proceed without metadata

        return time_series_tensor, metadata_tensor

# --- Test Dataset ---
class TestTimeSeriesDataset(Dataset):
    """
    Dataset for loading test data. Reads .txt files and returns
    the time series data along with the filename.
    """
    def __init__(self, test_data_dir):
        super().__init__()
        self.test_data_dir = test_data_dir
        if not os.path.isdir(test_data_dir):
             raise FileNotFoundError(f"Error: Test data directory not found: {test_data_dir}")
        self.file_paths = glob.glob(os.path.join(test_data_dir, '*.txt'))
        if not self.file_paths:
            print(f"Warning: No .txt files found in test directory: {test_data_dir}")
        # Sort numerically by base filename
        self.file_paths.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if not (0 <= idx < len(self.file_paths)):
            raise IndexError(f"Index {idx} out of bounds for test dataset size {len(self.file_paths)}")

        txt_path = self.file_paths[idx]
        filename = os.path.basename(txt_path)
        try:
            time_series_data = np.loadtxt(txt_path, dtype=np.float32)
            # Handle edge cases
            if time_series_data.ndim == 0: return None
            elif time_series_data.ndim == 1:
                if len(time_series_data) == 6: time_series_data = np.expand_dims(time_series_data, axis=0)
                else: return None
            elif time_series_data.shape[1] != 6: return None
            elif len(time_series_data) == 0: return None
        except Exception as e:
            print(f"Error loading test file {txt_path}: {e}")
            return None # Signal collate_fn to skip

        time_series_tensor = torch.from_numpy(time_series_data.copy())
        return time_series_tensor, filename

# --- Universal Collate Function ---
def universal_collate_fn(batch):
    """
    Collates batches from TimeSeriesDataset (ts, meta) or
    TestTimeSeriesDataset (ts, filename). Handles padding and filtering.
    """
    # Filter out None items from __getitem__ and items with empty tensors
    original_batch_size = len(batch)
    batch = [item for item in batch if item is not None and item[0] is not None and item[0].nelement() > 0]
    filtered_count = original_batch_size - len(batch)
    # if filtered_count > 0:
    #     print(f"Collate Warning: Filtered out {filtered_count} invalid samples from batch.")

    if not batch:
        return None, None, None, None # Return all Nones if batch is empty after filtering

    # Determine batch type (metadata or filename)
    has_metadata = isinstance(batch[0][1], torch.Tensor)
    has_filename = isinstance(batch[0][1], str)

    time_series_list = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch] if has_metadata else None
    filename_list = [item[1] for item in batch] if has_filename else None

    # Pad time series sequences
    try:
        lengths = torch.tensor([len(ts) for ts in time_series_list], dtype=torch.long)
        padded_time_series = pad_sequence(time_series_list, batch_first=True, padding_value=0.0)
    except Exception as e:
        print(f"Error during padding in collate_fn: {e}")
        # Try to identify problematic sequence lengths
        print(f"Sequence lengths in failed batch: {[len(ts) for ts in time_series_list]}")
        return None, None, None, None # Cannot proceed

    # Stack metadata if present
    stacked_metadata = None
    if has_metadata:
        try:
            stacked_metadata = torch.stack(metadata_list, dim=0)
            # Sanity check shape
            if stacked_metadata.shape[0] != len(batch):
                 print(f"Collate Warning: Metadata stack size mismatch ({stacked_metadata.shape[0]} vs {len(batch)})")
                 # Decide how to handle: return partial data or fail
                 # Returning partial data here:
                 # return padded_time_series, None, lengths, None
        except Exception as e:
            print(f"Error stacking metadata in collate_fn: {e}")
            # Optionally check shapes of individual metadata tensors
            # for i, m in enumerate(metadata_list): print(f" Meta {i} shape: {m.shape}")
            # Decide how to handle: return partial data or fail
            # Returning partial data here:
            # return padded_time_series, None, lengths, None


    # Return the appropriate tuple structure
    if has_metadata:
        return padded_time_series, stacked_metadata, lengths, None
    elif has_filename:
        return padded_time_series, None, lengths, filename_list
    else:
        # This case should ideally not be reached if datasets return tuples
        print("Collate Warning: Unknown batch item structure.")
        return padded_time_series, None, lengths, None