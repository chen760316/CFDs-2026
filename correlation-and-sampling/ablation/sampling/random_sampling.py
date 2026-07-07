"""
Random Sampling Baseline
"""
import pandas as pd
import numpy as np
import os
import csv


class RandomSamplingProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # Read only the header to get attribute names
        self.attributes = pd.read_csv(file_path, nrows=0).columns.tolist()

    def run_random_sampling(self, N_bound=8400):
        """
        Stream-scans the file to get the total row count, then performs random sampling without replacement with extremely low memory usage
        """
        print(f"Starting random sampling process (Target N={N_bound})...")

        # 1. First pass: only count the total rows in the file (excluding header)
        print("Scanning full data to calculate total row count...")
        total_rows = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for idx, _ in enumerate(reader):
                total_rows += 1
                if idx % 500000 == 0 and idx > 0:
                    print(f"Scanned {idx} rows...")

        print(f"Total rows in dataset (excluding header): {total_rows}")

        if total_rows <= N_bound:
            print("Warning: Total rows are less than or equal to the target sampling bound, full data will be returned.")
            return pd.read_csv(self.file_path)

        # 2. Randomly select row indices (sampling without replacement)
        print(f"Randomly selecting {N_bound} row indices from 0 to {total_rows - 1}...")
        # Use a set to speed up subsequent index matching and lookups
        S_indices = set(np.random.choice(total_rows, size=N_bound, replace=False))

        # 3. Second pass: stream-extract data based on selected random indices
        print(f"Extracting chosen {len(S_indices)} rows of data from original file...")
        result_rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in S_indices:
                    result_rows.append(row)
                if len(result_rows) == len(S_indices):
                    break  # Break early once enough samples are collected to save time

        sampled_df = pd.DataFrame(result_rows)
        print(f"Random sampling completed.")
        return sampled_df


if __name__ == "__main__":
    input_csv = '../../large_dataset/crop.csv'
    output_csv = 'output/random_sampling_baseline.csv'
    os.makedirs('output', exist_ok=True)

    # Initialize processor (does not load data)
    processor = RandomSamplingProcessor(input_csv)

    # Execute Random Sampling Baseline (Target 8400 rows)
    final_sampled_data = processor.run_random_sampling(N_bound=8400)

    # Save final results
    final_sampled_data.to_csv(output_csv, index=False)
    print(f"Random sampling Baseline dataset has been saved to: {output_csv}")