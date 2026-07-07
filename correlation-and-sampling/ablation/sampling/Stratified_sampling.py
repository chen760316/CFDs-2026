"""
Crop Dataset Stratified Sampling Test (Stratified Sampling Baseline)
"""
import pandas as pd
import numpy as np
import os
import csv

class CropStratifiedSamplingProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # Read only the header to get attribute names
        self.attributes = pd.read_csv(file_path, nrows=0).columns.tolist()
        print(f"Successfully read dataset header, containing attributes: {self.attributes}")

    def run_stratified_sampling(self, N_bound=8400, stratify_column='label'):
        """
        Stream-scans the file to perform proportional stratified sampling based on the crop category tag (stratify_column).
        :param N_bound: Target total rows for final sampling (8400)
        :param stratify_column: The attribute column name used for stratification (usually 'label' in crop datasets)
        """
        print(f"Starting stream stratified sampling process (Target N={N_bound}, Stratify by='{stratify_column}')...")

        if stratify_column not in self.attributes:
            raise ValueError(f"Error: The specified classification column '{stratify_column}' does not exist in the crop dataset! Please check the header.")

        # 1. First pass: count the distribution of crop categories and record the global row indices corresponding to each category
        print("Performing the first scan: counting crop category distributions and categorizing indices...")
        category_indices = {}
        total_rows = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                cat_val = row[stratify_column]
                if cat_val not in category_indices:
                    category_indices[cat_val] = []
                category_indices[cat_val].append(idx)
                total_rows += 1

                if idx % 100000 == 0 and idx > 0:
                    print(f"Scanned {idx} rows...")

        print(f"Total rows in dataset: {total_rows}, detected number of crop categories: {len(category_indices)}")

        # If the total row count is already less than or equal to the target sampling bound of 8400, return full data directly
        if total_rows <= N_bound:
            print("Notice: Total rows in dataset are less than or equal to the target sampling bound, full data will be returned.")
            return pd.read_csv(self.file_path)

        # 2. Calculate the number of samples to extract for each crop type (proportional allocation)
        print("Calculating sampling quotas for each crop...")
        S_indices = set()
        allocated_total = 0

        # Sort categories in ascending order based on their sample size
        sorted_categories = sorted(category_indices.items(), key=lambda x: len(x[1]))

        for cat_val, indices in sorted_categories:
            cat_size = len(indices)
            # Calculate the theoretical quota for the current category: (crop total count / dataset total count) * 8400
            quota = max(1, round((cat_size / total_rows) * N_bound))
            quota = min(quota, cat_size)

            # If adding the current quota exceeds the total limit, truncate the quota
            if allocated_total + quota > N_bound:
                quota = N_bound - allocated_total

            if quota > 0:
                # Randomly sample from this crop's index pool without replacement
                chosen_ids = np.random.choice(indices, size=quota, replace=False)
                S_indices.update(chosen_ids)
                allocated_total += len(chosen_ids)
                print(f"  - Crop [{cat_val}]: original size={cat_size}, allocated sampling quota={quota}")

        # If final sample count is short of 8400 due to rounding errors, fill the gap from the largest category with the most samples
        if len(S_indices) < N_bound:
            residual = N_bound - len(S_indices)
            largest_cat_indices = sorted_categories[-1][1]
            remaining_pool = [uid for uid in largest_cat_indices if uid not in S_indices]
            if len(remaining_pool) >= residual:
                extra_chosen = np.random.choice(remaining_pool, size=residual, replace=False)
                S_indices.update(extra_chosen)

        # 3. Second pass: stream-extract data based on selected indices
        print(f"Performing the second scan on original file: extracting chosen {len(S_indices)} rows of crop tuples...")
        result_rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in S_indices:
                    result_rows.append(row)
                if len(result_rows) == len(S_indices):
                    break  # Break early to perfectly optimize performance

        sampled_df = pd.DataFrame(result_rows)
        print(f"Crop dataset stratified sampling completed.")
        return sampled_df

if __name__ == "__main__":
    # Physically switch to crop dataset path
    input_csv = '../../large_dataset/crop.csv'
    output_csv = 'output/crop_stratified_sampled.csv'
    os.makedirs('output', exist_ok=True)

    # Initialize processor
    processor = CropStratifiedSamplingProcessor(input_csv)

    # Execute stratified sampling (Target 8400 tuples, default metric is crop label 'label')
    # If the crop class column name in your csv is different (e.g., 'crop_type' or 'class'), you can modify it here accordingly
    final_sampled_data = processor.run_stratified_sampling(N_bound=8400, stratify_column='label')

    # Save final results
    final_sampled_data.to_csv(output_csv, index=False)
    print(f"Representative stratified sampled dataset has been successfully saved to: {output_csv}")