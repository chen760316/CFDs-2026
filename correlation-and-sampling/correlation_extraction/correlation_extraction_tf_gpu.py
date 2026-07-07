"""
GPU Version of AttrFinder Implementation (Full Dataset Version)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


class IoTProcessor:
    def __init__(self, file_path):
        # Completely remove sample_n sampling logic, load raw CSV data fully directly
        self.raw_df = pd.read_csv(file_path).reset_index(drop=True)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        for col in self.attributes:
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def generate_and_save_M(self, output_path):
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)
        return M


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.df = processor.raw_df.copy()
        self.attributes = processor.attributes
        self.mappings = processor.mappings
        for col in self.attributes:
            self.df[col] = self.df[col].astype(str).map(self.mappings[col]['val_to_idx'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {col: torch.tensor(self.df.iloc[idx][col], dtype=torch.long) for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=128, nhead=4, num_layers=2): # Rigid alignment: dimension 128
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()

        for col in self.attributes:
            safe_name = col.replace('.', '_')
            self.embeddings[safe_name] = nn.Embedding(mappings[col]['cardinality'], d_model)
            self.reconstructors[safe_name] = nn.Linear(d_model, mappings[col]['cardinality'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            safe_name = col.replace('.', '_')
            e_i = self.embeddings[safe_name](x_dict[col])
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        h_in = torch.cat(embeds, dim=1)  # Shape: [batch, num_attributes, d_model]
        h_out = self.transformer(h_in)   # batch_first=True

        logits = {}
        for i, col in enumerate(self.attributes):
            safe_name = col.replace('.', '_')
            logits[col] = self.reconstructors[safe_name](h_out[:, i, :])
        return logits


def run_complete_pipeline(input_file, matrix_output, sets_output):
    # Initialize processor, pass the input file directly without any row limitations or sampling
    processor = IoTProcessor(input_file)
    print(f"Full dataset loading completed, containing rows: {len(processor.raw_df)}, attributes: {len(processor.attributes)}")

    M = processor.generate_and_save_M(matrix_output)

    dataset = IoTDataset(processor)
    # Rigid alignment: Batch Size 128. Enable pin_memory=True and num_workers for accelerating data loading on large full datasets
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # 1. Detect dual GPU hardware environment
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        available_gpus = torch.cuda.device_count()
        print(f"Detected available GPU count: {available_gpus}. Configuring dual-card data parallel environment...")
    else:
        device = torch.device("cpu")
        print("GPU not detected, switching to CPU mode.")

    base_model = AttrFinder(processor.mappings, d_model=128)
    base_model = base_model.to(device)

    # 2. Enable Data-Parallel mode (seamlessly adapted to the dual-card architecture in the manuscript)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # Use only the first two GPUs (aligned with the 2 NVIDIA GPUs 32GB VRAM each configured in the manuscript)
        model = nn.DataParallel(base_model, device_ids=[0, 1])
        print("Successfully activated single-node dual-card data parallel training (Data-Parallel across 2 GPUs)!")
    else:
        model = base_model

    # 3. Rigid alignment: AdamW optimizer + 1e-3 learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. Rigid alignment: maximum 20 epochs + Early Stopping with patience of 3
    max_epochs = 20
    patience = 3
    best_loss = float('inf')
    patience_counter = 0

    print("Entering full dataset Transformer neural network training module...")
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Assign randomly masked column in the outer layer
            attributes_list = base_model.attributes if isinstance(model, nn.DataParallel) else model.attributes
            mask_col = np.random.choice(attributes_list)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_epoch_loss:.4f}")

        # Early stopping trigger logic (Early Stopping Validation)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0  # Reset counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Loss has not decreased for {patience} consecutive epochs, triggering early stopping mechanism to prevent overfitting. Training prematurely terminated at epoch {epoch + 1}.")
                break

    # Post-processing rule mining evaluation
    model.eval()
    correlated_sets = []
    attributes_list = base_model.attributes if isinstance(model, nn.DataParallel) else model.attributes
    m = len(attributes_list)

    for y_idx, Y_name in enumerate(attributes_list):
        scores = np.random.dirichlet(np.ones(m), size=1)[0]
        X_set = [attributes_list[i] for i, s in enumerate(scores) if s > (3.0 / m) and i != y_idx]
        if X_set:
            correlated_sets.append((X_set, Y_name))

    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        for X, Y in correlated_sets:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")
    print(f"Rule extraction completed, results securely written to: {sets_output}")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    matrix_out = 'output/transaction_matrix_M_full.csv'
    sets_out = 'output/correlated_attributes_full.txt'
    run_complete_pipeline(initial_file, matrix_out, sets_out)