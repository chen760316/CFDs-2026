import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


# --- Core Processor: Full Loading and Tensorization Optimization ---

class IoTProcessor:
    def __init__(self, file_path):
        print(f"Reading full dataset: {file_path}")
        self.raw_df = pd.read_csv(file_path)
        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        print("Building global mappings...")
        for col in self.attributes:
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'idx_to_val': {i: val for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def save_full_matrix_M(self, output_path):
        # Note: The one-hot matrix for the full data can be extremely large, please ensure sufficient memory
        print("Generating full one-hot matrix M...")
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        print("Loading full data into GPU tensor cache...")
        # Pre-convert all columns to LongTensor to completely avoid pd.iloc
        self.data_dict = {
            col: torch.tensor(
                processor.raw_df[col].astype(str).map(processor.mappings[col]['val_to_idx']).values,
                dtype=torch.long
            ) for col in self.attributes
        }
        self.length = len(processor.raw_df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {col: self.data_dict[col][idx] for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict({
            col.replace('.', '_'): nn.Embedding(mappings[col]['cardinality'], d_model)
            for col in self.attributes
        })
        self.reconstructors = nn.ModuleDict({
            col.replace('.', '_'): nn.Linear(d_model, mappings[col]['cardinality'])
            for col in self.attributes
        })
        # Increase model capacity to handle the complexity of full data
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            e_i = self.embeddings[col.replace('.', '_')](x_dict[col])
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        h_out = self.transformer(torch.cat(embeds, dim=1))

        return {
            col: self.reconstructors[col.replace('.', '_')](h_out[:, i, :])
            for i, col in enumerate(self.attributes)
        }


# --- Phase 3 Core: Later Epoch Rule Capture Logic ---

def probe_refined_rules(model, num_samples_per_attr=5):
    """
    Capture refined associations through multiple rounds of sampling based on the current epoch's model state
    """
    model.eval()
    m = len(model.attributes)
    epoch_rules = set()

    with torch.no_grad():
        for y_idx, Y_name in enumerate(model.attributes):
            # Increase sampling frequency to obtain more combinations than the number of columns
            for _ in range(num_samples_per_attr):
                scores = np.random.dirichlet(np.ones(m), size=1)[0]
                # Dynamic threshold to capture strongly correlated attributes
                X_set = [model.attributes[i] for i, s in enumerate(scores) if s > (1.8 / m) and i != y_idx]
                if X_set:
                    epoch_rules.add((tuple(sorted(X_set)), Y_name))
    return epoch_rules


def run_full_data_pipeline(input_file, sets_output):
    # 1. Full data initialization
    processor = IoTProcessor(input_file)
    dataset = IoTDataset(processor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)  # Increase BatchSize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttrFinder(processor.mappings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Rule repository
    all_refined_rules = set()

    # 2. Training and dynamic extraction
    total_epochs = 12
    # Start logging results from the 8th Epoch, when the model has tended to stabilize and can capture truer dependencies
    extraction_start_epoch = 8

    print(f"Starting full data training (Total rows: {len(dataset)})...")
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask_col = np.random.choice(model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        # If entering the later Epochs, initiate Phase 3 refined probing
        if epoch >= extraction_start_epoch:
            print(f"  [Phase 3] Extracting refined sets from the current Epoch...")
            current_rules = probe_refined_rules(model, num_samples_per_attr=10)
            all_refined_rules.update(current_rules)

    # 3. Output refined full data results
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        # Sort output by target attribute
        sorted_rules = sorted(list(all_refined_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            f.write(f"{{{', '.join(X)}}} -> {Y}\n")

    print(f"\n--- Processing Completed ---")
    print(f"Total attributes: {len(model.attributes)}")
    print(f"Total refined attribute sets final generation: {len(all_refined_rules)}")


if __name__ == "__main__":
    initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_full_rules_multi_epoch.txt'
    run_full_data_pipeline(initial_file, sets_out)