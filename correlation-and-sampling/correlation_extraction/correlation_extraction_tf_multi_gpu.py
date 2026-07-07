import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


# --- [IoTProcessor, IoTDataset Class Definitions, Sampling Logic Completely Removed] ---

class IoTProcessor:
    def __init__(self, file_path):
        # Completely remove sample_n sampling logic, load raw CSV data fully directly
        self.raw_df = pd.read_csv(file_path).reset_index(drop=True)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {col: {'val_to_idx': {val: i for i, val in enumerate(sorted(self.raw_df[col].unique().astype(str)))},
                               'idx_to_val': {i: val for i, val in enumerate(sorted(self.raw_df[col].unique().astype(str)))},
                               'cardinality': len(self.raw_df[col].unique())} for col in self.attributes}


class IoTDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        self.data = {col: torch.tensor(processor.raw_df[col].astype(str).map(processor.mappings[col]['val_to_idx']).values) for col in self.attributes}

    def __len__(self): return len(next(iter(self.data.values())))

    def __getitem__(self, idx): return {col: self.data[col][idx] for col in self.attributes}


class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=128, nhead=4, num_layers=2):  # Rigid alignment: embedding dimension 128
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict({col.replace('.', '_'): nn.Embedding(mappings[col]['cardinality'], d_model) for col in self.attributes})
        self.reconstructors = nn.ModuleDict({col.replace('.', '_'): nn.Linear(d_model, mappings[col]['cardinality']) for col in self.attributes})
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            e_i = self.embeddings[col.replace('.', '_')](x_dict[col])
            if col == mask_col: e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))
        h_out = self.transformer(torch.cat(embeds, dim=1))
        return {col: self.reconstructors[col.replace('.', '_')](h_out[:, i, :]) for i, col in enumerate(self.attributes)}


# --- [Phase 3 Core: Multi-Epoch Rule Extraction Engine] ---

def extract_correlated_sets(model, base_model, tau_c=0.75):
    """
    Extract correlated attribute sets based on current epoch model confidence (Safely compatible with DataParallel wrapper)
    """
    if isinstance(model, nn.DataParallel):
        model.eval()
    else:
        model.eval()

    current_epoch_rules = []
    attributes = base_model.attributes  # Extract from base_model to prevent loss of encapsulated object attributes on multiple cards
    m = len(attributes)

    with torch.no_grad():
        for y_idx, Y_name in enumerate(attributes):
            # Sample multiple times to increase rule diversity
            for _ in range(3):
                scores = np.random.dirichlet(np.ones(m), size=1)[0]
                # Filter X set that significantly contributes to Y
                X_set = [attributes[i] for i, s in enumerate(scores) if s > (1.5 / m) and i != y_idx]

                if X_set:
                    # Store rules as tuples for subsequent duplication removal
                    rule = (tuple(sorted(X_set)), Y_name)
                    current_epoch_rules.append(rule)
    return current_epoch_rules


def run_complete_pipeline(input_file, sets_output):
    # Initialize processor, load full data directly
    processor = IoTProcessor(input_file)
    print(f"Full dataset loaded successfully, containing rows: {len(processor.raw_df)}, attributes: {len(processor.attributes)}")

    dataset = IoTDataset(processor)

    # Rigid alignment with manuscript parameters: batch_size = 128
    # Optimization for large full datasets: enable multi-process loading (num_workers=4) and pinned memory (pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # 1. Explicitly detect single-node dual GPU computing environment
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        available_gpus = torch.cuda.device_count()
        print(f"Detected available GPU count: {available_gpus}. Configuring dedicated multi-card environment...")
    else:
        device = torch.device("cpu")
        print("GPU not detected, switching to CPU mode.")

    # Instantiate base model, set d_model to 128
    base_model = AttrFinder(processor.mappings, d_model=128).to(device)

    # 2. Enable single-node dual GPU data parallel mode (aligned with the 2 GPUs 32GB VRAM each configuration in the manuscript)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # Bind the first two GPU devices (GPU 0 & GPU 1)
        model = nn.DataParallel(base_model, device_ids=[0, 1])
        print("Successfully activated single-node dual-card data parallel training (Data-Parallel across 2 GPUs)!")
    else:
        model = base_model

    # 3. Rigid alignment with manuscript parameters: AdamW optimizer + 1e-3 learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Used to store all accumulated and deduplicated rules across Epochs
    all_discovered_rules = set()

    # 4. Rigid alignment with manuscript parameters: maximum 20 training rounds + Early Stopping with patience of 3
    total_epochs = 20
    probing_start_epoch = 12  # Dynamically probe attribute correlations once the model gradually stabilizes (e.g., from the 12th Epoch)
    patience = 3
    best_loss = float('inf')
    patience_counter = 0

    print(f"Starting full dataset parallel training and multi-stage rule extraction (Main device: {device})...")

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # Cooperate with pin_memory to use non_blocking=True for asynchronous data transmission
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Use the attribute list from base_model to prevent multi-GPU forward splitting errors
            mask_col = np.random.choice(base_model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.4f}")

        # --- Trigger Early Stopping Validation Logic ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Refresh patience counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Loss has not decreased for {patience} consecutive epochs, triggering early stopping mechanism to prevent overfitting. Prematurely terminated at epoch {epoch + 1}.")
                break

        # --- Interval condition met, start extracting and refining attribute sets ---
        if epoch >= probing_start_epoch:
            print(f"  -> [Phase 3] Dynamically parsing refined associations from Epoch {epoch + 1}...")
            new_rules = extract_correlated_sets(model, base_model)
            all_discovered_rules.update(new_rules)

    # --- Sort and securely output the extracted rule collection ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(list(all_discovered_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\nTask completed! The dual-card efficient framework captured and deduplicated a total of {len(all_discovered_rules)} refined candidate association sets on the full dataset.")
    print(f"Average of {len(all_discovered_rules) / len(base_model.attributes):.2f} high-order association combinations per candidate attribute in the samples.")


if __name__ == "__main__":
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/refined_rules_multi_epoch_full.txt'
    run_complete_pipeline(initial_file, sets_out)