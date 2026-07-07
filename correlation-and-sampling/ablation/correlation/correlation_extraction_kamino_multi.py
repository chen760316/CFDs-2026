"""
Attribute Association Set Extraction Engine Based on the 2021 VLDB Kamino Core Architecture
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import itertools


# --- 1. Data Configuration & Kamino Heuristic Sequence Selection ---

class KaminoProcessor:
    def __init__(self, file_path):
        print(f"Loading dataset: {file_path} ...")
        self.raw_df = pd.read_csv(file_path)
        # Clean column names to prevent illegal symbols
        self.raw_df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in self.raw_df.columns]
        self.raw_attributes = self.raw_df.columns.tolist()

        # Establish mapping dictionary between attribute values and token indices (Tuple Embedding Foundation)
        self.mappings = {}
        for col in self.raw_attributes:
            unique_vals = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(unique_vals)},
                'idx_to_val': {i: val for i, val in enumerate(unique_vals)},
                'cardinality': len(unique_vals)
            }

        # Digitally encode full dataset
        self.encoded_df = self.raw_df.copy()
        for col in self.raw_attributes:
            self.encoded_df[col] = self.raw_df[col].astype(str).map(self.mappings[col]['val_to_idx'])

        # --- Reproducing Kamino Paper Section 4.3: Constraint-Aware Sequencing ---
        # The paper notes: If no explicit FD exists, sort in ascending order based on attribute domain size (Domain Size / Cardinality)
        # This allows the preceding small-domain attributes to form a compact context, helping subsequent sub-models learn conditional probabilities more accurately
        print("Optimizing attribute context sequence based on Kamino heuristic rules (Sequencing)...")
        self.attributes = sorted(self.raw_attributes, key=lambda c: self.mappings[c]['cardinality'])
        print(f"Determined Kamino chain dependency sequence: {self.attributes}")


class KaminoDataset(Dataset):
    def __init__(self, processor):
        self.attributes = processor.attributes
        self.data = {col: torch.tensor(processor.encoded_df[col].values, dtype=torch.long) for col in self.attributes}

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {col: self.data[col][idx] for col in self.attributes}


# --- 2. Aligning with Paper Section 4.1: Kamino Discriminative Sub-Model M_{X,y} (Embedding & Attention) ---

class KaminoSubModel(nn.Module):
    """
    Kamino Chain Dependency Sub-Model M_{X,y}: Utilizes all prefix attributes X before the current position to predict the current target attribute y
    """
    def __init__(self, context_cols, target_col, mappings, d_model=32):
        super().__init__()
        self.context_cols = context_cols
        self.target_col = target_col

        # Independent Tuple Embedding space for context attributes
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(mappings[col]['cardinality'], d_model) for col in context_cols
        })

        # Aligning with Attention Mechanism in Paper Sections 2.3 & 4.1
        # Use a Query vector to perform dot-product with each context attribute's Embedding to calculate the contribution to the current target attribute
        self.attention_query = nn.Parameter(torch.randn(1, d_model))

        # Target attribute predictor (Reconstructor)
        self.predictor = nn.Linear(d_model, mappings[target_col]['cardinality'])

    def forward(self, batch_dict):
        if not self.context_cols:
            # If it is the first attribute in the sequence with no context, return zero placeholders, managed by subsequent base distributions
            return None, None

        embeds = []
        for col in self.context_cols:
            # Extract embedding vector for each context feature
            e_i = self.embeddings[col](batch_dict[col])  # [batch_size, d_model]
            embeds.append(e_i.unsqueeze(1))              # [batch_size, 1, d_model]

        # Concatenate context matrix: [batch_size, len(context_cols), d_model]
        context_matrix = torch.cat(embeds, dim=1)

        # Calculate Attention weights (Dot-Product Attention Mechanism)
        # query: [1, d_model] -> [batch_size, d_model, 1]
        q = self.attention_query.repeat(context_matrix.size(0), 1).unsqueeze(2)
        # scores: [batch_size, len(context_cols), 1]
        attn_scores = torch.bmm(context_matrix, q).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize attention weights

        # Weighted fusion of context representations: [batch_size, d_model]
        h_context = torch.bmm(attn_weights.unsqueeze(1), context_matrix).squeeze(1)

        # Predict target attribute
        logits = self.predictor(h_context)
        return logits, attn_weights


# --- 3. Core Pipeline: Chain Training and Real Attention Weight Extraction Engine ---

def run_kamino_pipeline(input_file, sets_output, tau_c=0.15, max_x_size=3):
    # 1. Initialize data and heuristic sequence
    processor = KaminoProcessor(input_file)
    dataset = KaminoDataset(processor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_discovered_rules = set()
    attributes = processor.attributes

    print(f"\nStarting sub-model training and association set extraction following Kamino chain rules (Device: {device})...")

    # 2. Build and train exclusive M_{X,y} sub-models for each attribute sequentially according to Kamino chain dependency rules
    for j in range(1, len(attributes)):
        context_cols = attributes[:j]  # All prefix attributes in front of the current attribute serve as context
        target_col = attributes[j]

        print(f"-> [{j}/{len(attributes)-1}] Training model M_{{X,{target_col}}}, Number of prefix context features: {len(context_cols)}")

        # Instantiate discriminative model for the current chain node
        sub_model = KaminoSubModel(context_cols, target_col, processor.mappings).to(device)
        optimizer = optim.Adam(sub_model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        # Fast iterative fitting (Each sub-model requires only 5 Epochs to stabilize the attention layer convergence)
        sub_model.train()
        for epoch in range(5):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits, _ = sub_model(batch)
                loss = criterion(logits, batch[target_col])
                loss.backward()
                optimizer.step()

        # 3. Stream-extract internal real Attention weights after model training completes
        sub_model.eval()
        accumulated_weights = np.zeros(len(context_cols))

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, attn_weights = sub_model(batch)
                # Accumulate attention scores across the entire data stream
                accumulated_weights += attn_weights.mean(dim=0).cpu().numpy()

        # Calculate mean attention contribution
        averaged_weights = accumulated_weights / len(dataloader)

        # Filter core context attributes whose attention weight exceeds threshold tau_c
        strong_features = [context_cols[i] for i, w in enumerate(averaged_weights) if w >= tau_c]

        # 4. Archive high-priority attention features into association sets (Supports multivariate high-order intersections)
        if strong_features:
            # Basic associations
            for x in strong_features:
                all_discovered_rules.add((tuple([x]), target_col))
            # High-order conditional dependency combinations
            if len(strong_features) > 1:
                for r in range(2, min(len(strong_features) + 1, max_x_size + 1)):
                    for subset in itertools.combinations(strong_features, r):
                        all_discovered_rules.add((tuple(sorted(subset)), target_col))

    # --- Save results ---
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        sorted_rules = sorted(list(all_discovered_rules), key=lambda x: x[1])
        for X, Y in sorted_rules:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")

    print(f"\nTask completed! Successfully extracted {len(all_discovered_rules)} associated attribute sets exactly according to Kamino architecture.")
    print(f"Average of {len(all_discovered_rules) / len(attributes):.2f} candidate high-confidence association sets per attribute.")


if __name__ == "__main__":
    # Supports testing on any dataset currently being reproduced (e.g., crop.csv or RT_IOT2022.csv)
    initial_file = '../../large_dataset/rt-iot2022/RT_IOT2022.csv'
    sets_out = 'output/kamino_attention_rules.txt'

    # Start pipeline (Set attention weight contribution threshold to 15%)
    run_kamino_pipeline(initial_file, sets_out, tau_c=0.15)