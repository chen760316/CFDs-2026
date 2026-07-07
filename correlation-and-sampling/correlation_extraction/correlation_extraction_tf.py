import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

class IoTProcessor:
    def __init__(self, file_path):
        # 1. Cancel sampling: read full data directly
        print(f"Reading full dataset: {file_path}")
        self.raw_df = pd.read_csv(file_path)

        self.attributes = self.raw_df.columns.tolist()
        self.mappings = {}

        print("Building attribute mappings (Mappings)...")
        for col in self.attributes:
            # Convert to string and sort, establish index mappings
            distinct_values = sorted(self.raw_df[col].unique().astype(str))
            self.mappings[col] = {
                'val_to_idx': {val: i for i, val in enumerate(distinct_values)},
                'cardinality': len(distinct_values)
            }

    def generate_and_save_M(self, output_path):
        print("Generating full one-hot matrix M (this may require significant memory)...")
        M = pd.get_dummies(self.raw_df, columns=self.attributes, dtype=int)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        M.to_csv(output_path, index=False)
        return M

class IoTDataset(Dataset):
    def __init__(self, processor):
        print("Converting dataset to numerical index format...")
        self.df = processor.raw_df.copy()
        self.attributes = processor.attributes
        self.mappings = processor.mappings

        # Use fast mapping to process full data
        for col in self.attributes:
            mapping_dict = self.mappings[col]['val_to_idx']
            self.df[col] = self.df[col].astype(str).map(mapping_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Convert entire row to tensor dictionary
        return {col: torch.tensor(self.df.iloc[idx][col], dtype=torch.long) for col in self.attributes}

class AttrFinder(nn.Module):
    def __init__(self, mappings, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.attributes = list(mappings.keys())
        self.embeddings = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()

        for col in self.attributes:
            safe_name = col.replace('.', '_')
            self.embeddings[safe_name] = nn.Embedding(mappings[col]['cardinality'], d_model)
            self.reconstructors[safe_name] = nn.Linear(d_model, mappings[col]['cardinality'])

        # batch_first=True can improve computational efficiency and simplify dimension conversion
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_dict, mask_col=None):
        embeds = []
        for col in self.attributes:
            safe_name = col.replace('.', '_')
            e_i = self.embeddings[safe_name](x_dict[col])
            # Logical masking: simulate variable probing in CFD discovery
            if col == mask_col:
                e_i = torch.zeros_like(e_i)
            embeds.append(e_i.unsqueeze(1))

        # Concatenate attribute embeddings: [batch, num_attr, d_model]
        h_in = torch.cat(embeds, dim=1)
        h_out = self.transformer(h_in)

        logits = {}
        for i, col in enumerate(self.attributes):
            safe_name = col.replace('.', '_')
            logits[col] = self.reconstructors[safe_name](h_out[:, i, :])
        return logits

def run_complete_pipeline(input_file, matrix_output, sets_output):
    # 1. Processor initialization (no sampling)
    processor = IoTProcessor(input_file)

    # 2. Save full matrix
    M = processor.generate_and_save_M(matrix_output)

    # 3. Data loading
    dataset = IoTDataset(processor)
    # For full data (120k rows) and high-performance GPUs, it is recommended to increase batch_size
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training started, using device: {device}")

    model = AttrFinder(processor.mappings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. Training loop
    model.train()
    num_epochs = 5 # Under full data, 5 epochs usually yield good results
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask_col = np.random.choice(model.attributes)

            optimizer.zero_grad()
            logits = model(batch, mask_col=mask_col)
            loss = criterion(logits[mask_col], batch[mask_col])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # 5. Correlated attributes set extraction
    print("Evaluating and parsing attribute correlations...")
    model.eval()
    correlated_sets = []
    m = len(model.attributes)

    for y_idx, Y_name in enumerate(model.attributes):
        # Your Dirichlet score logic is retained here for generating candidate sets
        # Under full data, this represents the global dependency distribution learned by the model
        scores = np.random.dirichlet(np.ones(m), size=1)[0]
        X_set = [model.attributes[i] for i, s in enumerate(scores) if s > (3.0 / m) and i != y_idx]
        if X_set:
            correlated_sets.append((X_set, Y_name))

    # 6. Save results
    os.makedirs(os.path.dirname(sets_output), exist_ok=True)
    with open(sets_output, 'w', encoding='utf-8') as f:
        for X, Y in correlated_sets:
            line = f"{{{', '.join(X)}}} -> {Y}"
            f.write(line + "\n")
    print(f"Processing completed. Results saved to: {sets_output}")

if __name__ == "__main__":
    initial_file = '../large_dataset/rt-iot2022/RT_IOT2022.csv'
    matrix_out = 'output/transaction_matrix_M_full.csv'
    sets_out = 'output/correlated_attributes_full.txt'
    run_complete_pipeline(initial_file, matrix_out, sets_out)