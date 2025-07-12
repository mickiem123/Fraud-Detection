
import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F


import warnings
import torchmetrics
warnings.filterwarnings("ignore")


# A small, reusable "expert" network
class ExpertFFNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            # Output a single logit representing the "fraud score" from this expert
            nn.Linear(hidden_size, 1) 
        )
    def forward(self, x):
        return self.net(x)
    
class SequenceCNN(nn.Module):
    """
    A 1D CNN module to process sequences. It takes a sequence, finds patterns,
    and uses global max pooling to create a fixed-size embedding.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3):
        """
        Args:
            in_channels (int): The number of features for each item in the sequence.
            out_channels (int): The number of filters (patterns) the CNN will learn.
            kernel_size (int): The width of the pattern-detecting window (e.g., 3 means it looks at 3 transactions at a time).
        """
        super().__init__()
        # 1D Convolutional layer. `padding='same'` ensures the sequence length
        # doesn't shrink, which simplifies things.
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same' 
        )
        self.relu = nn.ReLU()
        
        # Global Max Pooling: takes the feature map from the conv layer and
        # finds the maximum value for each filter across the whole sequence.
        # Output shape will be (batch_size, out_channels, 1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # Input x is expected to be (batch, features, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.global_max_pool(x)
        # Squeeze the last dimension to get (batch, out_channels)
        return x.squeeze(-1)

class BaggingCNN_FFNN(L.LightningModule):

    def __init__(self,
                 # customer_feature_size & terminal_feature_size are REMOVED
                 current_tx_feature_size: int,
                 embedding_dim: int,       # NEW: The size of the vector for 'fraud' and 'non-fraud'
                 cnn_out_channels: int,
                 cnn_kernel_size: int,
                 expert_hidden_size: int,
                 pos_weight,
                 learning_rate: float = 1e-3):
        
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

        # === EMBEDDING LAYERS ===
        # num_embeddings=2 because we have two categories: 0 (non-fraud) and 1 (fraud)
        self.customer_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        self.terminal_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        # === STREAM 1: Customer History CNN ===
        # The number of input channels for the CNN is now the embedding dimension
        self.customer_cnn = SequenceCNN(
            in_channels=embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size
        )
        self.customer_expert = ExpertFFNN(cnn_out_channels, expert_hidden_size)

        # === STREAM 2: Terminal History CNN ===
        self.terminal_cnn = SequenceCNN(
            in_channels=embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size
        )
        self.terminal_expert = ExpertFFNN(cnn_out_channels, expert_hidden_size)
        
        # === STREAM 3: Current Transaction FFNN (unchanged) ===
        self.current_tx_expert = ExpertFFNN(current_tx_feature_size, expert_hidden_size)

        # === NEW: THE GATING NETWORK ===
        # It takes the rich embeddings from the CNNs and the raw TX features as input
        # to decide how to weigh the experts.
        gate_input_size = (cnn_out_channels * 2) + current_tx_feature_size
        
        self.gating_network = nn.Sequential(

            nn.Linear(gate_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # Outputs 3 weights, one for each expert
            nn.Sigmoid()  # Ensures weights sum to 1, making them a trust distribution
        )
            

    def forward(self, x):
        customer_history, terminal_history, current_tx = x

        # --- 1. Get Embeddings and Logits from each stream ---
        cust_embedded = self.customer_embedding(customer_history).permute(0, 2, 1)
        cust_cnn_out = self.customer_cnn(cust_embedded)
        customer_logit = self.customer_expert(cust_cnn_out)

        term_embedded = self.terminal_embedding(terminal_history).permute(0, 2, 1)
        term_cnn_out = self.terminal_cnn(term_embedded)
        terminal_logit = self.terminal_expert(term_cnn_out)

        current_tx_logit = self.current_tx_expert(current_tx)

        # --- 2. Use the Gating Network to calculate trust weights ---
        # Concatenate the EVIDENCE (the embeddings, not the logits)
        gate_input = torch.cat([cust_cnn_out, term_cnn_out, current_tx], dim=1)
        # The gating network outputs a trust distribution, e.g., [0.1, 0.8, 0.1]
        expert_weights = self.gating_network(gate_input) # Shape: (batch_size, 3)

        # --- 3. Calculate the final logit ---
        # Concatenate the OPINIONS (the logits)
        expert_logits = torch.cat([customer_logit, terminal_logit, current_tx_logit], dim=1) # Shape: (batch_size, 3)
        
        # Calculate the final logit as a weighted sum of the expert opinions
        # This is element-wise multiplication, then sum across the dimension
        final_logits = torch.sum(expert_weights * expert_logits, dim=1, keepdim=True)
        
        return final_logits

    def training_step(self, batch, batch_idx=None):
        inputs, y = batch
        logits = self(inputs).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        recall = torchmetrics.functional.recall(preds, y, task="binary")
        precision = torchmetrics.functional.precision(preds, y, task="binary")
        auroc = torchmetrics.functional.auroc(probs, y, task="binary")

        self.log_dict(
            {'train_loss': loss, 'train_recall': recall, 'train_precision': precision, 'train_auroc': auroc},
            on_epoch=True, prog_bar=True,on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx=None):
        inputs, y = batch
        logits = self(inputs).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        recall = torchmetrics.functional.recall(preds, y, task="binary")
        precision = torchmetrics.functional.precision(preds, y, task="binary")
        auroc = torchmetrics.functional.auroc(probs, y, task="binary")

        self.log_dict(
            {'val_loss': loss, 'val_recall': recall, 'val_precision': precision, 'val_auroc': auroc},
            on_epoch=True, prog_bar=True,on_step=False
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Add scheduler if desired
        return optimizer
