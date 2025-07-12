import os
import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AUROC
from src.components.nn_data_ingestion import BaggingSequentialFraudDetectionDataset

from src.components.data_ingestion import DataIngestorFactory, DataIngestorConfig
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # Updated import
from src.components.utils import setup_logger, seed_everything
from src.components.features_engineering import PreprocessorPipeline
from lightning.pytorch.profilers import SimpleProfiler
import warnings
from datetime import datetime
import torchmetrics
warnings.filterwarnings("ignore")

logger = setup_logger()
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
if __name__ == "__main__":
    batch_size = 512
    epoch = 20
    patience = 5
    pos_mul = 3
    hidden_size = 128
    start_date = "2018-04-01"
    end_train_date = "2018-12-31"
    start_test_date = "2018-08-01"
    end_test_date = "2018-10-01"
    c_seq_len = 7
    t_seq_len = 7
    embedding_dim=16        # NEW Hyperparameter: A good starting point. Tune this (e.g., 8, 16, 32).
    cnn_out_channels=128      # Number of patterns to learn.
    cnn_kernel_size=3        # Size of the pattern window.
    expert_hidden_size=512    
    accumulate_grad_batches=1
    fast_dev_run=False
    config = DataIngestorConfig()

    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, _ = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_upsampled_data_both",
        start_train_date=start_date,
        end_train_date= end_train_date,
        end_test_date= end_test_date,
        start_test_date= start_test_date)
    _, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_data",
        start_train_date=start_date,
        end_train_date= end_train_date,
        end_test_date= end_test_date,
        start_test_date= start_test_date)
    train_df =train_df.sort_values("TX_DATETIME").reset_index(drop=True)
    validation_df =validation_df.sort_values("TX_DATETIME").reset_index(drop=True)




    train_preprocessed = PreprocessorPipeline(train_df,add_method=["scale"]).process()
    validation_preprocessed = PreprocessorPipeline(validation_df,add_method=['scale']).process()
    pos_weight = pos_mul * 1/torch.tensor(train_preprocessed[DataIngestorConfig().output_feature].sum()
                              / (len(train_preprocessed) - train_preprocessed[DataIngestorConfig().output_feature].sum()))

    pos = train_df[DataIngestorConfig().output_feature].sum()
    pos_processed = train_preprocessed[DataIngestorConfig().output_feature].sum()

    # Preparing data
    train_data = BaggingSequentialFraudDetectionDataset(train_preprocessed, t_seq_len= t_seq_len,c_seq_len=c_seq_len,mode="target")
    validation_data = BaggingSequentialFraudDetectionDataset(validation_preprocessed, t_seq_len= t_seq_len,c_seq_len=c_seq_len,mode="target")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        pin_memory= True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=6,
        persistent_workers=True,
        shuffle=False,
        pin_memory= True

    )

    logger.info("-"*10)
    
    logger.info(f"total pos: {pos}, total pos after process: {pos_processed}")
    logger.info(f"total data: {len(train_df)}, total data after process: {len(train_preprocessed)}")
    logger.info(pos_weight)
    
    logger.info("-"*10)
    #initiating
    run = 1
    log_dir = os.path.join("log", "CNN_BAG_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            print(f"----------RUN_{run}----------")
            logger = TensorBoardLogger(save_dir="log/", name="CNN_BAG_log", version=version)
            profiler = SimpleProfiler(
                        dirpath=log_dir,
                        filename=f"A_Profiler_run{run}",
                        \
                        )
            break

    
    model = BaggingCNN_FFNN(
        # The old feature_size parameters are gone
        current_tx_feature_size=len(config.input_features_transformed),
        embedding_dim=embedding_dim,          # NEW Hyperparameter: A good starting point. Tune this (e.g., 8, 16, 32).
        cnn_out_channels=cnn_out_channels,      # Number of patterns to learn.
        cnn_kernel_size=cnn_kernel_size,         # Size of the pattern window.
        expert_hidden_size=expert_hidden_size,    # Size of the expert networks.
        pos_weight=float(pos_weight),
        learning_rate=1e-3
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=(
            f"bagging-{date_str}-run{run}"
            f"bs{batch_size}-ep{epoch}-{pos_weight}-"
            "{epoch:02d}-{val_loss:.2f}"
        ),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )

    trainer = L.Trainer(
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=10,
        max_epochs=epoch,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback,early_stop_callback],
        profiler= profiler,
        accumulate_grad_batches=accumulate_grad_batches
        
    )


    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )