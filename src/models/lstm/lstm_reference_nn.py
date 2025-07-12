import os
import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AveragePrecision, Precision, Recall, F1Score
from src.components.nn_data_ingestion import SequentialFraudDetectionDataset
from src.components.data_ingestion import DataIngestorFactory, DataIngestorConfig
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # Updated import
from src.components.utils import setup_logger, seed_everything
from src.components.features_engineering import PreprocessorPipeline
from lightning.pytorch.profilers import AdvancedProfiler,SimpleProfiler,PyTorchProfiler
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

logger = setup_logger()

class LSTM_REF(L.LightningModule):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 learning_rate): # Removed pos_weight
        
        super().__init__()
        # Storing hparams manually to be closer to plain PyTorch style
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # We also need to save them for logging purposes if desired
        self.save_hyperparameters()

        # === ARCHITECTURE (Strictly following the reference) ===
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        # The reference uses two separate linear layers with a ReLU in between
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        
        # The reference model explicitly includes a Sigmoid layer at the end
        self.sigmoid = nn.Sigmoid()
        
        # === METRICS ===
        self.ap = AveragePrecision(task="binary")
        # ...

    def forward(self, x):
        # The reference Dataloader produces [batch, features, seq_len].
        # The reference model's FIRST operation is to transpose it.
        # To be faithful, we must assume our input `x` has this shape
        # and we must perform the same transpose.
        # After transpose: [batch, seq_len, features]
        x_transposed = x
        
        # LSTM processes the correctly shaped data
        # representation[1] in their code is the tuple of (h_n, c_n)
        _, (h_n, c_n) = self.lstm(x_transposed)
        
        # h_n shape is [num_layers, batch, hidden_size].
        # Get the hidden state from the final layer. h_n[-1] is a robust way to do this.
        final_hidden_state = h_n[-1]
        
        # Pass through the classification head as defined in the reference
        hidden = self.fc1(final_hidden_state)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        
        # Apply the final sigmoid to get a probability
        probs = self.sigmoid(output)
        
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        # The model now outputs probabilities directly
        probs = self(x).squeeze(1)
        
        # We must use BCELoss because the model outputs probabilities (0-1), not logits.
        loss = F.binary_cross_entropy(probs, y.float())
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ap', self.ap(probs, y.int()), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x).squeeze(1)
        
        loss = F.binary_cross_entropy(probs, y.float())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ap', self.ap(probs, y.int()), prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
if __name__ == "__main__":
    batch_size = 512
    epoch = 20
    patience = 5
    pos_mul = 1.5
    hidden_size = 128
    train_duration=7
    test_duration=7
    seq_len = 5
    fast_dev_run=False
    seed_everything(42)
    
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_data",
        start_train_date="2018-05-15",
        train_duration=train_duration,
        test_duration=test_duration,
        delay=7
    )

    train_preprocessed = PreprocessorPipeline(train_df,add_method=["scale"]).process()
    validation_preprocessed = PreprocessorPipeline(validation_df,add_method=['scale']).process()
    pos_weight = pos_mul * 1/torch.tensor(train_preprocessed[DataIngestorConfig().output_feature].sum()
                              / (len(train_preprocessed) - train_preprocessed[DataIngestorConfig().output_feature].sum()))

    pos = train_df[DataIngestorConfig().output_feature].sum()
    pos_processed = train_preprocessed[DataIngestorConfig().output_feature].sum()

    # Preparing data
    train_data = SequentialFraudDetectionDataset(train_preprocessed, seq_len= seq_len)
    validation_data = SequentialFraudDetectionDataset(validation_preprocessed,seq_len =seq_len)
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory= True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        shuffle=False,
        pin_memory= True

    )
    
    logger.info(f"total pos: {pos}, total pos after process: {pos_processed}")
    logger.info(f"total data: {len(train_df)}, total data after process: {len(train_preprocessed)}")
    logger.info(pos_weight)
    logger.info(len(train_data.config.input_features_transformed) * seq_len)

    #initiating
    run = 1
    log_dir = os.path.join("log", "LSTM_REF_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            logger = TensorBoardLogger(save_dir="log/", name="LSTM_REF_log", version=version)
            profiler = SimpleProfiler(
                        dirpath=log_dir,
                        filename=f"REF_Profiler_run{run}",
                        \
                        )
            break

    model = LSTM_REF(
        input_size=len(train_data.config.input_features_transformed),
        hidden_size=100,      # As in reference
        num_layers=1,         # As in reference
        dropout=0.0,          # As in reference
        learning_rate=1e-4    # As in reference
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
            f"LSTM_REF-{date_str}-run{run}"
            f"bs{batch_size}-ep{epoch}-pat{patience}-{pos_weight}-"
            f"traindur{train_duration}-testdur{test_duration}-hidden_size{hidden_size}"
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
        callbacks=[early_stop_callback,checkpoint_callback],
        profiler= profiler,
        accumulate_grad_batches=4
        
    )


    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
