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



class LSTM_NN(L.LightningModule):
    def __init__(self, input_size, hidden_size, pos_weight):
        super().__init__()
        # Save only basic types as hyperparameters for logging
        self.save_hyperparameters({"input_size": input_size, "hidden_size": hidden_size, "pos_weight": float(pos_weight)})

        # Convert pos_weight to tensor
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size ),
            nn.Linear(hidden_size , hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 2, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size //2, 1)
        )

        # Metrics
        self.ap = AveragePrecision(task="binary")
        self.recall = Recall(task="binary")
        self.precision = Precision(task="binary")
        self.f1 = F1Score(task="binary")
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        final_hidden_state = h_n[-1]  # Last layer's hidden state
        logits = self.classifier(final_hidden_state)
        return logits

    def training_step(self, batch, _):
        x, y = batch  # x: [batch_size, seq_len, input_size], y: [batch_size]
        logits = self(x).squeeze(1)  # [batch_size]
        
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        # Cast y and preds to int for metrics
        y_int = y.int()
        preds_int = preds.int()
        self.log_dict({
            "train_loss": loss,
            "train_ap": self.ap(probs, y),
            "train_recall": self.recall(preds_int, y_int),
            "train_precision": self.precision(preds_int, y_int),
            "train_f1": self.f1(preds_int, y_int)
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    def on_before_optimizer_step(self, optimizer):
        # Log the L2 norm of gradients to check for vanishing gradients
        # This is called after the .backward() pass and before the optimizer.step()
        if self.trainer.global_step % 25 == 0:  # Log every 25 steps to avoid clutter
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.log('train/grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=True)
    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        # Cast y and preds to int for metrics
        y_int = y.int()
        preds_int = preds.int()
        self.log_dict({
            "val_loss": loss,
            "val_ap": self.ap(probs, y),
            "val_recall": self.recall(preds_int, y_int),
            "val_precision": self.precision(preds_int, y_int),
            "val_f1": self.f1(preds_int, y_int)
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss"
        #     }
        # }
        
if __name__ == "__main__":
    batch_size = 1024
    epoch = 20
    patience = 5
    pos_mul = 1.5
    hidden_size = 128
    train_duration= 54
    test_duration=7
    seq_len = 7
    fast_dev_run=False
    seed_everything(42)
    
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_upsampled_data",
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
    train_data = SequentialFraudDetectionDataset(train_preprocessed, seq_len= seq_len,transformed="agg")
    validation_data = SequentialFraudDetectionDataset(validation_preprocessed,seq_len =seq_len,transformed="agg")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory= True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=8,
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
    log_dir = os.path.join("log", "LSTM_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            logger = TensorBoardLogger(save_dir="log/", name="LSTM_log", version=version)
            profiler = SimpleProfiler(
                        dirpath=log_dir,
                        filename=f"Profiler_run{run}",
                        \
                        )
            break

    model = LSTM_NN(input_size = len(train_data.config.input_features_terminal) , 
                    hidden_size=hidden_size,    
                    pos_weight=pos_weight)
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/LSTM",
        filename=(
            f"LSTM-{date_str}-run{run}"
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
