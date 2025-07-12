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



class Attention(nn.Module):
    """
    Implements the Bahdanau-style (contextual) attention.
    This is our "Detective's Interrogation" model.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        # This linear layer helps combine the attended context with the query
        self.linear_out = torch.nn.Linear(dim*2, dim)

    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): The "detective's profile". Shape: [batch, 1, dim]
            context (torch.Tensor): The "witness statements". Shape: [batch, seq_len, dim]
        """
        batch_size = query.size(0)
        hidden_size = query.size(2)
        input_len = context.size(1)

        # 1. Calculate similarity scores:
        # This is the core "interrogation". It uses a dot product (bmm) to compare
        # the query against every item in the context.
        # [b, 1, dim] @ [b, dim, seq_len] -> [b, 1, seq_len]
        attn_scores = torch.bmm(query, context.transpose(1, 2))
        
        # 2. Normalize scores into probabilities (the attention weights)
        # Squeeze, apply softmax, and then unsqueeze back.
        attn_weights = F.softmax(attn_scores.squeeze(1), dim=1).unsqueeze(1) # Shape: [b, 1, seq_len]

        # 3. Create the final context vector by taking a weighted sum of the original context
        # [b, 1, seq_len] @ [b, seq_len, dim] -> [b, 1, dim]
        weighted_context = torch.bmm(attn_weights, context)

        # 4. (Optional but good practice) Combine the query and the weighted context
        combined = torch.cat((weighted_context, query), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        # We return the final combined vector and the weights for interpretability
        return output, attn_weights

class LSTM_A_NN(L.LightningModule):
    def __init__(self, input_size, hidden_size, pos_weight):
        super().__init__()
        self.save_hyperparameters({"input_size": input_size, "hidden_size": hidden_size, "pos_weight": float(pos_weight)})
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        # This linear layer creates the "detective's profile" from the target transaction's features.
        self.context_projector = nn.Linear(input_size, hidden_size)
        
        # --- MODIFIED ---
        # Use our new ContextualAttention class
        self.attention = Attention(hidden_size)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        self.ap = AveragePrecision(task="binary")
        self.recall = Recall(task="binary")
        self.precision = Precision(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        # x arrives with shape [batch, seq_len, features], e.g., [512, 7, 15]

        # --- NEW LOGIC ---
        # 1. Split the data into history (the witnesses) and target (the crime)
        historical_x = x[:, :-1, :]   # All steps EXCEPT the last one -> Shape: [512, 6, 15]
        target_x = x[:, -1, :]        # ONLY the last step's features -> Shape: [512, 15]
        
        # 2. Process the history with the LSTM to get the "witness statements"
        hidden_states, _ = self.lstm(historical_x) # Shape: [512, 6, hidden_size]
        
        # 3. Create the "detective's profile" (the query) from the target transaction
        # We need to add a dummy sequence dimension of 1 for the attention layer
        # target_x.unsqueeze(1) -> Shape: [512, 1, 15]
        query = self.context_projector(target_x.unsqueeze(1)) # Shape: [512, 1, hidden_size]

        # 4. Perform the "interrogation"
        # The attention layer compares the query to the historical hidden_states
        combined_state, attn_weights = self.attention(query=query, context=hidden_states)
        # combined_state shape: [512, 1, hidden_size]

        # 5. Classify the result
        # We must squeeze out the dummy sequence dimension before feeding to the classifier
        final_vector = combined_state.squeeze(1) # Shape: [512, hidden_size]
        logits = self.classifier(final_vector)
        
        return logits

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
        
if __name__ == "__main__":
    batch_size = 512
    epoch = 20
    patience = 5
    pos_mul = 1.5
    hidden_size = 128
    train_duration=35
    test_duration=7
    seq_len = 21
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
    log_dir = os.path.join("log", "LSTM_A_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            logger = TensorBoardLogger(save_dir="log/", name="LSTM_A_log", version=version)
            profiler = SimpleProfiler(
                        dirpath=log_dir,
                        filename=f"A_Profiler_run{run}",
                        \
                        )
            break

    model = LSTM_A_NN(input_size = len(train_data.config.input_features_transformed) , 
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
        dirpath="checkpoints/",
        filename=(
            f"LSTM_A-{date_str}-run{run}"
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
