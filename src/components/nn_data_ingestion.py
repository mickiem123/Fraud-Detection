import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from src.components.data_ingestion import DataIngestorFactory,DataIngestorConfig
from src.components.utils import setup_logger
from src.components.exception import CustomException
import os
import pandas as pd
import numpy as np
import sys
import time

logger =setup_logger()
class FraudDetectionDataset(Dataset):
    """
    A custom PyTorch Dataset for fraud detection tasks.
    This dataset wraps a pandas DataFrame and provides access to feature and target tensors
    for use in neural network models. The features and target columns are determined by the
    DataIngestorConfig configuration.
    Args:
        df (pandas.DataFrame): The input DataFrame containing both features and target columns.
        mode (str): transformed or none
    Attributes:
        config (DataIngestorConfig): Configuration object specifying input and output feature columns.
        df (pandas.DataFrame): The original DataFrame.
        features (pandas.DataFrame): DataFrame containing feature columns.
        target (pandas.Series or pandas.DataFrame): Series or DataFrame containing the target variable.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a tuple (features, target) for the given index, both as torch.float32 tensors.
    """

    def __init__(self,df,mode=None):
        self.config = DataIngestorConfig()
        
        self.df = df
        if mode == "transformed":
            self.features = df.loc[:,self.config.input_features_transformed]
        else:
             self.features = df.loc[:,self.config.input_features]
        self.target = df[self.config.output_feature]

    def __len__(self):
        return len(self.df)
    def debug(self):
        print(self.df)
        print(self.features)
        print(self.features.columns)
    def __getitem__(self, index):
        features = torch.tensor(self.features.iloc[index],dtype=torch.float32)
        target = torch.tensor(self.target.iloc[index],dtype=torch.int8)
        return features,target
        
class SequentialFraudDetectionDataset(Dataset):
    def __init__(self,df:pd.DataFrame,seq_len =5,transformed = "agg"):
        try:
            self.df = df.copy()
            self.df.sort_values("TX_DATETIME",inplace=True)
            self.df.reset_index(drop=True,inplace=True)
            self.df["tmp_tx"] = self.df.index
            self.num_samples = len(self.df)
            self.config = DataIngestorConfig()
            

            self.target =  torch.tensor(df.loc[:,self.config.output_feature].values,dtype=torch.int8)
            self.target = torch.hstack([self.target,torch.tensor(0)])
            if transformed == "agg":
                features = self.config.input_features_transformed
            elif transformed == "customer":
                features =  self.config.input_features_customer
            elif transformed == "terminal":
                features =  self.config.input_features_terminal
            else:
                features =  self.config.input_features
            features_cols = torch.tensor(df.loc[:,features].values,dtype=torch.float32)
            padding = torch.zeros(1,features_cols.shape[1],dtype=torch.float32)
            self.features = torch.vstack([features_cols,padding])
            padding_idx = self.num_samples

            if transformed == "terminal":
                by= "TERMINAL_ID"
            else:
                by = "CUSTOMER_ID"
            df_for_indices = pd.DataFrame({
            by: self.df[by],
            'tmp_idx': np.arange(self.num_samples) # Values 0...9999
            })

            df_groupby = df_for_indices.groupby(by)
    
            sequences = {
                # shift(0) is the current transaction, shift(1) is the one before, etc.
                f"tx_step_{i}": df_groupby['tmp_idx'].shift(i)
                for i in range(seq_len - 1, -1, -1)
            }
            sequences_df = pd.DataFrame(sequences)
            self.tx_indices = torch.tensor(sequences_df.fillna(padding_idx).values, dtype=torch.long)
        except Exception as e:
            raise CustomException(e,sys)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        # Use precomputed tensors with gathered indices
        #st = time.time()
        tx_ids = self.tx_indices[index]
        # Gather features for the sequence
        features = self.features[tx_ids]
        target = self.target[index]
        #logger.info(f"time{time.time()-st}")
        return features, target


class BaggingSequentialFraudDetectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, c_seq_len=14,t_seq_len=28,mode=None):
        try:
            self.mode =mode
            self.config = DataIngestorConfig()
            
            self.t_historical_len = t_seq_len - 1
            self.c_historical_len = c_seq_len - 1
            df_source = df.copy().sort_values("TX_DATETIME").reset_index(drop=True)
            self.num_samples = len(df_source)

              # --- 1. Prepare Target and Current Transaction Features ---
            self.targets = torch.tensor(df_source[self.config.output_feature].values, dtype=torch.int8)
            self.targets_for_features = torch.hstack([self.targets,torch.zeros(1,dtype=torch.float32)])
            # The "current" stream gets all transformed features
            self.current_tx_features = torch.tensor(df_source[self.config.input_features_transformed].values, dtype=torch.float32)

            # --- 2. Prepare Customer History Stream ---
            customer_features_tensor = torch.tensor(df_source[self.config.input_features_customer].values, dtype=torch.float32)
            cust_padding = torch.zeros(1, customer_features_tensor.shape[1], dtype=torch.float32)
            self.customer_features_pool = torch.vstack([customer_features_tensor, cust_padding])
            cust_padding_idx = len(self.customer_features_pool) - 1

            # --- 3. Prepare Terminal History Stream ---
            terminal_features_tensor = torch.tensor(df_source[self.config.input_features_terminal].values, dtype=torch.float32)
            term_padding = torch.zeros(1, terminal_features_tensor.shape[1], dtype=torch.float32)
            self.terminal_features_pool = torch.vstack([terminal_features_tensor, term_padding])
            term_padding_idx = len(self.terminal_features_pool) - 1

            # --- 4. Create Index Mappings for Both History Types ---
            df_for_indices = pd.DataFrame({
                'CUSTOMER_ID': df_source['CUSTOMER_ID'],
                'TERMINAL_ID': df_source['TERMINAL_ID'],
                'tmp_idx': np.arange(self.num_samples)
            })

            # Customer history indices (looks back 1 to `historical_len` steps)
            cust_groupby = df_for_indices.groupby('CUSTOMER_ID')
            cust_sequences = {f"tx_{i}": cust_groupby['tmp_idx'].shift(i) for i in range(1, self.c_historical_len + 1)}
            self.customer_indices = torch.tensor(pd.DataFrame(cust_sequences).fillna(cust_padding_idx).values, dtype=torch.long)
            
            # Terminal history indices (looks back 1 to `historical_len` steps)
            term_groupby = df_for_indices.groupby('TERMINAL_ID')
            term_sequences = {f"tx_{i}": term_groupby['tmp_idx'].shift(i) for i in range(1, self.t_historical_len + 1)}
            self.terminal_indices = torch.tensor(pd.DataFrame(term_sequences).fillna(term_padding_idx).values, dtype=torch.long)
            
            logger.info("BaggingSequentialDataset initialized successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # --- Assemble Customer History ---
        cust_hist_indices = self.customer_indices[index]
        customer_history_seq = self.customer_features_pool[cust_hist_indices]

        # --- Assemble Terminal History ---
        term_hist_indices = self.terminal_indices[index]
        terminal_history_seq = self.terminal_features_pool[term_hist_indices]

        # --- Get Current Transaction Info ---
        current_tx = self.current_tx_features[index]

        # --- Get Target Label ---
        target = self.targets[index]
        if not self.mode:
            return (customer_history_seq, terminal_history_seq, current_tx), target
        elif self.mode == "target":
            # Assuming 'customer_history_labels' is a numpy array of 0s and 1s
            customer_history_labels=self.targets_for_features[cust_hist_indices]
            terminal_history_labels=self.targets_for_features[term_hist_indices]
            customer_history_tensor = torch.tensor(customer_history_labels, dtype=torch.long)
            terminal_history_tensor = torch.tensor(terminal_history_labels, dtype=torch.long)
            return (customer_history_tensor,terminal_history_tensor,current_tx),target
        # if self.mode =="eda":
        #     return (customer_history_seq, terminal_history_seq, current_tx), (self.targets[cust_hist_indices],self.targets[term_hist_indices],self.targets[index])

if __name__ == "__main__":
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_data",
        start_train_date="2018-05-15",
        train_duration=7,
        test_duration=7,
        delay=7
    )
    dataset = SequentialFraudDetectionDataset(train_df,seq_len=7)
    for i in range(100):
        dataset[i]