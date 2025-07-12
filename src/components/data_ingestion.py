"""
data_ingestion.py
-----------------
This module provides classes and utilities for ingesting and preparing data for fraud detection tasks.
It supports loading transaction data from files, splitting into train and test sets based on date ranges,
and integrating with PyTorch for deep learning workflows. The design is modular, using abstract base classes
and a factory pattern for extensibility.

Classes:
    - DataIngestorConfig: Configuration for data ingestion, including feature selection and device setup.
    - DataIngestor: Abstract base class for data ingestion strategies.
    - DurationTrainTestDataIngestor: Loads train/test data based on date ranges and file naming.
    - DataIngestorFactory: Factory for creating data ingestors and integrating with PyTorch datasets.

Usage:
    The main block demonstrates how to use the factory to create an ingestor, load data, and convert it for PyTorch usage.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from src.components.utils import setup_logger
from src.components.exception import CustomException
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

logger = setup_logger()

@dataclass
class DataIngestorConfig():
    """
    Configuration for data ingestion.

    Attributes:
        output_feature (str): Name of the target variable for fraud detection.
        input_features_transformed (list of str): List of transformed feature names  to be used as model inputs.
        input_features(list of str): List of feature names to be used as model inputs.


    """

    output_feature="TX_FRAUD"

    input_features_transformed=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']

    input_features = ["TRANSACTION_ID",		"CUSTOMER_ID",	"TERMINAL_ID",	"TX_AMOUNT",
                      "TX_TIME_SECONDS",	"TX_TIME_DAYS",]
    
    input_features_customer = [ 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
    input_features_terminal = ['TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']
class DataIngestor(ABC):
    """
    Abstract base class for data ingestion strategies.

    Methods:
        ingest(path): Abstract method to load data from a given path.
    """
    
    
    @abstractmethod
    def ingest(self,path)-> pd.DataFrame:
        pass

class DurationTrainTestDataIngestor(DataIngestor):
    """
    Loads training and testing data based on date ranges from a directory of files.

    Methods:
        ingest(dir_path, *, start_train_date, train_duration, delay, test_duration, ...):
            Loads and splits data into train and test sets based on date logic.
        get_files(files_list):
            Loads and concatenates pickled DataFrames from a list of file names.

    Parameters:
        dir_path (str): Directory containing data files.
        start_train_date (str): Start date for training data.
        train_duration (int): Number of days for training data.
        delay (int): Gap between train and test periods.
        test_duration (int): Number of days for test data.
        end_train_date, end_test_date, start_test_date (str, optional):
            Optional overrides for date logic.

    Returns:
        (train_df, test_df): Tuple of DataFrames for train and test sets.
    """
    def ingest(self,dir_path,*, start_train_date=None, train_duration=None,delay=None,test_duration=None,
               end_train_date = None,end_test_date = None,start_test_date = None) -> pd.DataFrame:
        try:
            train_file = []
            test_file =[]
            self.dir_path = dir_path
            
            try:
                if not end_train_date:
                    end_train_date = pd.Timestamp(start_train_date) + pd.Timedelta(days=train_duration)

                if not start_test_date:
                    start_test_date = pd.Timestamp(end_train_date) + pd.Timedelta(days=delay)

                if not end_test_date:
                    end_test_date = pd.Timestamp(start_test_date) + pd.Timedelta(days=test_duration)
                    
            except Exception as e:
                logger.info(f"train_duration is {train_duration}")
                logger.info(f"test_duration is {test_duration}")

            if isinstance(train_duration,str) :
                for file in os.listdir(dir_path):
                    file,self.file_extension = os.path.splitext(file)
                    train_file.append(file)
            elif  isinstance(test_duration,str) :
                for file in os.listdir(dir_path):
                    file,self.file_extension = os.path.splitext(file)
                    test_file.append(file)       
            else:
                for file in os.listdir(dir_path):
                    file,self.file_extension = os.path.splitext(file)

                    if pd.Timestamp(start_train_date) <= pd.Timestamp(file)<= pd.Timestamp(end_train_date):
                        train_file.append(file)

                    if pd.Timestamp(start_test_date) <= pd.Timestamp(file)<= pd.Timestamp(end_test_date):
                        test_file.append(file)        
            
            #logger.info("Loading file into Dataframe")
            logger.info(f"Train: {start_train_date} - {end_train_date}")
            train_df = self.get_files(train_file)
            logger.info(f"Test: {start_test_date} - {end_test_date}")
            test_df = self.get_files(test_file)
            
            if train_df is None or train_df.empty:
                logger.warning("train_df is empty or None")
            if test_df is None or test_df.empty:
                logger.warning("test_df is empty or None")

            return train_df,test_df
        except Exception as e:
            raise CustomException(e,sys)

            
    def get_files(self,files_list):
        """
        Loads and concatenates pickled DataFrames from a list of file names.

        Parameters:
            files_list (list of str): List of file names (without extension).

        Returns:
            pd.DataFrame: Concatenated DataFrame from all files.
        """
        files_list = pd.Series(files_list).map(lambda x: os.path.join(self.dir_path,f"{x}.pkl"))
        ##logger.info(files_list)
        # Efficiently load all DataFrames and concatenate at once
        dfs = [pd.read_pickle(file) for file in files_list]
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()


class DataIngestorFactory():
    """
    Factory for creating data ingestors and integrating with PyTorch datasets.

    Methods:
        create_ingestor(data_type):
            Returns an instance of a DataIngestor subclass based on data_type.
        cuda_intergrate(df_train, df_test):
            Converts DataFrames to PyTorch datasets for CUDA/CPU usage.
    """
    @staticmethod
    def create_ingestor(data_type):
        """ types include :['duration_pkl']"""
        try:
               
            if data_type == "duration_pkl":
                return DurationTrainTestDataIngestor()
            
        except Exception as e:
            raise CustomException(e,sys)

        
if __name__ == "__main__":
    print(os.getcwd())
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")

    train_df,test_df = ingestor.ingest(
        rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data",
        start_train_date="2018-04-01",
        train_duration=1,
        delay=5,
        test_duration=1)
    train_df,test_df = factory.cuda_intergrate(train_df,test_df)
    print(train_df,test_df)
    #artifact_dir = os.path.join(os.getcwd(), "artifact")
    #os.makedirs(artifact_dir, exist_ok=True)

    #train_data.to_csv(os.path.join(artifact_dir, "train_data.csv"), index=False)
    #test_data.to_csv(os.path.join(artifact_dir, "test_data.csv"), index=False)
    
    ##logger.info(f"Exported train and test data to {artifact_dir}")
    