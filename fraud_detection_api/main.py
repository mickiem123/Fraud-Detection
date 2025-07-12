# main.py

import torch
import joblib
import pandas as pd
import numpy as np
import redis
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from src.model import BaggingCNN_FFNN
from src.config import *
from src.features import calculate_realtime_features, assemble_feature_vector
from src.redis_cache import get_or_fetch_history, update_entity_state
import logging

# ... other imports ...

# --- Configure a basic logger ---
# This will print timestamped logs with levels (INFO, ERROR, etc.) to your console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- API Artifacts & App Initialization ---
api_artifacts = {}
app = FastAPI(title="Fraud Detection API", version="3.0.0", description="A real-time, stateful fraud detection service.")

# --- Data Models (Input/Output Schemas) ---
class TransactionInput(BaseModel):
    transaction_id: str = Field(..., example="TX123456789")
    tx_datetime: datetime = Field(..., example=datetime.now())
    customer_id: int = Field(..., example=1234)
    terminal_id: int = Field(..., example=5678)
    tx_amount: float = Field(..., example=125.75)

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision_threshold: float = 0.5

# --- Startup and Shutdown Events ---
@app.on_event("startup")
def startup_event():
    print("API starting up...")
    
    api_artifacts["model"] = BaggingCNN_FFNN.load_from_checkpoint(

    checkpoint_path=MODEL_PATH,
     map_location="cpu",
    current_tx_feature_size=len(FEATURE_ORDER),
    embedding_dim=EMBEDDING_DIM,          # NEW Hyperparameter: A good starting point. Tune this (e.g., 8, 16, 32).
    cnn_out_channels=CNN_OUT_CHANNELS,      # Number of patterns to learn.
    cnn_kernel_size=CNN_KERNEL_SIZE,         # Size of the pattern window.
    expert_hidden_size=EXPERT_HIDDEN_SIZE,    # Size of the expert networks.
    pos_weight=float(1),
    learning_rate=1e-3
) 

    api_artifacts["model"].eval()
    api_artifacts["model"].freeze()
    
    api_artifacts["scaler"] = joblib.load(SCALER_PATH)
    
    api_artifacts["redis_client"] = redis.from_url(REDIS_URL, decode_responses=True)
    api_artifacts["redis_client"].ping()
    
    try:
        db_df = pd.read_csv(DB_PATH)
        db_df['TX_DATETIME'] = pd.to_datetime(db_df['TX_DATETIME'])
        api_artifacts["db_df"] = db_df
        print(f"Successfully loaded {len(db_df)} historical records into memory.")
    except FileNotFoundError:
        api_artifacts["db_df"] = None
        print(f"WARNING: {DB_PATH} not found. Cache hydration will be skipped.")
    
    print("Startup complete.")

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Fraud Detection API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(transaction: TransactionInput):
    logging.info(f"Received prediction request for transaction_id: {transaction.transaction_id}")
    
    try:
        # --- Stage 1: Get Artifacts ---
        model = api_artifacts["model"]
        scaler = api_artifacts["scaler"]
        redis_client = api_artifacts["redis_client"]
        db_df = api_artifacts["db_df"]
        logging.info("Successfully retrieved artifacts.")
        
        # --- Stage 2: Fetch History ---
        logging.info(f"Fetching history for customer {transaction.customer_id}...")
        cust_history_full = get_or_fetch_history(transaction.customer_id, "customer", redis_client, db_df)
        term_history_full = get_or_fetch_history(transaction.terminal_id, "terminal", redis_client, db_df)
        logging.info(f"History fetched successfully.")
        
        # --- Stage 3: Feature Engineering ---
        logging.info("Calculating real-time features...")
        cust_features = calculate_realtime_features(cust_history_full, transaction.tx_datetime, transaction.tx_amount)
        term_features = calculate_realtime_features(term_history_full, transaction.tx_datetime, transaction.tx_amount)
        feature_vector = assemble_feature_vector(transaction, cust_features, term_features)
        logging.info(f"Feature vector assembled with {len(feature_vector)} features.")
        
        # --- Stage 4: Scaling and Tensor Preparation ---
        logging.info("Scaling features and preparing tensors...")
        scaled_features = scaler.transform(np.array(feature_vector).reshape(1, -1))
        
        cust_fraud_sequence = cust_history_full['frauds'][:C_SEQ_LEN]
        term_fraud_sequence = term_history_full['frauds'][:T_SEQ_LEN]
        cust_hist_seq = np.pad(cust_fraud_sequence, (C_SEQ_LEN - len(cust_fraud_sequence), 0), 'constant')
        term_hist_seq = np.pad(term_fraud_sequence, (T_SEQ_LEN - len(term_fraud_sequence), 0), 'constant')
        
        cust_tensor = torch.tensor(cust_hist_seq, dtype=torch.long).unsqueeze(0)
        term_tensor = torch.tensor(term_hist_seq, dtype=torch.long).unsqueeze(0)
        tx_tensor = torch.tensor(scaled_features, dtype=torch.float)
        logging.info("Tensors created successfully.")
        
        # --- Stage 5: Model Inference (The Critical Step) ---
        logging.info("Running model inference...")
        model_input = (cust_tensor, term_tensor, tx_tensor)
        logits = model(model_input)
        
        # DEBUG: Check the output of the model
        logging.info(f"Model output (logits) type: {type(logits)}, value: {logits}")
        if not isinstance(logits, torch.Tensor):
             raise TypeError(f"Model did not return a tensor! Got {type(logits)} instead.")
             
        probability = torch.sigmoid(logits).item()
        
        # DEBUG: Check the final probability value
        logging.info(f"Calculated probability: {probability:.4f}")

        # --- Stage 6: Decision and State Update ---
        is_fraud = probability > PredictionResponse.model_fields['decision_threshold'].default
        logging.info(f"Prediction made. is_fraud: {is_fraud}")
        
        update_entity_state(redis_client, transaction.customer_id, "customer", transaction.tx_datetime, transaction.tx_amount, int(is_fraud))
        update_entity_state(redis_client, transaction.terminal_id, "terminal", transaction.tx_datetime, transaction.tx_amount, int(is_fraud))
        logging.info("Redis state updated.")
        
        # --- Stage 7: Prepare and Return Response ---
        logging.info("Constructing final API response.")
        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(probability, 4),
            is_fraud=is_fraud
        )
        logging.info("Response constructed successfully.")
        return response

    except Exception as e:
        # If ANY error occurs anywhere in the `try` block, it will be caught here.
        logging.error(f"An unhandled error occurred for transaction_id {transaction.transaction_id}: {e}", exc_info=True)
        # We raise an HTTPException to send a clean 500 error back to the user.
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")