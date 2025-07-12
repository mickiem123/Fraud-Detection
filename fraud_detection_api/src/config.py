import os

BATCH_SIZE = 512
EPOCH = 20
PATIENCE = 5
POS_MUL = 3
HIDDEN_SIZE = 128
C_SEQ_LEN = 7
T_SEQ_LEN = 7
EMBEDDING_DIM = 16        # NEW Hyperparameter: A good starting point. Tune this (e.g., 8, 16, 32).
CNN_OUT_CHANNELS = 128      # Number of patterns to learn.
CNN_KERNEL_SIZE = 3        # Size of the pattern window.
EXPERT_HIDDEN_SIZE = 512
FEATURE_ORDER = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']
# We dynamically calculate this for safety
CURRENT_TX_FEATURE_SIZE = len(FEATURE_ORDER)

# --- Artifact & Data Paths ---
MODEL_PATH = "artifacts/fraud_model.ckpt"
SCALER_PATH = "artifacts/scaler.joblib"
DB_PATH = "data/historical_transactions.csv"

# --- Deployment Settings ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://fraud-redis:6379")