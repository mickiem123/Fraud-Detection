# src/utils.py

import logging
import sys
import os
from datetime import datetime
import random
import numpy as np
import torch

def setup_logger():
    """Set up a logger for the project with log file stored in log/ folder."""
    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)

    # Log file name as datetime
    log_filename = datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("fraud_detection_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(file_formatter)

    # Avoid adding multiple handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    


