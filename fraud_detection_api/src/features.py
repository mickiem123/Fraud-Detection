# src/features.py

import numpy as np
from datetime import datetime
from .config import FEATURE_ORDER

def calculate_realtime_features(history: dict, tx_datetime: datetime, tx_amount: float) -> dict:
    """
    Calculates features for a single entity (customer or terminal) based on its history.

    Args:
        history (dict): A dictionary containing lists of 'timestamps', 'amounts', and 'frauds'.
        tx_datetime (datetime): The datetime of the current transaction.
        tx_amount (float): The amount of the current transaction.

    Returns:
        dict: A dictionary of calculated features for this entity.
    """
    features = {}
    current_ts = tx_datetime.timestamp()

    for days in [1, 7, 30]:
        cutoff = current_ts - (days * 24 * 3600)
        
        # Filter transactions within the window
        window_indices = [i for i, ts in enumerate(history['timestamps']) if ts > cutoff]
        window_amounts = [history['amounts'][i] for i in window_indices]
        window_frauds = [history['frauds'][i] for i in window_indices]

        # Calculate features for window
        features[f"NB_TX_{days}DAY_WINDOW"] = len(window_amounts)
        features[f"AVG_AMOUNT_{days}DAY_WINDOW"] = np.mean(window_amounts) if window_amounts else 0
        features[f"RISK_{days}DAY_WINDOW"] = np.mean(window_frauds) if window_frauds else 0

    return features


def assemble_feature_vector(transaction, cust_features: dict, term_features: dict) -> list:
    """
    Assembles the final, ordered feature vector for the model.
    """
    tx_time = transaction.tx_datetime
    
    feature_dict = {
        'TX_AMOUNT': transaction.tx_amount,
        'TX_DURING_WEEKEND': 1 if tx_time.weekday() >= 5 else 0,
        'TX_DURING_NIGHT': 1 if tx_time.hour <= 6 else 0,
        
        'CUSTOMER_ID_NB_TX_1DAY_WINDOW': cust_features['NB_TX_1DAY_WINDOW'],
        'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW': cust_features['AVG_AMOUNT_1DAY_WINDOW'],
        'CUSTOMER_ID_NB_TX_7DAY_WINDOW': cust_features['NB_TX_7DAY_WINDOW'],
        'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW': cust_features['AVG_AMOUNT_7DAY_WINDOW'],
        'CUSTOMER_ID_NB_TX_30DAY_WINDOW': cust_features['NB_TX_30DAY_WINDOW'],
        'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW': cust_features['AVG_AMOUNT_30DAY_WINDOW'],
        
        'TERMINAL_ID_NB_TX_1DAY_WINDOW': term_features['NB_TX_1DAY_WINDOW'],
        'TERMINAL_ID_RISK_1DAY_WINDOW': term_features['RISK_1DAY_WINDOW'],
        'TERMINAL_ID_NB_TX_7DAY_WINDOW': term_features['NB_TX_7DAY_WINDOW'],
        'TERMINAL_ID_RISK_7DAY_WINDOW': term_features['RISK_7DAY_WINDOW'],
        'TERMINAL_ID_NB_TX_30DAY_WINDOW': term_features['NB_TX_30DAY_WINDOW'],
        'TERMINAL_ID_RISK_30DAY_WINDOW': term_features['RISK_30DAY_WINDOW'],
    }
    
    # Return the feature values in the correct order
    return [feature_dict[feature_name] for feature_name in FEATURE_ORDER]