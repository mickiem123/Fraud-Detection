# src/redis_cache.py

import redis
import pandas as pd
from datetime import datetime

def get_or_fetch_history(entity_id: int, entity_type: str, redis_client: redis.Redis, db_df: pd.DataFrame) -> dict:
    """
    Tries to get history from Redis. If miss, falls back to the in-memory DataFrame and warms the cache.
    """
    ts_key = f"{entity_type}:{entity_id}:timestamps"
    
    if redis_client.exists(ts_key):
        # Cache Hit - Fetch from Redis
        timestamps = [float(ts) for ts in redis_client.lrange(ts_key, 0, -1)]
        amounts = [float(amt) for amt in redis_client.lrange(f"{entity_type}:{entity_id}:amounts", 0, -1)]
        frauds = [int(f) for f in redis_client.lrange(f"{entity_type}:{entity_id}:frauds", 0, -1)]
        return {"timestamps": timestamps, "amounts": amounts, "frauds": frauds}

    # Cache Miss - Fallback to in-memory DataFrame
    if db_df is None:
        return {"timestamps": [], "amounts": [], "frauds": []}
    
    print(f"CACHE MISS for {entity_type}:{entity_id}. Querying in-memory DB...")
    
    id_column = f'{entity_type.upper()}_ID'
    entity_history_df = db_df.loc[db_df[id_column] == entity_id].sort_values('TX_DATETIME', ascending=False).head(50)

    if entity_history_df.empty:
        # To prevent future DB lookups for this new entity, we can cache the empty state.
        # This is an optimization.
        pipe = redis_client.pipeline()
        pipe.lpush(ts_key, -1) # Push a placeholder
        pipe.lpush(f"{entity_type}:{entity_id}:amounts", -1)
        pipe.lpush(f"{entity_type}:{entity_id}:frauds", -1)
        pipe.expire(ts_key, 600) # Expire placeholder after 10 mins
        pipe.execute()
        return {"timestamps": [], "amounts": [], "frauds": []}

    # Hydrate Redis cache for next time
    print(f"Hydrating Redis cache for {entity_type}:{entity_id}...")
    pipe = redis_client.pipeline()
    timestamps = [dt.timestamp() for dt in entity_history_df['TX_DATETIME']]
    amounts = entity_history_df['TX_AMOUNT'].tolist()
    frauds = entity_history_df['TX_FRAUD'].tolist()
    
    # LPUSH pushes to the left, so we reverse to maintain chronological order in Redis
    timestamps.reverse()
    amounts.reverse()
    frauds.reverse()
    
    if timestamps:
        pipe.lpush(ts_key, *timestamps)
        pipe.lpush(f"{entity_type}:{entity_id}:amounts", *amounts)
        pipe.lpush(f"{entity_type}:{entity_id}:frauds", *frauds)
        pipe.ltrim(ts_key, 0, 49) # Keep list size fixed
        pipe.ltrim(f"{entity_type}:{entity_id}:amounts", 0, 49)
        pipe.ltrim(f"{entity_type}:{entity_id}:frauds", 0, 49)
        pipe.execute()
    
    # Return the newly fetched history in correct (newest first) order
    timestamps.reverse()
    amounts.reverse()
    frauds.reverse()
    return {"timestamps": timestamps, "amounts": amounts, "frauds": frauds}

def update_entity_state(redis_client: redis.Redis, entity_id: int, entity_type: str, tx_datetime: datetime, tx_amount: float, is_fraud: int):
    """Updates the state of an entity in Redis after a transaction."""
    ts_key = f"{entity_type}:{entity_id}:timestamps"
    amt_key = f"{entity_type}:{entity_id}:amounts"
    fraud_key = f"{entity_type}:{entity_id}:frauds"
    
    pipe = redis_client.pipeline()
    pipe.lpush(ts_key, tx_datetime.timestamp())
    pipe.lpush(amt_key, tx_amount)
    pipe.lpush(fraud_key, is_fraud)
    pipe.ltrim(ts_key, 0, 49)
    pipe.ltrim(amt_key, 0, 49)
    pipe.ltrim(fraud_key, 0, 49)
    pipe.execute()