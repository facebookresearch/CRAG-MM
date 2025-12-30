"""
Configuration constants for CRAG-MM evaluation.

This module centralizes all configuration parameters used throughout the evaluation process,
including model settings, batch sizes, token limits, and dataset constants.
"""

import os

# Model Configuration
DEFAULT_EVAL_MODEL = "gpt-4o"
MAX_API_RETRIES = 3
DEFAULT_NUM_WORKERS = 1

# Batch Size Configuration
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = int(
    os.environ.get("MAX_BATCH_SIZE", "16")
)  # Default to 16, configurable via environment

# Token Limits
MAX_RESPONSE_LENGTH_IN_TOKENS = 75
MAX_EVAL_TOKENS = 1024

# Dataset Constants - Sessions to Skip
SESSIONS_TO_SKIP = {
    "single-turn": [
        "04d98259-27af-41b1-a7be-5798fd1b8e95",
        "695b4b5c-7c65-4f7b-8968-50fe10482a16",
    ],
    "multi-turn": [
        "04d98259-27af-41b1-a7be-5798fd1b8e95",
        "695b4b5c-7c65-4f7b-8968-50fe10482a16",
    ],
}

# Dataset Constants - Sessions to Update
SESSIONS_TO_UPDATE = {
    "single-turn": {
        "0df99a9a-6a5d-4ff4-b8e4-e3ee90209dfe": {
            "image_url": "https://web.archive.org/web/20250521025904/https://upload.wikimedia.org/wikipedia/commons/1/13/BMW_5_SERIES_LWB_SEDAN_(G30)_China_(82).jpg"
        }
    }
}
