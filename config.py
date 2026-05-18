# config.py
# ─────────────────────────────────────────────────────────
# Central place for all settings. Change things here only.
# ─────────────────────────────────────────────────────────

# SimFin free API key — get yours at: https://app.simfin.com/login
# Takes 2 minutes, completely free
SIMFIN_API_KEY = "e0bc180b-aa44-4058-ba3b-1830ff76fcaa"

# Where data lives
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

# How many yfinance requests to make before pausing.
BATCH_SIZE = 5

# Seconds to sleep between batches
BATCH_SLEEP = 60
