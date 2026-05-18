import os

from dotenv import load_dotenv


load_dotenv()

SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")


RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"


BATCH_SIZE = 5


BATCH_SLEEP = 60
