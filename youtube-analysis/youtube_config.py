# config.py
import os

# the folder this file lives in
BASE_DIR = os.path.dirname(__file__)

# YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Zero-shot classifier
ZS_MODEL            = "facebook/bart-large-mnli"
# ZS_MODEL = "valhalla/distilbart-mnli-12-1" // can use this if the CPU/GPU/MPS not enough ie model too big
ZS_CANDIDATE_LABELS = ["about campaign", "not about campaign"]
ZS_BATCH_SIZE       = int(os.getenv("ZS_BATCH_SIZE", "16"))

# Campaign success predictor
# point MODEL_DIR at the real folder on disk
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(BASE_DIR, "best_campaign_model")
)
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "distilbert-base-uncased")
TOP_K          = int(os.getenv("TOP_K", "50"))
DUMMY_TOKEN    = "[NO_COMMENTS]"