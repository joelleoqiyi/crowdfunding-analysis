from youtube_client import YouTubeClient
from youtube_classifier import ZeroShotClassifier
from youtube_predictor import CampaignPredictor
from googleapiclient.errors import HttpError

# Instantiate clients once
yt_client = YouTubeClient()
zs_classifier = ZeroShotClassifier()
campaign_predictor = CampaignPredictor()

def run_pipeline(campaign_name: str) -> dict:
    """
    1) Scrape YouTube comments for `campaign_name`
    2) Zero-shot score each comment
    3) Predict campaign success

    Returns a dict with keys:
      - campaign_name: str
      - prediction: "success" or "fail"
      - probability: float

    Raises HttpError on YouTube API errors.
    """
    # 1) Fetch comments
    try:
        comments = yt_client.scrape_comments(campaign_name)
        print("Comments here", comments)
    except HttpError:
        # Propagate for upstream handling
        raise

    print(len(comments))

    # 2) Score comments
    scored = zs_classifier.score_comments(comments)
    print("Scoring", scored)

    # 3) Predict success/fail
    prediction, success_prob, confidence, texts = campaign_predictor.predict(scored)
    print("Prediction", prediction)
    print("Probability", success_prob)

    return {
        "prediction": prediction,
        "probability": success_prob,
        "confidence": confidence,
        "comments": scored
    }
