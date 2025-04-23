from comment_scraper import CommentsUpdatesScraper
from comment_scraper_helper import Scraper
from sentiment_scores import KickstarterSentiment
import datetime
import numpy as np
import joblib
import os

class KickstarterCommentPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KickstarterCommentPredictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_model.pkl")
        self.rf_model = joblib.load(model_path)
        self.comment_scraper = CommentsUpdatesScraper(scraper=Scraper())

    def conversation_chain_stats(self, comments):
        if not comments:
            return {
                "total_chains": 0,
                "avg_chain_length": 0,
                "max_chain_length": 0
            }

        chains = []
        current_chain = []

        for i, comment in enumerate(comments):
            role = 'creator' if comment['is_creator'] else 'user'
            if not current_chain:
                current_chain.append(role)
            else:
                if role != current_chain[-1]:
                    current_chain.append(role)
                else:
                    if len(current_chain) > 1:
                        chains.append(len(current_chain))
                    current_chain = [role]

        if len(current_chain) > 1:
            chains.append(len(current_chain))

        if not chains:
            return {
                "total_chains": 0,
                "avg_chain_length": 0,
                "max_chain_length": 0
            }

        return {
            "total_chains": len(chains),
            "avg_chain_length": sum(chains) / len(chains),
            "max_chain_length": max(chains)
        }

    def get_comment_frequencies(self, comments):
        if not comments:
            return 0.0

        timestamps = [int(c["timestamp"]) for c in comments if not c["is_creator"]]
        if not timestamps:
            return 0.0

        dates = [datetime.utcfromtimestamp(ts).date() for ts in timestamps]
        start_date = min(dates)
        end_date = max(dates)
        total_days = (end_date - start_date).days + 1

        if total_days == 0:
            return float(len(dates))

        return len(dates) / total_days

    def get_prediction(self, url):
        comments = self.comment_scraper.get_comments_texts(url)
        sent_scorer = KickstarterSentiment()
        avg_sentiment_score = np.mean(sent_scorer.get_sentiment_scores(comments))
        chain_stats = self.conversation_chain_stats(comments)
        max_chain_length = chain_stats["max_chain_length"]
        total_chains = chain_stats["total_chains"]
        comment_frequency = self.get_comment_frequencies(comments)
        X_pred = np.array([[avg_sentiment_score, total_chains, max_chain_length, comment_frequency]])
        y_pred_log = self.rf_model.predict(X_pred)
        y_pred = np.expm1(y_pred_log)
        return y_pred