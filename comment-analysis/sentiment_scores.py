from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

class KickstarterSentiment:
    _instance = None  # Class-level singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KickstarterSentiment, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = "cpu"  # or "cuda" if available and desired
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        print("Loading model and tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def get_sentiment_scores(self, comments, verbose=False):
        if not comments:
            return np.array([])

        inputs = self.tokenizer(comments, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        row_sum = logits.abs().sum(dim=1, keepdim=True)
        normalized_logits = logits / row_sum

        # Positivity index scaled to [0, 1]
        positivity_index = (normalized_logits[:, 1] + 1) / 2

        if verbose:
            for i, comment in enumerate(comments):
                pos_score = positivity_index[i].item()
                sentiment = "positive" if pos_score > 0.5 else "negative"
                print(f"Comment: {comment}")
                print(f"  → Sentiment: {sentiment}, Positivity Index: {pos_score:.3f}")

        return positivity_index.cpu().numpy()

    def get_avg_sentiment_for_slug(self, comments_by_slug, slug):
        comments_objs = comments_by_slug.get(slug, [])
        if not comments_objs:
            return None

        comment_texts = [
            c['comment'] for c in comments_objs
            if 'comment' in c and c['comment'].strip() and not c['is_creator']
        ]

        if not comment_texts:
            return None

        scores = self.get_sentiment_scores(comment_texts)
        return float(np.mean(scores)) if len(scores) > 0 else None

