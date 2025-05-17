from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# import logging
# from tqdm.auto import tqdm
# # 1) Turn on HF logging at INFO or DEBUG
# from transformers import logging as hf_logging
# hf_logging.set_verbosity_info()

class KickstarterSentiment:
    _instance = None  # Class-level singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KickstarterSentiment, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # or "cuda" if available and desired
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        print(self.device)
        print("Loading model and tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def get_sentiment_scores(self, comments, batch_size=8, verbose=False):
        """
        comments: List[str]
        batch_size: how many comments to process per forward pass
        """
        if not comments:
            return np.array([])

        all_scores = []
        n          = len(comments)
        n_batches  = (n + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Sentiment batches", unit="batch"):
            # slice this batch
            batch = comments[batch_idx*batch_size : (batch_idx+1)*batch_size]

            # tokenise
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            # move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # extract logits → positivity index
            logits    = outputs.logits                           # (bs, 2)      
            row_sum = logits.abs().sum(dim=1, keepdim=True)      # (bs, 1)
            normalized_logits = logits / row_sum                 # normalise

            # Positivity index scaled to [0, 1]
            positivity_index = (normalized_logits[:, 1] + 1) / 2

            all_scores.append(positivity_index.cpu().numpy())

        # flatten back to (n,)
        all_scores = np.concatenate(all_scores, axis=0)

        if verbose:
            print()
            for comment, score in zip(comments, all_scores):
                lbl = "POS" if score > 0.5 else "NEG"
                print(f"{lbl} ({score:.3f}): {comment}")

        return all_scores

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