import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from youtube_config import MODEL_DIR, TOKENIZER_NAME, TOP_K, DUMMY_TOKEN

class CampaignPredictor:
    def __init__(self):
        # Use Apple MPS if available, else CPU
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)

        # Load model locally
        model_path = os.path.abspath(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

        self.top_k = TOP_K
        self.dummy = DUMMY_TOKEN

    def predict(self, scored_comments: list):
        """
        Args:
            scored_comments: list of dicts each having a 'score' and 'comment'
        Returns:
            prediction: 'success' or 'fail'
            success_prob: probability of the 'success' class (float)
            confidence: max(prob(success), prob(fail)) (float)
            texts: list of top-K comment texts used
        """
        # 1) Select top-K comments by score
        top = sorted(scored_comments, key=lambda c: c['score'], reverse=True)[:self.top_k]

        # 2) Prepare text inputs
        if not top:
            texts = [self.dummy]
        else:
            texts = [c['comment'].strip() for c in top if c['comment'].strip()]
            if not texts:
                texts = [self.dummy]
        batch_text = " ".join(texts)

        # Tokenize and move to device
        inputs = self.tok(
            batch_text,
            padding='max_length',
            truncation=True,
            max_length=self.tok.model_max_length or 512,
            return_tensors='pt'
        ).to(self.device)

        # 3) Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # 4) Extract metrics
        # Probability of success class (assumes index 1 is 'success')
        success_prob = float(probs[1])
        # Confidence is the highest probability among all classes
        confidence = float(np.max(probs))
        # Predicted label
        label_id = int(np.argmax(probs))
        prediction = 'success' if label_id == 1 else 'failure'

        return prediction, success_prob, confidence, texts
