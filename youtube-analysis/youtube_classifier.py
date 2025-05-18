# classifier.py
import os
import torch
from transformers import pipeline
from youtube_config import ZS_MODEL, ZS_CANDIDATE_LABELS, ZS_BATCH_SIZE

# 1) Limit PyTorch threads to avoid oversubscription
torch.set_num_threads(4)

# 2) Device detection: prefer MPS on Apple Silicon, else CPU
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# for HF pipeline, MPS is device index 0, CPU is -1
PIPELINE_DEVICE = 0 if DEVICE.type == "mps" else -1

# 3) Where your HF cache lives
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

class ZeroShotClassifier:
    def __init__(self):
        self.pipe = None
        self.bs   = ZS_BATCH_SIZE

    def _init_pipe(self):
        """Lazy-init the zero-shot pipeline on first use."""
        self.pipe = pipeline(
            "zero-shot-classification",
            model=ZS_MODEL,
            device=PIPELINE_DEVICE,
            cache_dir=CACHE_DIR,
            local_files_only=True,    # don’t hit the network
        )

    def score_comments(self, comments: list) -> list:
        """
        comments: list of dict { youtube_vid, comment, author }
        returns: same list but each item has a `score` key
        """
        if not comments:
            return []

        if self.pipe is None:
            self._init_pipe()

        scored = []
        total = len(comments)
        for i in range(0, total, self.bs):
            batch = comments[i : i + self.bs]
            seqs  = [c["comment"] for c in batch]

            print(f"[ZeroShot] Scoring comments {i}–{min(i+self.bs, total)} of {total} on {DEVICE}")
            outs = self.pipe(
                sequences=seqs,
                candidate_labels=ZS_CANDIDATE_LABELS,  # <-- fixed
                multi_label=False,
                batch_size=self.bs
            )

            for c, out in zip(batch, outs):
                idx   = out["labels"].index(ZS_CANDIDATE_LABELS[0])
                score = out["scores"][idx]
                scored.append({ **c, "score": float(score) })

        return scored