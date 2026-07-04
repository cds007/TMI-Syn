from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, mode="higher", patience=10, filename="checkpoint.pt"):
        if mode not in {"higher", "lower"}:
            raise ValueError("mode must be 'higher' or 'lower'")
        self.mode = mode
        self.patience = patience
        self.filename = Path(filename)
        self.best_score = None
        self.counter = 0

    def _is_better(self, score):
        if self.best_score is None:
            return True
        if self.mode == "higher":
            return score > self.best_score
        return score < self.best_score

    def step(self, score, model):
        if self._is_better(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            return False

        self.counter += 1
        return self.counter >= self.patience

    def save_checkpoint(self, model):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        state_dict = torch.load(self.filename, map_location="cpu")
        model.load_state_dict(state_dict)
