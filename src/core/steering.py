import torch
import torch.nn.functional as F
from typing import Optional

class HallucinationSteerer:
    """
    Projects model activations onto a predefined 'hallucination vector'
    using cosine similarity to generate a normalized hallucination score.
    """
    def __init__(self, hallucination_vector: torch.Tensor, smoothing_factor: float = 0.9):
        self.hallucination_vector = hallucination_vector.detach().cpu()
        self.smoothing_factor = smoothing_factor
        self._current_score: float = 0.0

    def compute_score(self, activation: torch.Tensor) -> float:
        """
        Calculates the cosine similarity and applies exponential smoothing.
        Expects activation to be a (seq_len, hidden_dim) or (hidden_dim,) tensor.
        """
        # Ensure correct shape for cosine similarity
        if activation.dim() > 1:
            # Taking the mean across the sequence dimension for simplicity
            # Alternatively, we could take the last token's activation
            activation = activation.mean(dim=0)

        # Normalize both vectors for cosine similarity calculation
        norm_activation = F.normalize(activation.view(1, -1), p=2, dim=1)
        norm_vector = F.normalize(self.hallucination_vector.view(1, -1), p=2, dim=1)

        # Cosine similarity is in the range [-1, 1]
        similarity = torch.mm(norm_activation, norm_vector.t()).item()

        # Map similarity to a [0, 1] score (assuming positive similarity is hallucination)
        # We clamp to 0 if similarity is negative for this specific application
        raw_score = max(0.0, similarity)

        # Apply exponential smoothing to prevent jitter in audio modulation
        self._current_score = (self.smoothing_factor * self._current_score) + \
                              ((1 - self.smoothing_factor) * raw_score)

        return self._current_score

    def reset(self):
        self._current_score = 0.0
