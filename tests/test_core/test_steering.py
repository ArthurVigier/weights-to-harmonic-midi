import torch
import pytest
from src.core.steering import HallucinationSteerer

def test_steering_similarity_calculation():
    # Set up vectors: 10 hidden dimensions
    hallucination_vector = torch.zeros(10)
    hallucination_vector[0] = 1.0 # First dimension is hallucination

    # Case 1: Activation is exactly the hallucination vector
    steerer = HallucinationSteerer(hallucination_vector, smoothing_factor=0.0) # No smoothing
    activation = torch.zeros(10)
    activation[0] = 1.0
    score = steerer.compute_score(activation)
    assert pytest.approx(score) == 1.0

    # Case 2: Activation is orthogonal to hallucination vector
    activation = torch.zeros(10)
    activation[1] = 1.0 # Second dimension
    score = steerer.compute_score(activation)
    assert pytest.approx(score) == 0.0

    # Case 3: Partial similarity
    activation = torch.zeros(10)
    activation[0] = 0.5
    activation[1] = 0.5
    # Normalize: (0.5/sqrt(0.5), 0.5/sqrt(0.5)) dot (1, 0) = 0.5/0.707 = 0.707
    score = steerer.compute_score(activation)
    assert pytest.approx(score, rel=1e-3) == 0.7071

def test_steering_smoothing():
    hallucination_vector = torch.ones(10)
    steerer = HallucinationSteerer(hallucination_vector, smoothing_factor=0.9)

    # Initial score is 0.0
    activation = torch.ones(10)
    
    # First step: 0.9 * 0 + 0.1 * 1.0 = 0.1
    score = steerer.compute_score(activation)
    assert pytest.approx(score) == 0.1

    # Second step: 0.9 * 0.1 + 0.1 * 1.0 = 0.19
    score = steerer.compute_score(activation)
    assert pytest.approx(score) == 0.19

def test_steering_dimensionality_handling():
    hallucination_vector = torch.randn(10)
    steerer = HallucinationSteerer(hallucination_vector, smoothing_factor=0.0)
    
    # 2D activation: (sequence, hidden_dim)
    activation = torch.randn(5, 10)
    score = steerer.compute_score(activation)
    assert 0.0 <= score <= 1.0

def test_steering_reset():
    hallucination_vector = torch.ones(10)
    steerer = HallucinationSteerer(hallucination_vector, smoothing_factor=0.9)
    steerer.compute_score(torch.ones(10))
    assert steerer._current_score > 0
    
    steerer.reset()
    assert steerer._current_score == 0.0
