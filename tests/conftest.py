import sys
from unittest.mock import MagicMock

# Mock pyo before any other imports
mock_pyo = MagicMock()
sys.modules["pyo"] = mock_pyo

# Define some basic pyo classes that are used in the code
class MockSig:
    def __init__(self, value):
        self.value = value
    def setFreq(self, freq):
        self.value = freq

mock_pyo.Sig.side_effect = lambda value: MockSig(value)
mock_pyo.Server.return_value.boot.return_value = mock_pyo.Server.return_value
mock_pyo.Server.return_value.start.return_value = mock_pyo.Server.return_value
