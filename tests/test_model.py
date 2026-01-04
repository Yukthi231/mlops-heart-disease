from src.inference import load_model
import pytest


def test_model_missing_handled():
    with pytest.raises(FileNotFoundError):
        load_model()
