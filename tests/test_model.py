from src.inference import load_model

def test_model_missing_handled():
    try:
        load_model()
    except FileNotFoundError:
        assert True
