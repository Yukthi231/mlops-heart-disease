from src.data_loader import load_data

def test_data_loaded():
    df = load_data()
    assert not df.empty
