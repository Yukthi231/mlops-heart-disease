import pandas as pd
import os
from config import DATA_URL

def load_data():
    df = pd.read_csv(DATA_URL)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/heart.csv", index=False)
    return df

if __name__ == "__main__":
    load_data()
