from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_splits(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    num_features = X.columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features)
        ]
    )

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), preprocessor
