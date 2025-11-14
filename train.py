import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle, os

DATA_FILE = "movies.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_PKL = os.path.join(MODEL_DIR, "tfidf.pkl")
MATRIX_NPZ = os.path.join(MODEL_DIR, "matrix.npz")
META_PKL = os.path.join(MODEL_DIR, "meta.pkl")

def load_data():
    df = pd.read_csv(DATA_FILE)
    df["genres"] = df["genres"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["text"] = df["genres"] + " " + df["overview"]
    return df

def train():
    df = load_data()
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df["text"])
    sparse.save_npz(MATRIX_NPZ, tfidf_matrix)
    with open(TFIDF_PKL, "wb") as f:
        pickle.dump(tfidf, f)
    meta = {"titles": df["title"].tolist()}
    with open(META_PKL, "wb") as f:
        pickle.dump(meta, f)
    print("Model trained.")

if __name__ == "__main__":
    train()
