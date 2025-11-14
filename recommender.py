import pickle, os
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

MODEL_DIR = "models"
TFIDF_PKL = os.path.join(MODEL_DIR, "tfidf.pkl")
MATRIX_NPZ = os.path.join(MODEL_DIR, "matrix.npz")
META_PKL = os.path.join(MODEL_DIR, "meta.pkl")

class Recommender:
    def __init__(self):
        with open(TFIDF_PKL, "rb") as f:
            self.tfidf = pickle.load(f)
        self.matrix = sparse.load_npz(MATRIX_NPZ)
        with open(META_PKL, "rb") as f:
            meta = pickle.load(f)
        self.titles = meta["titles"]

    def get_index(self, title):
        t = title.lower()
        for i, tt in enumerate(self.titles):
            if tt.lower() == t:
                return i
        for i, tt in enumerate(self.titles):
            if t in tt.lower():
                return i
        return None

    def recommend(self, title, top_n=10):
        idx = self.get_index(title)
        if idx is None:
            return []
        sims = linear_kernel(self.matrix[idx], self.matrix).flatten()
        indices = sims.argsort()[::-1]
        indices = [i for i in indices if i != idx][:top_n]
        return [{"title": self.titles[i], "score": float(sims[i])} for i in indices]
