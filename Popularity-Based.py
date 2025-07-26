# ---------------------------------------------------------------------------
# Streamlit front‑end for the PopularityRecommender (Book‑Crossing datasets)
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_DIR   = Path("data")       # adjust if your files live elsewhere
RATINGS_FP = DATA_DIR / "Ratings.csv"
BOOKS_FP   =  "clustered_outputs/clustered_books_metadata.csv"
USERS_FP   = DATA_DIR / "Users.csv"
M_MIN      = 50

# ---------------- RECOMMENDER CLASS (same as before, trimmed) --------------
class PopularityRecommender:
    def __init__(self, ratings_fp, books_fp, users_fp, m_min=50):
        self.m_min = m_min
        self._load(ratings_fp, books_fp, users_fp)
        self._compute()

    def _load(self, r_fp, b_fp, u_fp):
        self.ratings = (
            pd.read_csv(r_fp, dtype={"ISBN": str, "User-ID": int})
              .query("`Book-Rating` > 0")
        )
        self.books = (
            pd.read_csv(b_fp, dtype={"ISBN": str}, usecols=["ISBN", "Book-Title", "Book-Author"])
              .rename(columns={"Book-Title": "Book_Title",
                               "Book-Author": "Book_Author"})
        )
        self.user_ids = set(pd.read_csv(u_fp, usecols=["User-ID"])["User-ID"])

    def _compute(self):
        global_avg = self.ratings["Book-Rating"].mean()
        agg = (
            self.ratings
              .groupby("ISBN")["Book-Rating"]
              .agg(num_ratings="size", avg_rating="mean")
              .reset_index()
        )
        v, m, R, C = agg["num_ratings"], self.m_min, agg["avg_rating"], global_avg
        agg["score"] = (v / (v + m)) * R + (m / (v + m)) * C

        self.table = (
            agg.merge(self.books, on="ISBN", how="left")
              .dropna(subset=["Book_Title", "Book_Author"])
              .sort_values("score", ascending=False)
              .reset_index(drop=True)
        )

    def top_n(self, n=20):
        return self.table.head(n).copy()

    def recommend_for_user(self, user_id, n=20):
        if user_id not in self.user_ids:
            return self.top_n(n)
        return self.top_n(n)  # pure‑popularity model

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Book Popularity Recommender", layout="wide")
st.title("📚 Popular Books (Global)")

# Sidebar controls
st.sidebar.header("Options")
n = st.sidebar.slider("How many books to display?", min_value=5, max_value=50, value=20, step=5)
custom_user = st.sidebar.text_input("Pretend user‑ID (leave blank for anonymous):", value="")

# Instantiate recommender (cached so it loads only once)
@st.cache_data
def load_recommender():
    return PopularityRecommender(RATINGS_FP, BOOKS_FP, USERS_FP, m_min=M_MIN)

recommender = load_recommender()

# Get recommendations
if custom_user.strip():
    recs = recommender.recommend_for_user(custom_user.strip(), n)
else:
    recs = recommender.top_n(n)

# Show table
st.subheader(f"Top {n} books" + ("" if not custom_user else f" for user {custom_user}"))
st.dataframe(
    recs[["Book_Title", "Book_Author", "num_ratings", "avg_rating", "score"]]
        .rename(columns={
            "Book_Title": "Title",
            "Book_Author": "Author",
            "num_ratings": "#Ratings",
            "avg_rating": "Avg Rating",
            "score": "Bayes Score"}),
    height=600,
    use_container_width=True
)

# Footer
st.caption("Cold‑start handled by Bayesian smoothing (items) and global fallback (users).")
