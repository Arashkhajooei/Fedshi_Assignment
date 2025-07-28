# Fedshi_Assignment

This project builds a **book‑recommendation system** based on the [Book‑Ranking dataset] and explores multiple recommendation strategies. The repository contains data cleaning and enrichment notebooks, simple baseline models, a full graph‑based recommender trained with **LightGCN**, and Streamlit dashboards for interactive exploration.
<p align="center">
  <img src="./overview.png" alt="System Overview" width="600"/>
</p>


---

## 📁 Repository Overview

### 🔍 Data & Notebooks

- `EDA.ipynb`, `Enrichment.ipynb`, `LightGCN.ipynb`, and `NCF_architecture.ipynb` contain exploratory data analysis, data enrichment, and model experimentation.
- The **enrichment notebook** produces metadata files like:
  - `Enriched_Books_Metadata.csv`
  - `Normalized_Enriched_Metadata.csv`
  - `final_normalized_metadata.csv`

These enriched files capture titles, authors, genres, and ratings information.

- Folders like `clustered_outputs/` and `heatmaps/` contain visualizations and clustering results.
- Raw and processed data: `data/Ratings.csv`, `data/Books.csv`.
- Intermediate results live in the root directory.

---
## 🔍 Metadata Enrichment & Clustering Pipeline

To build a smarter recommendation engine beyond user-item ratings, this project enriches book metadata using GPT and organizes it via semantic clustering. Here's how it works:

---

### 📘 Step 1: Metadata Enrichment via GPT

The original Book-Crossing dataset lacked rich metadata. To solve this, we developed a scalable microservice that uses **OpenAI's GPT API** to extract and generate the following fields for each book:

- **Summary** – A concise synopsis of the book
- **Genre** – Literary style (e.g., Thriller, Sci-Fi, Biography)
- **Category** – Broad classification (e.g., Fiction, Non-fiction)
- **Theme** – Narrative motifs (e.g., Love, War, Redemption)
- **Tone** – Emotional flavor (e.g., Dark, Uplifting, Informative)
- **Audience** – Target reader group (e.g., Children, Adults, Academics)

The enrichment process uses:
- ✅ Asynchronous batching for performance
- 🔁 Exponential backoff retries for stability
- 🧩 Auto-alignment to handle missing fields

---

### Step 2: Clustering Enriched Metadata

Once enriched, the metadata is clustered for normalization and analysis:

- We use **SentenceTransformer** (`all-MiniLM-L6-v2`) to embed each field
- **KMeans** clustering is applied independently per metadata column
- Optimal `n_clusters` values are tuned per column (e.g., genre: 20, theme: 25)

Each book receives a cluster label per field:
- `genre_cluster`, `category_cluster`, `theme_cluster`, `tone_cluster`, `audience_cluster`

---

### 🏷️ Step 3: Replacing Clusters with Human-Friendly Labels

To make clusters interpretable:
- We assign each cluster the **most common label** inside that group
- The result is a unified version of `genre`, `category`, `theme`, etc.
- Final output: `final_normalized_metadata.csv` with cleaned, readable fields

---

### 📊 Step 4: Analysis & Visualization

We explore metadata relationships using:
- **Seaborn heatmaps** to show co-occurrence patterns
- **3D semantic surface plots** (e.g., Genre × Audience → Tone)
- **KDE pairplots** to visualize metadata distributions
- Combined heatmaps to assess cluster alignment across fields

---

### 🚀 Use Cases

- Cold-start user handling using semantic groupings
- Metadata-based filtering and faceted browsing
- Enhanced content-based recommendation using unified attributes

---

*See `scripts/` and `notebooks/` for enrichment and clustering code.*


### 🔸 Baseline Models

####  Popularity-Based Recommender
- Script: `Popularity-Based.py`
- Uses `Streamlit` and `pandas` to show top-rated books by frequency.
- A basic benchmark for evaluating advanced models.

####  Collaborative Filtering
- Notebook: `Collaborative-Filtering.ipynb`
- Stub: `Collaborative-Filtering.py`
- Applies matrix factorization and collaborative filtering to the ratings matrix.

---

## 🔗 LightGCN Recommender System

Located in: `lightgcn-recommender/`

### Training Script
- `scripts/train_lightgcn.py`:
  - Loads and splits ratings data.
  - Builds a bipartite interaction graph.
  - Trains a **LightGCN** model with negative sampling.
  - Metrics: `Recall@10`, `NDCG@10`

- Hyperparameters are defined in [`params.yaml`](lightgcn-recommender/params.yaml).

---

### 📦 Data Version Control with DVC
- `dvc.yaml` defines the `train_lightgcn` stage.
  - Dependencies: `data/Ratings.csv`, training script
  - Outputs:
    - `artifacts/embeddings.pt`
    - `models/lightgcn_model.pt`
    - `artifacts/metrics_plot.png`
- `dvc.lock` ensures reproducibility.

---

### 💡 Streamlit Dashboards

#### `scripts/app.py`
- Simple UI for user-based recommendations.
- Loads user embeddings from `artifacts/embeddings.pt`.

#### `scripts/streamlit_dashboard.py`
- Dual-mode UI:
  - User-based recommendation
  - Item-based similarity using cosine similarity on item embeddings
- Displays book info and cover images.

---

### 📊 Evaluation
- Script: `scripts/evaluate.py`
- Saves metrics to `artifacts/metrics.json`
- Customizable for computing `Precision`, `Recall`, `NDCG`, etc.

---

## Other Assets

- 📷 **Images**:
  - `NCF.png`, `output.png`, `output2.png`, `output3.png`, `Colabrating Filtering Result.png`

- 📄 **Metadata CSVs**:
  - Enriched and normalized metadata files are pre-included.
  - Regenerable using the enrichment notebook.

- 🗃️ `.dvc/`, `airflow-dvc/`, `artifacts/`, `models/`:
  - Store trained models, pipeline outputs, and cache (not tracked by Git).

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/Arashkhajooei/Fedshi_Assignment.git
cd Fedshi_Assignment/lightgcn-recommender

# Install dependencies
pip install -r requirements.txt
```

### Fetch Data (if using DVC)

```bash
dvc pull



# Standard Python script
python scripts/train_lightgcn.py
```
### Or via DVC
```bash
dvc repro train_lightgcn

# Full dashboard (user + item similarity)
streamlit run scripts/app.py

```
