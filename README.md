# Fandom Overlap Project

This repository collects, cleans, and analyzes YouTube and Bluesky data to study cross-platform fandom audience overlap and per-user IP affinity. It also includes a collaborative filtering model on YouTube interactions and a Streamlit dashboard that surfaces the results.

## Stack

- Python (managed via `uv`)
- `polars` and `duckdb` for data wrangling and analytical storage
- `scikit-learn` truncated SVD for collaborative filtering
- `matplotlib` and `seaborn` for visualization
- Jupyter for EDA and modeling
- `streamlit` for the dashboard

## Setup

1. Copy `.env.example` to `.env` and fill in credentials.
2. Install dependencies:

```bash
uv sync
```

3. Validate the credentials file:

```bash
uv run python src/check_setup.py
```

> Credentials are only required if you want to re-run the collectors. To explore the existing data, dashboard, or notebooks you can skip the API setup and just download the data archives below.

## Get The Data

The project has two data tiers, neither of which lives in regular git:

| Tier | Location | Size | How to fetch |
|---|---|---|---|
| Processed parquets + CF artifacts (`data/processed/`) | **Git LFS** in this repo | ~156 MB | Comes down automatically on `git clone` when `git-lfs` is installed |
| Raw API payloads (`data/raw/`) | **GitHub Release asset** | 482 MB zipped (~3.5 GB unzipped) | Download manually from the Releases page (link below) |

### 1. Clone the repo (with Git LFS)

```bash
# install git-lfs once if you don't have it
sudo apt install git-lfs            # Debian / Ubuntu / WSL
brew install git-lfs                # macOS
git lfs install                     # one-time per machine

git clone git@github.com:dmikaxyz/cis2450-fandom-big-data-project.git
cd cis2450-fandom-big-data-project
```

If a `data/processed/*.parquet` opens as a small text "pointer" file, your clone didn't fetch LFS blobs. Run:

```bash
git lfs pull
```

### 2. Download the raw data archive

The raw API payloads are too big for git, so they're attached to the project's GitHub Release.

- **Releases page:** <https://github.com/dmikaxyz/cis2450-fandom-big-data-project/releases/latest>
- **Direct download:** [`fandom-project-raw-data.zip`](https://github.com/dmikaxyz/cis2450-fandom-big-data-project/releases/latest/download/fandom-project-raw-data.zip) (482 MB)
- **SHA256:** `7bf32e6e9f3dd138c0d1ac9d72c60ba1bf5a9a83077da9a5203646b19d7b7e03`

From the repo root:

```bash
# (optional) verify integrity
sha256sum fandom-project-raw-data.zip

# extract into data/raw/ (the zip preserves the folder structure)
unzip fandom-project-raw-data.zip
```

After this you should have populated `data/raw/youtube_runs/`, `data/raw/bluesky_runs/`, and the top-level YouTube/Bluesky JSON files.

### 3. (Optional) Rebuild the DuckDB warehouse

`data/duckdb/fandom.duckdb` is a build artifact and is **not** distributed. Once the parquets are in place, regenerate it with:

```bash
uv run python src/load_duckdb.py
```

## API Setup

### YouTube

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Open `APIs & Services` -> `Library`.
4. Enable `YouTube Data API v3`.
5. Open `APIs & Services` -> `Credentials`.
6. Click `Create credentials` -> `API key`.
7. Copy the value into `.env` as `YOUTUBE_API_KEY=...`.

### Bluesky

1. Sign in at [bsky.app](https://bsky.app/) with the account you want to use.
2. Open `Settings` -> `App passwords` and create one.
3. Add to `.env`:

```bash
BLUESKY_HANDLE=your-handle.bsky.social
BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

## Data Flow

```
Collectors → raw JSON in data/raw/
Builders   → cleaned Parquet in data/processed/
load_duckdb → DuckDB views in data/duckdb/fandom.duckdb
Notebooks  → EDA + modeling
Dashboard  → Streamlit pages
```

Key scripts in `src/`:

- `collect_youtube.py` — searches each configured fandom query, caches the top 50 results per query, and saves metadata for qualifying videos.
- `collect_youtube_comments.py` and `collect_youtube_comments_round_robin.py` — fetch top-level comments for selected videos.
- `build_youtube_video_tables.py` and `build_youtube_comment_tables.py` — turn a saved `youtube_run_*` directory into normalized Parquet tables (`youtube_videos`, `youtube_video_query_map`, `youtube_query_results`, `youtube_comments`, `youtube_users`, `youtube_user_video_map`, `youtube_user_ip_activity`).
- `collect_bluesky.py` — searches each configured Bluesky query and writes a `bluesky_run_*` directory of raw API JSON.
- `build_bluesky_tables.py` — turns a saved `bluesky_run_*` directory into normalized Parquet tables (`bluesky_posts`, `bluesky_users`, `bluesky_user_ip_activity`).
- `load_duckdb.py` — registers every available processed Parquet file as a DuckDB view in `data/duckdb/fandom.duckdb`.

IP catalog and per-platform queries are defined in `src/ip_config.py`.

## Run The YouTube Pipeline

```bash
uv run python src/collect_youtube.py --min-comment-count 100
uv run python src/build_youtube_video_tables.py --run-dir data/raw/youtube_runs/youtube_run_<RUN_ID>
uv run python src/collect_youtube_comments_round_robin.py --run-dir data/raw/youtube_runs/youtube_run_<RUN_ID> --max-videos 24
uv run python src/build_youtube_comment_tables.py --run-dir data/raw/youtube_runs/youtube_run_<RUN_ID>
uv run python src/load_duckdb.py
```

The round-robin collector walks ordered `selected_videos` lists per IP and cycles across IPs until it hits `--max-videos`, giving broad coverage across fandoms instead of exhausting quota on one IP.

## Run The Bluesky Pipeline

```bash
uv run python src/collect_bluesky.py --max-pages-per-query 10 --days-back 30
uv run python src/build_bluesky_tables.py --run-dir data/raw/bluesky_runs/bluesky_run_<RUN_ID>
uv run python src/load_duckdb.py
```

## Notebooks

- `notebooks/eda_with_duckdb_and_polars.ipynb` — YouTube EDA: per-IP stats, shared commenter overlap heatmap, top cross-IP videos, top cross-IP video pairs.
- `notebooks/eda_with_duckdb_and_polars_bluesky.ipynb` — Bluesky EDA: multi-IP authors, IP-breadth distribution, audience Jaccard heatmap.
- `notebooks/collaborative_filtering_model.ipynb` — implicit-feedback SVD model on YouTube interactions, hyperparameter search, held-out test evaluation, and softmax-normalized IP-IP co-affinity.

The CF notebook also exports artifacts that the dashboard reads:

- `cf_validation_results.parquet`, `cf_test_results.parquet`, `cf_train_matrix_stats.parquet`
- `cf_ip_affinity.parquet`
- `cf_user_factors.npy`, `cf_item_factors.npy`, `cf_user_index.parquet`, `cf_video_index.parquet`, `cf_user_history.parquet`, `cf_video_metadata.parquet`

## Dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

Pages:

- **Overview** — totals across IPs and platforms.
- **YouTube** — per-IP video stats, total comments bar chart, shared-commenter overlap heatmap, top cross-IP videos, top cross-IP video pairs.
- **Bluesky** — IP-breadth distribution, top multi-IP authors, audience Jaccard heatmap.
- **Collaborative filtering** — training stats, validation curves, held-out test metrics, IP-IP co-affinity heatmap, and an interactive “try the model on a single user” widget with random-user, IP-scoring, and top-10 recommendation actions.
- **Cross-platform** — YouTube vs Bluesky Jaccard heatmaps side-by-side and a sortable pair-by-pair comparison.
