# Emerging CS Research Finder

Semantic search and topic-trend analysis over **128,795 OpenAlex Computer Science papers (2019-2026)**.
The data pipeline is built with **Ray Data** and runs on a free Google Colab T4.

**Live demo:** [ADD YOUR STREAMLIT LINK HERE]

## What it does
1. **Collect:** pulls CS papers with abstracts from the OpenAlex API. It saves every 10K papers, so it can resume after a disconnect.
2. **Clean (Ray Data):** rebuilds abstract text from OpenAlex's inverted index, drops abstracts under 30 words, removes duplicates.
3. **Embed (Ray Data, GPU):** embeds title + abstract with `BAAI/bge-small-en-v1.5` using a stateful actor pool. It saves one file per year, so a failed run skips finished years.
4. **Cluster:** MiniBatchKMeans (80 clusters). Clusters are named from their most distinctive title words (TF-IDF).
5. **Trend analysis:** each cluster's share of all papers per year, compared between 2019-2021 and 2025-2026.
6. **Search app:** type a topic, get the most similar papers, filter by year.

## Results
Share of all papers in the sample (average 2019-2021 vs average 2025-2026):

| Topic group | Early share | Recent share |
|---|---|---|
| AI in education (4 clusters combined) | ~0.9% | ~11.0% |
| LLM code generation / prompt injection | 0.11% | 3.13% |
| Chain-of-thought, dialogue, QA | 0.66% | 3.04% |
| Agentic AI | 0.42% | 1.89% |
| Chatbots in marketing / customer service | 0.57% | 1.79% |

Fading (share dropped): neural architecture search / image classification (3.33% to 0.95%), blockchain (1.57% to 0.69%), edge offloading (1.95% to 0.77%).

[ADD YOUR CHART: emerging_topics.png]

## Performance (Colab free tier: 2 CPU cores, 1x T4 GPU)
| Step | Data | Time |
|---|---|---|
| Clean with Ray Data | 129K records, 13 files | 35.7 s |
| Embed with Ray Data (GPU) | 128,795 papers | 973 s (~132 papers/s), including 8 model loads |

## Reliability tests
[FILL IN AFTER YOU RUN THE TESTS - see the notebook cells]
- Resume test: deleted the 2025 and 2026 embedding files, re-ran, and the pipeline skipped 2019-2024: [your result and time]
- Bad data test: added corrupt lines and empty abstracts to the input: [your result]

## Limitations (please read)
- The papers are a **sample, not all of OpenAlex**. 2019-2023 came from one API pull, and 2024-2026 from separate pulls of about 15K papers per year. The API returns papers in its own order, so the sample may lean towards some kinds of papers. Trends are shown as *share of papers*, not raw counts, for this reason.
- 2026 is a partial year.
- KMeans clusters can overlap (four clusters are about AI in education). One cluster mixes English and Spanish papers.
- "Fading" means a falling share, not that the topic disappeared.

## What I would change at 10x scale
- Read the OpenAlex Parquet snapshot from S3 and sample randomly per year.
- Run the embedding actor pool on several GPUs, and keep the model loaded between years.
- Use an approximate index (FAISS IVF or HNSW) instead of a full scan.
- Use BERTopic or HDBSCAN for cleaner topics, and check them manually on a sample.

## Run it yourself
```bash
pip install streamlit fastembed pandas pyarrow numpy
streamlit run app.py
```
Put `meta.parquet` and `emb_part0.npy` ... `emb_part4.npy` next to `app.py`.
The notebook `pipeline.ipynb` has the full pipeline. Add your own OpenAlex API key where it says `YOUR_OPENALEX_KEY`.

Data: [OpenAlex](https://openalex.org). Model: BAAI/bge-small-en-v1.5.
