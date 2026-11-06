import numpy as np, pandas as pd, streamlit as st
from fastembed import TextEmbedding

PREFIX = "Represent this sentence for searching relevant passages: "

@st.cache_resource
def load():
    emb = np.concatenate([np.load(f"emb_part{i}.npy") for i in range(5)])  # float16
    meta = pd.read_parquet("meta.parquet")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return emb, meta, model

emb, meta, model = load()

def similarity(q):
    out = np.empty(len(emb), dtype=np.float32)
    for s in range(0, len(emb), 20000):          # small chunks to save memory
        out[s:s+20000] = emb[s:s+20000].astype(np.float32) @ q
    return out

st.title("Emerging CS Research Finder")
st.caption("Semantic search over ~129K OpenAlex Computer Science papers "
           "(2019-2026 sample; 2026 is a partial year).")

query = st.text_input("Topic or question", placeholder="e.g. LLM agents that use tools")
top_k = st.slider("How many papers", 5, 20, 10)
years = st.slider("Years", 2019, 2026, (2019, 2026))

if query.strip():
    q = np.array(list(model.embed([PREFIX + query]))[0], dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = similarity(q)
    top = np.argpartition(-scores, 400)[:400]
    top = top[np.argsort(-scores[top])]
    shown = 0
    for i in top:
        r = meta.iloc[i]
        if not r["title"] or not (years[0] <= r["year"] <= years[1]):
            continue
        st.markdown(f"**{r['title']}**  \n{r['year']} · group: {r['label']} · "
                    f"score {scores[i]:.2f} · [open]({r['id']})")
        shown += 1
        if shown >= top_k:
            break
    if shown == 0:
        st.write("No papers found for these years.")
