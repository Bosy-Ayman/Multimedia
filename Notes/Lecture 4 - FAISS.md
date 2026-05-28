


```python

import numpy as np
import faiss

# 1. Load pretrained GloVe vectors
glove_file = 'glove.6B.50d.txt'
word2vec = {}
with open(glove_file, 'r', encoding='utf8') as f:
    for line in f:
        parts = line.strip().split()
        word2vec[parts[0]] = np.array(parts[1:], dtype='float32')

# 2. Build sentence embeddings
def sentence_embedding(sent, word2vec, d=50):
    words = sent.lower().split()
    vecs = [word2vec[w] for w in words if w in word2vec]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d, dtype='float32')

sentence_embeddings = np.array([sentence_embedding(s, word2vec) for s in sentences], dtype='float32')
faiss.normalize_L2(sentence_embeddings)

# 3. Initialize FAISS
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(50), 50, 3)
index.train(sentence_embeddings)
index.add(sentence_embeddings)

# 4. Search
query_vec = sentence_embedding("perfect program", word2vec).reshape(1, -1)
faiss.normalize_L2(query_vec)
index.nprobe = 3
D, I = index.search(query_vec, k=2)

```


```
Load GloVe
   ↓
Convert words → vectors
   ↓
Convert sentences → averaged vectors
   ↓
Normalize vectors
   ↓
Train FAISS clusters
   ↓
Store vectors in index
   ↓
Convert query → vector
   ↓
Search nearest sentences

```

---

