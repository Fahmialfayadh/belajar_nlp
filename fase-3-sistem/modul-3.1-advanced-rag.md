# MODUL 3.1 — Advanced RAG: Lebih dari Sekedar "Cari & Tempel"

> **Estimasi waktu**: 5–7 jam
> **Prerequisite**: Modul 1.3 (Embeddings), pemahaman dasar RAG

---

### 🎯 Tujuan Belajar

- Memahami kelemahan fundamental RAG naif dan *mengapa* ia gagal
- Menjelaskan dan mengimplementasikan Hybrid Search (BM25 + Vector)
- Memahami strategi Chunking semantik yang tepat
- Mengerti konsep Reranking dan kapan ia diperlukan

---

### 🤔 Kenapa Ini Penting?

RAG (Retrieval-Augmented Generation) adalah arsitektur paling dominan untuk LLM-based products saat ini. Hampir semua chatbot dokumen, internal search engine perusahaan, dan AI assistant yang butuh data real-time menggunakan RAG.

Tapi mayoritas implementasi RAG *gagal di produksi* bukan karena model LLM-nya buruk — tapi karena *retrieval*-nya buruk. LLM hanya bisa memberikan jawaban sebaik konteks yang ia terima. "Garbage in, garbage out" berlaku keras di sini.

---

### 📖 Mengapa RAG Naif Sering Gagal

**RAG naif** biasanya bekerja seperti ini:
1. Potong dokumen jadi chunk 512 karakter
2. Embed setiap chunk
3. Simpan di vector database
4. Saat ada query → embed query → ambil top-5 chunk paling mirip → berikan ke LLM

Ini terdengar masuk akal, tapi ada beberapa titik kegagalan fundamental:

**Kegagalan 1: Vector Search Tidak Baik untuk Query Faktual**

Query "berapa harga produk X?" adalah query berbasis *kata kunci*, bukan berbasis *makna semantik*. Vector search (yang berbasis kemiripan semantik) mungkin mengambil dokumen yang temanya relevan tapi tidak mengandung harga spesifik tersebut.

Sebaliknya, BM25 (algoritma pencarian berbasis frekuensi kata) akan langsung menemukan dokumen yang mengandung kata "harga" dan "produk X". Solusinya: gabungkan keduanya — **Hybrid Search**.

**Kegagalan 2: Chunking Sembarangan Memotong Konteks**

Bayangkan dokumen PDF berisi:
```
...Pasal 5 membahas hal berikut:
(1) Pihak pertama bertanggung jawab atas...
(2) Apabila terjadi wanprestasi, maka...
```

Chunking di 512 karakter mungkin memotong tepat di tengah ayat (2). Chunk yang berisi "Apabila terjadi wanprestasi, maka..." tanpa konteks pasal sebelumnya menjadi tidak bermakna.

**Kegagalan 3: Top-K Pertama Tidak Selalu yang Terbaik**

Vector search mengambil n chunk dengan similarity score tertinggi. Tapi similarity score hanyalah estimasi relevansi — tidak sempurna. Sebuah chunk bisa punya score tinggi karena mengandung banyak kata yang sama dengan query, bukan karena benar-benar relevan.

---

### 📖 Hybrid Search: BM25 + Vector Search

**BM25** (Best Match 25) adalah algoritma pencarian klasik yang sudah terbukti handal selama puluhan tahun (digunakan di Elasticsearch, Solr). Ia menghitung skor relevansi berdasarkan:
- Frekuensi kata query di dokumen (Term Frequency)
- Seberapa "langka" kata tersebut di seluruh corpus (Inverse Document Frequency)
- Panjang dokumen (dokumen panjang yang mengandung kata query satu kali, lebih rendah skor-nya dari dokumen pendek)

**Cara menggabungkan**:

```
score_final = α × score_bm25_normalized + (1-α) × score_vector_normalized
```

Di mana α adalah hyperparameter antara 0 dan 1. Teknik ini disebut **Reciprocal Rank Fusion (RRF)** atau weighted combination.

Dalam praktik, α sering ditentukan secara empiris berdasarkan domain:
- Query yang banyak mengandung nama, angka, kode → α tinggi (lebih berat ke BM25)
- Query konseptual / semantik → α rendah (lebih berat ke vector search)

---

### 📖 Chunking Semantik: Memotong dengan Cerdas

Prinsip utama chunking yang baik: **setiap chunk harus bisa berdiri sendiri dan bermakna tanpa konteks dokumen lainnya.**

**Strategi 1 — Fixed-size dengan overlap**
```
Chunk 1: karakter 0-512
Chunk 2: karakter 400-912   ← overlap 112 karakter dengan chunk 1
Chunk 3: karakter 800-1312
```
Overlap memastikan konsep yang melintasi batas chunk tetap terwakili. Ini bukan solusi terbaik tapi mudah diimplementasikan.

**Strategi 2 — Sentence-aware splitting**
Potong di batas kalimat, bukan di batas karakter. Library seperti `nltk.sent_tokenize` atau `spaCy` bisa mendeteksi batas kalimat. Ini menghindari potongan di tengah kalimat.

**Strategi 3 — Recursive character splitting**
Coba potong di `\n\n` (paragraf) dulu. Kalau masih terlalu panjang, potong di `\n` (baris). Kalau masih terlalu panjang, baru potong di spasi. Ini mempertahankan struktur hierarkis dokumen.

**Strategi 4 — Semantic chunking (terbaik, termahal)**
Embed setiap kalimat, lalu deteksi "boundary" topik dengan melihat perubahan mendadak dalam cosine similarity antar kalimat berurutan. Potong di sana. Library `semantic-chunkers` dari Aurelio AI mengimplementasikan ini.

**Strategi 5 — Docling / Unstructured untuk PDF**
PDF adalah format yang notoriusnya sulit di-parse. Tabel, kolom multi-kolom, header/footer — semuanya bisa menghasilkan teks yang acak-acakan jika di-extract naif dengan `PyPDF2`. Library seperti **Docling** (IBM) atau **Unstructured.io** mampu memahami layout dokumen dan mengekstrak konten dengan lebih akurat, termasuk tabel.

---

### 📖 Reranking: Filter Kedua yang Lebih Cerdas

Setelah retrieval (BM25 + vector), kamu punya, misalnya, 20 kandidat chunk. Tidak mungkin semuanya diberikan ke LLM (mahal, dan LLM bisa bingung dengan terlalu banyak konteks). Kamu perlu memilih yang terbaik.

**Reranker** adalah model kecil yang dilatih khusus untuk *menilai relevansi* pasangan (query, dokumen). Berbeda dari embedding model yang hanya membuat vektor secara terpisah, reranker melihat query dan dokumen *bersama-sama* (cross-encoder) sehingga lebih akurat.

Alur:
```
Query → BM25 + Vector Search → 20 kandidat chunk
                                        ↓
                              Reranker menilai setiap (query, chunk)
                                        ↓
                              Sort berdasarkan skor reranker
                                        ↓
                         Top-3 chunk terbaik → diberikan ke LLM
```

Model reranker populer (2025-2026): `BAAI/bge-reranker-v2-m3` (akurat, multilingual), `Cohere Rerank v3.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2` (cepat tapi kurang akurat untuk non-English).

**Trade-off**: Reranking lebih akurat, tapi lebih lambat (harus jalan N kali, satu untuk setiap kandidat). Dalam produksi, biasanya dilakukan pada 20-50 kandidat, bukan seluruh corpus.

---

### 💻 Kode: Implementasi RAG dengan Hybrid Search

```python
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi  # pip install rank-bm25
import torch
import torch.nn.functional as F
import numpy as np

# --- DATA CONTOH ---
dokumen = [
    "Harga paket internet 10GB adalah Rp 50.000 per bulan.",
    "Paket internet kami tersedia dalam berbagai pilihan kecepatan dan kuota.",
    "Cara mengaktifkan paket: ketik REG<spasi>PAKET ke 1234.",
    "Layanan pelanggan kami tersedia 24 jam sehari, 7 hari seminggu.",
    "Kuota internet 10GB cocok untuk streaming video definisi standar.",
    "Harga langganan premium adalah Rp 150.000 per bulan untuk unlimited.",
]

# === BAGIAN 1: BM25 SETUP ===
# BM25 bekerja dengan kata (tokenized), bukan embedding
tokenized_docs = [doc.lower().split() for doc in dokumen]
bm25 = BM25Okapi(tokenized_docs)

# === BAGIAN 2: VECTOR SEARCH SETUP ===
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def embed(teks_list):
    inputs = tokenizer(teks_list, return_tensors="pt", truncation=True, 
                       padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

doc_embeddings = embed(dokumen)  # embed semua dokumen sekali

# === BAGIAN 3: HYBRID SEARCH ===
def hybrid_search(query, top_k=3, alpha=0.5):
    """
    alpha: bobot untuk BM25 (0=pure vector, 1=pure BM25)
    """
    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Vector search scores
    query_emb = embed([query])
    vector_scores = (query_emb @ doc_embeddings.T).squeeze().numpy()
    
    # Gabungkan
    combined = alpha * bm25_norm + (1 - alpha) * vector_scores
    top_indices = np.argsort(combined)[::-1][:top_k]
    
    print(f"\n=== Query: '{query}' ===")
    for i, idx in enumerate(top_indices):
        print(f"[{i+1}] Score={combined[idx]:.3f} | BM25={bm25_norm[idx]:.3f} | Vec={vector_scores[idx]:.3f}")
        print(f"     → {dokumen[idx]}")

# Test
hybrid_search("berapa harga paket internet 10GB?", alpha=0.6)   # BM25 lebih dominan
hybrid_search("cara kontak customer service?", alpha=0.3)        # vector lebih dominan
```

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Tambah saja semua dokumen ke context LLM"**
LLM punya context window yang terbatas. Lebih penting: penelitian menunjukkan LLM buruk dalam *menemukan* informasi spesifik ketika context terlalu panjang (fenomena "lost in the middle"). Retrieval yang presisi lebih baik dari context yang panjang.

**Jebakan 2: "Chunk size yang kecil selalu lebih baik"**
Chunk terlalu kecil kehilangan konteks. Satu kalimat "Harganya Rp 50.000" tanpa konteks "produk apa" menjadi tidak berguna.

**Jebakan 3: "Embedding model terbaik = retrieval terbaik"**
Embedding model yang bagus untuk *kemiripan semantik umum* belum tentu bagus untuk *domain spesifik*. Embedding model untuk dokumen hukum, medis, atau kode perlu fine-tuning khusus.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan skenario konkret di mana: (a) BM25 lebih baik dari vector search, dan (b) vector search lebih baik dari BM25. Apa implikasinya untuk hybrid search?

**Level 2 — Aplikasi:**
Tambahkan reranker ke pipeline di atas. Setelah hybrid search menghasilkan top-5, gunakan `cross-encoder/ms-marco-MiniLM-L-6-v2` untuk menilai ulang dan pilih top-3. Apakah urutan hasilnya berubah?

**Level 3 — Eksplorasi:**
Riset tentang "RAPTOR" (Recursive Abstractive Processing for Tree-Organized Retrieval). Bagaimana ia mengatasi masalah informasi yang tersebar di banyak chunk? Kapan kamu akan menggunakannya vs. RAG standar?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| RAG Naif | Mudah diimplementasikan tapi gagal di produksi karena retrieval tidak presisi |
| Hybrid Search | BM25 untuk kata kunci, vector untuk semantik; gabungkan keduanya |
| Semantic Chunking | Potong di batas topik/kalimat, bukan batas karakter; gunakan overlap |
| Reranking | Cross-encoder untuk filter kedua; lebih akurat dari vector search murni |

> **Takeaway utama**: Kualitas RAG ditentukan 80% oleh kualitas retrieval, bukan oleh kualitas LLM. Investasikan waktu di sini.

**Selanjutnya → Modul 3.2: Vector Database** — Di mana dan bagaimana menyimpan jutaan embedding secara efisien.
