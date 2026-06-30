# MODUL 1.3 — Embeddings: Static vs Contextual

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Modul 1.1, 1.2

---

### 🎯 Tujuan Belajar

- Memahami *mengapa* static embeddings (Word2Vec) tidak cukup untuk bahasa yang ambigu
- Menjelaskan mekanisme contextual embeddings pada level konseptual
- Mampu menggunakan HuggingFace untuk menghasilkan embeddings dan mengukur kemiripan semantik

---

### 🤔 Kenapa Ini Penting?

Kalau tokenization adalah cara model *membaca*, embedding adalah cara model *memahami*. Setiap token yang masuk ke model harus diubah menjadi vektor sebelum operasi apa pun bisa dilakukan.

Pemahaman yang dalam tentang embedding akan membuatmu bisa:
- Men-debug kenapa sistem RAG-mu mengambil dokumen yang salah
- Memilih embedding model yang tepat untuk domain yang spesifik
- Memahami kenapa fine-tuning diperlukan — dan apa yang sebenarnya berubah

---

### 📖 Static Embeddings: Satu Kata, Satu Vektor (dan Masalahnya)

**Word2Vec, GloVe, FastText** — semuanya menghasilkan *static embedding*: setiap kata punya **satu vektor tetap**, terlepas dari konteks penggunaannya.

Ini berarti kata "apel" selalu punya vektor yang sama, baik di kalimat:
- "Saya makan **apel** merah yang manis."
- "**Apel** meluncurkan iPhone 17 kemarin."

Padahal makna "apel" di dua kalimat itu jelas berbeda (buah vs. perusahaan teknologi). Static embeddings mencoba merata-ratakan semua konteks penggunaan kata tersebut di seluruh corpus training — hasilnya adalah vektor yang "merata-rata" semua makna, yang tidak optimal untuk kalimat manapun.

**Analogi**: Bayangkan kamu membuat foto tunggal seseorang dengan memrata-ratakan semua foto mereka sepanjang hidup — dari bayi sampai tua, dari seragam sekolah sampai baju pesta. Hasilnya adalah gambar blur yang tidak merepresentasikan mereka di waktu mana pun dengan akurat.

---

### 📖 Contextual Embeddings: Vektor yang Bergantung pada Konteks

Model seperti **BERT, GPT, RoBERTa** menghasilkan *contextual embedding*: vektor sebuah token **berubah tergantung pada token-token di sekitarnya**.

Dengan kata lain:
- "apel" di kalimat buah → vektor **v₁**
- "apel" di kalimat teknologi → vektor **v₂**
- v₁ ≠ v₂ (dan perbedaan ini bisa diukur!)

**Bagaimana ini bisa terjadi?** Inilah di mana mekanisme **Attention** berperan — topik yang akan kita bahas mendalam di Fase 2. Untuk sekarang, cukup pahami bahwa setiap token "melihat" semua token lain dalam kalimat dan *memperbarui representasinya sendiri* berdasarkan konteks tersebut.

---

### 📖 Lapisan Embedding: Dari Token ke Vektor

Proses mengubah token menjadi vektor terjadi di **Embedding Layer** — lapisan pertama dari setiap model Transformer.

Secara teknis, ini adalah sebuah **lookup table**: matriks berukuran `[ukuran_vocabulary × dimensi_embedding]`. Misalnya untuk BERT-base:
- Vocabulary size: 30.522 token
- Embedding dimension: 768
- Ukuran embedding matrix: 30.522 × 768 = ~23,4 juta parameter

Ketika token dengan ID `4521` masuk, model mengambil baris ke-4521 dari matriks ini — itulah embedding awalnya. Baris-baris inilah yang dioptimasi selama training.

> **Insight**: Parameter di embedding matrix *adalah* pengetahuan model tentang kata-kata. Ketika kamu melakukan fine-tuning, sebagian besar perubahan terjadi di lapisan atas (attention layers), bukan di embedding layer ini — karena pengetahuan dasar tentang kata-kata sudah baik dari pre-training.

---

### 📖 Sentence Embeddings: Dari Token ke Kalimat

Untuk task seperti pencarian semantik atau RAG, kita butuh **satu vektor untuk satu kalimat/dokumen**, bukan satu vektor per token.

Beberapa pendekatan:

**1. Mean Pooling** — rata-ratakan semua token embeddings:
```
embedding_kalimat = mean(embedding_token_1, embedding_token_2, ..., embedding_token_n)
```

**2. [CLS] Token** — di BERT, token `[CLS]` ditambahkan di awal setiap input. Setelah melewati semua lapisan Transformer, embedding `[CLS]` didesain untuk menangkap "inti makna" dari seluruh kalimat.

**3. Model Khusus (Sentence-BERT / SBERT)** — BERT standar tidak dioptimasi untuk sentence similarity. SBERT dilatih secara khusus (dengan contrastive learning) agar embedding kalimat yang mirip maknanya benar-benar berdekatan di ruang vektor.

**4. Model Embedding Generasi Baru (2025-2026)** — Model seperti `Qwen3-Embedding-8B` (#1 MTEB leaderboard per Juni 2026), `Google Gemini Embedding v2`, `Cohere Embed v4`, dan `NV-Embed-v2` menawarkan performa yang jauh di atas SBERT asli, dengan dukungan multilingual lebih baik, konteks hingga 32K token, dan bahkan kemampuan multimodal (teks + gambar).

Untuk produksi (RAG, pencarian semantik), **selalu gunakan model yang didesain khusus untuk sentence/document embedding** — bukan BERT standar.

---

### 📖 Deep-Dive: Bagaimana Embedding Model Dilatih? (Contrastive Learning)

Modul sebelumnya menyebut "contrastive learning" tanpa menjelaskan. Mari kita bongkar, karena ini fundamental untuk memahami *mengapa* beberapa embedding model lebih baik dari yang lain.

**Masalah**: Bagaimana melatih model agar kalimat yang *mirip maknanya* punya embedding yang berdekatan, dan yang *berbeda maknanya* punya embedding yang berjauhan?

**Jawaban: Contrastive Learning** — latih model dengan "contoh kontras":

```
Training sample:
  Anchor:    "Kucing tidur di sofa"
  Positive:  "Ada kucing berbaring di kursi"      → tarik mendekat ✓
  Negative:  "Mobil melaju kencang di jalan"       → dorong menjauh ✗
  
Loss function (InfoNCE):
  → Minimize distance(anchor, positive)
  → Maximize distance(anchor, negative)
  → Dilakukan atas semua sample dalam satu batch
```

**Kenapa "hard negatives" penting?**

Negative yang terlalu mudah ("mobil" vs "kucing") tidak memberi banyak sinyal belajar — model sudah tahu mereka berbeda. Yang lebih berguna adalah **hard negatives** — contoh yang *hampir mirip* tapi sebenarnya berbeda:

```
Anchor:       "Cara mengajukan visa kerja Australia"
Easy negative: "Resep rendang"                        (terlalu beda, tidak informatif)
Hard negative: "Syarat mengajukan visa turis Australia" (mirip tapi BEDA intent!)
```

Model yang dilatih dengan hard negatives menghasilkan embedding yang jauh lebih tajam dan presisi.

#### Evolusi Teknik Training Embedding (2026)

| Teknik | Era | Cara Kerja |
|--------|-----|-----------|
| Contrastive (SBERT) | 2019 | Anchor + positive + negative, triplet loss |
| Hard Negative Mining | 2021+ | Cari negative yang paling "membingungkan" untuk model |
| Instruction-Tuned | 2024+ | Tambahkan instruksi di depan query: "retrieve a document answering..." |
| Distillation dari Reranker | 2025+ | Cross-encoder reranker (lebih akurat) mengajarkan embedding model |
| zELO | 2026 | Relevansi dimodelkan sebagai Elo rating (seperti ranking catur), bukan biner |

**Instruction-Tuned Embeddings** layak penjelasan lebih:

Model embedding terbaru (Qwen3-Embedding, NV-Embed-v2) dilatih dengan *task instructions* di depan query:

```
# Tanpa instruction (model lama):
embed("cara merawat kucing")

# Dengan instruction (model baru):
embed("Given a question, retrieve a relevant document: cara merawat kucing")
```

Instruksi memberi "petunjuk" ke model tentang *jenis* kesamaan apa yang harus dicari — semantic similarity, exact match, topical relevance, dll. Ini meningkatkan performa secara signifikan.

---

### 📖 Update 2026: Matryoshka Embeddings — Potong Dimensi Tanpa Retrain

Bayangkan kamu punya embedding 1024 dimensi yang disimpan di vector database untuk jutaan dokumen. Suatu hari, kamu perlu mempercepat search karena latensi terlalu tinggi. Harus retrain model? Harus re-embed semua dokumen?

**Matryoshka Representation Learning (MRL)** mengatasi ini:

- Model dilatih agar *dimensi awal* embedding sudah mengandung informasi terpenting
- Kamu bisa **memotong** embedding dari 1024D ke 256D (atau bahkan 64D!) dan hasilnya masih bermakna
- Seperti boneka Matryoshka Rusia — setiap "lapisan" mengandung informasi yang semakin detail

```
Embedding penuh (1024D): [0.23, 0.45, 0.12, ..., 0.89]  → akurasi 95%
Dipotong ke 512D:        [0.23, 0.45, 0.12, ..., 0.67]  → akurasi 93%  (↓2%)
Dipotong ke 256D:        [0.23, 0.45, 0.12, ..., 0.34]  → akurasi 90%  (↓5%)
Dipotong ke 64D:         [0.23, 0.45, 0.12, ..., 0.11]  → akurasi 82%  (↓13%)
```

**Kegunaan praktis**:
- **Hemat storage**: 256D vs 1024D = 4x lebih hemat di vector database
- **Search lebih cepat**: dimensi lebih rendah = jarak dihitung lebih cepat
- **Adaptive**: satu model, banyak "resolusi" embedding sesuai kebutuhan

Model modern (Gemini Embedding v2, Qwen3-Embedding, NV-Embed-v2) sudah mendukung Matryoshka secara default — cek dokumentasi untuk dimensi minimum yang direkomendasikan.

---

### 💻 Kode: Melihat Contextual Embeddings Bekerja

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# --- SETUP ---
# Model ringan untuk demo belajar — cukup cepat bahkan tanpa GPU.
# Untuk produksi, ganti ke: Qwen3-Embedding-8B, BAAI/bge-m3, atau Gemini Embedding v2
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # mode evaluasi, mematikan dropout

def get_embedding(teks):
    """Mengubah satu kalimat menjadi satu vektor embedding."""
    inputs = tokenizer(teks, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # tidak perlu hitung gradient saat inferensi
        outputs = model(**inputs)
    # Mean pooling: rata-ratakan embedding semua token
    # (kecuali token padding yang tidak bermakna)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
    # Normalisasi agar cosine similarity = dot product
    return F.normalize(embeddings, p=2, dim=1)

# --- EKSPERIMEN: Apakah Contextual Embeddings Menangkap Ambiguitas? ---
kalimat_buah    = "An apple is a very popular fruit around the world."
kalimat_tech    = "Apple is a very popular tech brand around the world."
kalimat_samsung = "Samsung produces new smartphones and electronic devices."
kalimat_mangga  = "Mangoes and apples are very popular fruits around the world."

emb_buah    = get_embedding(kalimat_buah)
emb_tech    = get_embedding(kalimat_tech)
emb_samsung = get_embedding(kalimat_samsung)
emb_mangga  = get_embedding(kalimat_mangga)

# Hitung similarity (karena sudah dinormalisasi, dot product = cosine similarity)
def sim(a, b):
    return (a @ b.T).item()

print("=== Kemiripan Semantik ===")
print(f"'apel buah' ↔ 'apel tech'    : {sim(emb_buah, emb_tech):.4f}")   # harusnya RENDAH
print(f"'apel buah' ↔ 'mangga & apel': {sim(emb_buah, emb_mangga):.4f}")  # harusnya TINGGI
print(f"'apel buah' ↔ 'Samsung'       : {sim(emb_buah, emb_samsung):.4f}")# harusnya RENDAH
print(f"'apel tech' ↔ 'Samsung'       : {sim(emb_tech, emb_samsung):.4f}") # LEBIH  TINGGI DARI APLE BUAH
```

> **Yang perlu kamu amati**: Apakah model berhasil memisahkan "apel buah" dari "apel perusahaan"? Jika similarity antara `kalimat_buah` dan `kalimat_mangga` lebih tinggi dari `kalimat_buah` dan `kalimat_tech` — model berhasil memahami konteks.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Embedding model terbaik untuk semua domain"**
Tidak ada. Untuk produksi, pilih berdasarkan kebutuhan: `Qwen3-Embedding-8B` (top MTEB, open source), `BAAI/bge-m3` (solid multilingual baseline), atau `Gemini Embedding v2` (API, multimodal). Untuk domain medis atau hukum, model umum seringkali performa buruk — butuh domain-specific embedding atau fine-tuning.

**Jebakan 2: "Menggunakan BERT biasa untuk sentence similarity"**
BERT standar dilatih untuk *masked language modeling*, bukan untuk *sentence similarity*. Embedding [CLS] dari BERT tidak serta-merta bermakna secara semantik untuk perbandingan. Gunakan **SBERT / Sentence-Transformers** atau model embedding generasi baru untuk task ini.

**Jebakan 3: "Embedding sekali jalan untuk semua input"**
Di sistem RAG, kamu meng-embed dokumen *satu kali* dan menyimpannya di vector database. Tapi query pengguna harus di-embed *setiap kali*. Pastikan kamu menggunakan **model embedding yang sama** untuk dokumen dan query — mencampur model yang berbeda akan menghasilkan vektor yang tidak sebanding.

**Jebakan 4: "Skor MTEB leaderboard = kualitas di domain saya"**
MTEB mengukur performa *umum* pada benchmark standar. Di produksi RAG, performa sangat bergantung pada domain spesifikmu (hukum, medis, e-commerce). Selalu lakukan *evaluasi sendiri* pada data yang representatif sebelum memilih model embedding.

---

### 🧩 Latihan

**Level 1 — Recall:**
Apa perbedaan fundamental antara static embedding (Word2Vec) dan contextual embedding (BERT)? Berikan contoh konkret di mana static embedding akan gagal tapi contextual embedding berhasil.

**Level 2 — Aplikasi:**
Buat fungsi `cari_paling_mirip(query, daftar_dokumen)` yang: (1) embed query dan semua dokumen, (2) hitung cosine similarity query terhadap setiap dokumen, (3) return dokumen yang paling mirip. Test dengan 5 kalimat tentang topik berbeda. Ini adalah inti dari sistem RAG paling sederhana.

**Level 3 — Eksplorasi:**
Cari tahu tentang **"embedding drift"** atau **"domain shift"** dalam embedding. Mengapa model embedding yang dilatih pada Wikipedia bisa perform buruk pada data e-commerce atau medis? Apa solusinya di industri?

---

### 🔬 Eksperimen Google Colab: Matryoshka Embeddings — Potong Dimensi, Ukur Dampaknya!

```python
# ============================================================
# EKSPERIMEN: Matryoshka Embeddings — Seberapa Banyak Dimensi yang Bisa Dipotong?
# ============================================================
# Lihat seberapa banyak dimensi yang bisa dipotong sebelum retrieval rusak

!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Simulasi RAG corpus
corpus = [
    "Python adalah bahasa pemrograman yang populer untuk data science.",
    "Machine learning menggunakan data untuk melatih model prediktif.",
    "Resep nasi goreng: goreng nasi dengan bumbu dan sayuran.",
    "Kucing adalah hewan peliharaan yang mandiri dan suka tidur.",
    "Transformer adalah arsitektur neural network untuk NLP.",
    "Jakarta adalah ibukota Indonesia dengan populasi terbesar.",
    "Deep learning adalah subset dari machine learning yang menggunakan neural network.",
    "Bali terkenal dengan pantai dan budayanya yang kaya.",
    "Fine-tuning LLM membutuhkan GPU dengan VRAM yang besar.",
    "Rendang adalah masakan Minang yang terkenal di seluruh dunia.",
]

query = "cara melatih model AI"

query_emb = model.encode(query)
corpus_embs = model.encode(corpus)
full_dim = query_emb.shape[0]

# Test berbagai level truncation (Matryoshka-style)
dims_to_test = [full_dim, 256, 128, 64, 32, 16, 8]
full_top1 = np.argmax(corpus_embs @ query_emb / 
                      (np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb)))

print(f"Full embedding dimension: {full_dim}")
print(f"Query: '{query}'")
print("=" * 80)

for dim in dims_to_test:
    q = query_emb[:dim]
    c = corpus_embs[:, :dim]
    
    q_norm = q / np.linalg.norm(q)
    c_norm = c / np.linalg.norm(c, axis=1, keepdims=True)
    scores = c_norm @ q_norm
    top_3_idx = np.argsort(scores)[::-1][:3]
    
    match = "✅" if full_top1 == top_3_idx[0] else "❌ DRIFT!"
    print(f"\n📐 Dimensi: {dim} (kompresi {(1 - dim/full_dim)*100:.0f}%) | Top-1 match: {match}")
    for rank, idx in enumerate(top_3_idx):
        print(f"   [{rank+1}] Score={scores[idx]:.4f} | {corpus[idx][:60]}...")

print("\n💡 INSIGHT: Perhatikan sampai dimensi berapa top-1 result masih sama.")
print("   Dimensi yang bisa 'dipotong' tanpa mengubah hasil = dimensi 'redundan'.")
print("   Ini prinsip di balik Matryoshka Embeddings!")
```

> **Yang perlu kamu amati**: Pada dimensi berapa top-1 result mulai berubah? Itu adalah "batas aman" truncation. Model yang dilatih dengan Matryoshka objective memiliki batas ini jauh lebih rendah dari model biasa.

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Static Embedding | Satu kata = satu vektor tetap; tidak bisa menangani ambiguitas |
| Contextual Embedding | Vektor berubah sesuai konteks; dihasilkan oleh Transformer |
| Sentence Embedding | Satu kalimat = satu vektor; gunakan SBERT atau model khusus, bukan BERT biasa |
| Contrastive Learning | Latih model dengan contoh mirip (positive) dan berbeda (negative) |
| Hard Negatives | Contoh negatif yang "hampir mirip" — kunci embedding yang presisi |
| Matryoshka (MRL) | Potong dimensi embedding tanpa retrain; dimensi awal = informasi terpenting |
| Instruction-Tuned | Embedding yang menerima instruksi task untuk presisi lebih tinggi |
| Domain Spesifik | Embedding terbaik adalah yang dilatih pada domain yang sama dengan use case |

> **Takeaway utama**: Embedding adalah "representasi pikiran" model tentang sebuah teks — dan kualitasnya bergantung pada teknik training (contrastive learning, hard negatives) dan kesesuaian domain. Pilih embedding model yang tepat, dan ketahui cara mengukur kualitasnya.

**Selanjutnya → Fase 2: Transformer** — Sekarang kita sudah punya token dan embedding. Saatnya memahami *mesin* yang mengolahnya.
