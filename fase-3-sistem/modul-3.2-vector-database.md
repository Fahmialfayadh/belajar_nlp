# MODUL 3.2 — Vector Database & HNSW

> **Estimasi waktu**: 2–3 jam
> **Prerequisite**: Modul 1.1, 1.3, 3.1

---

### 🎯 Tujuan Belajar

- Memahami *mengapa* vector database diperlukan dan apa yang membuatnya berbeda dari database relasional
- Menjelaskan intuisi algoritma HNSW untuk Approximate Nearest Neighbor search
- Bisa memilih vector database yang tepat berdasarkan kebutuhan

---

### 📖 Masalah: Nearest Neighbor di Ruang Berdimensi Tinggi

Misalnya kamu punya 10 juta embedding dokumen, masing-masing 768 dimensi. Ketika ada query, kamu perlu menemukan embedding yang paling mirip dengan query embedding.

**Brute force** (hitung cosine similarity query dengan semua 10 juta embedding): O(n × d) = sangat lambat untuk n besar.

**Solusinya**: **Approximate Nearest Neighbor (ANN)** — tidak perlu menemukan tetangga terdekat yang *pasti*, cukup yang *kira-kira* terdekat dengan cepat. Dalam praktik, ANN yang baik memberikan 95%+ akurasi dengan kecepatan 100x-1000x lebih tinggi dari brute force.

---

### 📖 HNSW: Navigable Small World untuk Vektor

**HNSW** (Hierarchical Navigable Small World) adalah algoritma ANN yang paling banyak digunakan saat ini. Intuisinya diilhami dari konsep **"six degrees of separation"** — ide bahwa setiap dua orang di dunia terhubung melalui maksimal 6 kenalan.

**Analogi: Mencari Teman di Kota Baru**

Bayangkan kamu tiba di kota baru dan ingin bertemu seseorang yang memiliki minat paling mirip denganmu. Kamu tidak kenal siapa-siapa.

Pendekatan naif: temui semua 1 juta penduduk kota satu per satu (brute force).

Pendekatan HNSW:
1. Kamu pertama bertemu **koneksi level tinggi** — tokoh-tokoh berpengaruh yang kenal banyak orang. Mereka tidak tahu siapa yang paling cocok denganmu, tapi bisa menunjukkan kamu ke *lingkungan* yang tepat.
2. Turun ke **koneksi level menengah** — orang-orang di komunitas yang relevan dengan minatmu. Mereka lebih spesifik.
3. Akhirnya masuk ke **koneksi level rendah** — orang-orang spesifik dalam komunitas tersebut. Di sinilah kamu menemukan yang paling cocok.

Ini adalah struktur hierarkis HNSW: **graph multi-layer** di mana layer atas adalah representasi "kasar" (sedikit node, long-range connections) dan layer bawah adalah representasi "detail" (semua node, short-range connections).

**Ketika search query masuk**:
1. Mulai dari entry point di layer paling atas
2. Greedy search: selalu pindah ke tetangga yang paling dekat dengan query
3. Turun ke layer berikutnya, ulangi
4. Di layer terbawah, ambil top-k hasil

Kompleksitas: **O(log n)** — jauh lebih baik dari O(n) brute force.

---

### 📖 Memilih Vector Database

| Database | Keunggulan | Kapan Digunakan |
|----------|-----------|----------------|
| **Qdrant** | Open source, Rust (cepat), filter metadata powerful, binary quantization | Self-hosted, performa tinggi |
| **Milvus** | Open source, skalabilitas enterprise, fitur lengkap, GPU index | Dataset sangat besar (>100M) |
| **Pinecone** | Fully managed, mudah, scale otomatis, serverless tier | Prototype cepat, tidak mau urus infra |
| **Weaviate** | GraphQL API, multimodal, built-in vectorizer | Query kompleks, data multimodal |
| **ChromaDB** | Embeddable, zero setup, in-memory/persistent | Development & testing lokal |
| **pgvector / pgvectorscale** | Extension PostgreSQL, familiar SQL interface | Sudah pakai Postgres, data < 5M |
| **LanceDB** | Embedded, serverless, columnar format, zero-copy | Edge deployment, dataset besar lokal |

**Untuk mulai belajar**: ChromaDB (zero setup) atau Qdrant (paling dekat ke produksi).

---

### 💻 Kode: Vector Database dengan Qdrant

```python
# pip install qdrant-client sentence-transformers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Setup: Qdrant in-memory (untuk development)
# Untuk produksi, ganti dengan QdrantClient(url="http://localhost:6333")
client = QdrantClient(":memory:")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

COLLECTION_NAME = "dokumen_perusahaan"
EMBEDDING_DIM = 384  # dimensi model di atas

# 1. Buat collection (seperti "tabel" di database biasa)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE  # metrik similarity yang digunakan
    )
)

# 2. Masukkan dokumen
dokumen_dengan_metadata = [
    {"teks": "Kebijakan cuti karyawan: 12 hari per tahun.", 
     "departemen": "HR", "tipe": "kebijakan"},
    {"teks": "Prosedur pengajuan reimbursement: isi form F-12 dan kirim ke finance.",
     "departemen": "Finance", "tipe": "prosedur"},
    {"teks": "Server production tidak boleh diakses tanpa approval dari DevOps lead.",
     "departemen": "Engineering", "tipe": "kebijakan"},
    {"teks": "Jadwal town hall Q2 adalah tanggal 15 Juli pukul 14:00 WIB.",
     "departemen": "General", "tipe": "pengumuman"},
]

# Embed semua teks sekaligus (lebih efisien dari satu per satu)
teks_list = [d["teks"] for d in dokumen_dengan_metadata]
embeddings = embedder.encode(teks_list).tolist()

# Insert ke Qdrant
points = [
    PointStruct(
        id=str(uuid.uuid4()),  # ID unik untuk setiap dokumen
        vector=emb,
        payload=doc  # metadata bisa di-filter saat search!
    )
    for doc, emb in zip(dokumen_dengan_metadata, embeddings)
]
client.upsert(collection_name=COLLECTION_NAME, points=points)

# 3. Search dengan filter metadata
query = "bagaimana cara minta cuti?"
query_vector = embedder.encode(query).tolist()

# Search hanya di dokumen dari departemen HR
from qdrant_client.models import Filter, FieldCondition, MatchValue

hasil = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=2,
    query_filter=Filter(  # filter metadata!
        must=[FieldCondition(key="departemen", match=MatchValue(value="HR"))]
    )
)

print(f"Query: '{query}'")
for r in hasil:
    print(f"Score: {r.score:.4f} | {r.payload['teks']}")
```

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Vector database bisa handle semua jenis filter"**
Vector database dioptimasi untuk similarity search. Filter metadata yang sangat kompleks (JOIN, aggregation, range query pada banyak kolom) lebih tepat menggunakan database relasional. Gunakan hybrid: vector DB untuk similarity, relational DB untuk metadata kompleks.

**Jebakan 2: "Simpan teks asli di luar vector database"**
Selalu simpan teks asli (atau setidaknya ID referensi ke sumber) sebagai payload bersama embedding. Kamu perlu teks asli untuk ditampilkan ke pengguna — embedding saja tidak cukup.

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| ANN | Approximately nearest neighbor — cukup "hampir" terdekat, tapi 100x lebih cepat |
| HNSW | Graph multi-layer; search O(log n); standar industri |
| Vector DB | Dioptimasi untuk ANN + filter metadata; bukan pengganti RDBMS |
| Pilihan DB | ChromaDB untuk dev; Qdrant/Milvus untuk produksi; Pinecone untuk managed |

> **Takeaway utama**: Vector database adalah "memory jangka panjang" dari sistem RAG. HNSW adalah algoritma di balik layar yang membuatnya cepat.

**Selanjutnya → Modul 3.3: LoRA & QLoRA** — Bagaimana mengubah "kepribadian" model 70 miliar parameter dengan GPU yang kamu punya.
