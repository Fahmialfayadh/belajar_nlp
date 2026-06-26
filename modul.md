# 🧠 Modul Belajar NLP Modern 2026
### Untuk Fahmi — AI Engineering, Tahun Pertama

> **Total estimasi waktu**: 40–60 jam belajar aktif
> **Level**: Menengah (Python & math sudah oke)
> **Filosofi modul ini**: Kode hanya alat *verifikasi* — teori adalah pemahaman sesungguhnya.

---

## 🗺️ Peta Besar: Ke Mana Kita Pergi

```
FASE 1: Fondasi
  └── Math Terapan (Dot Product, Cosine, Ruang Vektor)
  └── Tokenization Modern (BPE, SentencePiece)
  └── Embeddings (Static vs Contextual)
  └── Ekosistem: PyTorch + HuggingFace
       ↓
FASE 2: Arsitektur
  └── Transformer & Self-Attention (teori mendalam)
  └── MoE, Efficient Attention, SSM/Mamba
  └── Base Model vs Instruct Model
       ↓
FASE 3: Sistem
  └── Advanced RAG (Hybrid Search, Chunking, Reranking)
  └── Vector Database (HNSW, Qdrant/Milvus)
  └── LoRA & QLoRA (Parameter-Efficient Fine-Tuning)
  └── Evaluasi: RAGAS & LLM-as-a-Judge
       ↓
FASE 4: Agentic AI
  └── AI Agents & Orkestrasi (LangGraph)
  └── Tool Use / Function Calling
  └── Multi-Agent Systems
```

Setiap fase **membangun di atas fase sebelumnya**. Jangan loncat.

---
---

# 📦 FASE 1: Fondasi Matematika & Embeddings

> *"Kamu tidak bisa memahami kenapa Attention bekerja kalau kamu belum paham kenapa dot product itu bermakna."*

---

## MODUL 1.1 — Matematika Terapan untuk NLP Modern

> **Estimasi waktu**: 4–6 jam
> **Prerequisite**: Aljabar linear dasar (tahu apa itu vektor dan matriks)

---

### 🎯 Tujuan Belajar

Setelah modul ini, kamu bisa:
- Menjelaskan *mengapa* dot product bisa mengukur "kemiripan makna" antar teks
- Memahami intuisi di balik ruang vektor berdimensi tinggi
- Mengerti batasan matematis embedding vektor tunggal untuk reasoning kompleks

---

### 🤔 Kenapa Ini Penting?

Bayangkan kamu diminta membangun mesin pencari yang bisa memahami pertanyaan "cara mengobati flu" dan mencocokkannya dengan artikel berjudul "penanganan infeksi virus saluran napas" — meski tidak ada kata yang sama. Ini bukan keajaiban. Ini matematika.

Seluruh kemampuan LLM — dari ChatGPT sampai Gemini — berakar pada satu operasi matematika sederhana: **perkalian vektor**. Kalau kamu tidak paham ini di level intuisi, kamu hanya akan jadi *pengguna* library, bukan *insinyur* yang bisa men-debug ketika sesuatu salah.

---

### 📖 Konsep Inti: Dot Product sebagai Alat Ukur Kemiripan

#### Apa itu Dot Product?

Secara mekanis, dot product dari dua vektor **a** = [a₁, a₂, ..., aₙ] dan **b** = [b₁, b₂, ..., bₙ] adalah:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

Tapi *mekanis* bukan berarti *paham*. Mari kita bangun intuisinya dari nol.

**Analogi: Profil Selera Film**

Misalnya kamu merepresentasikan selera film seseorang sebagai vektor 3 dimensi:
`[suka_aksi, suka_romance, suka_horor]`

- Profil Budi: `[0.9, 0.1, 0.2]` → suka aksi banget, tidak terlalu romance/horor
- Profil Ani: `[0.8, 0.2, 0.1]` → mirip Budi
- Profil Cici: `[0.1, 0.9, 0.1]` → suka romance

Dot product Budi·Ani = (0.9×0.8) + (0.1×0.2) + (0.2×0.1) = 0.72 + 0.02 + 0.02 = **0.76** (tinggi)
Dot product Budi·Cici = (0.9×0.1) + (0.1×0.9) + (0.2×0.1) = 0.09 + 0.09 + 0.02 = **0.20** (rendah)

Tanpa kamu membaca profil mereka satu per satu, dot product *sudah memberitahumu* bahwa Budi lebih mirip Ani daripada Cici. **Ini persis cara kerja pencarian semantik di NLP.**

#### Mengapa Dot Product = Kesamaan Arah?

Ada versi lain dari dot product:

```
a · b = |a| × |b| × cos(θ)
```

Di mana θ adalah sudut antara dua vektor. Perhatikan implikasinya:
- Kalau θ = 0° (arah sama persis) → cos(0°) = 1 → dot product maksimum
- Kalau θ = 90° (tegak lurus, tidak ada kesamaan) → cos(90°) = 0 → dot product = 0
- Kalau θ = 180° (berlawanan arah) → cos(180°) = -1 → dot product negatif

Ini mengungkap sesuatu penting: **dot product mengukur seberapa "searah" dua vektor itu.** Dalam konteks NLP, "searah" berarti "bermakna serupa."

---

### 📖 Cosine Similarity: Dot Product yang Lebih Adil

Ada masalah dengan dot product mentah: ia terpengaruh **panjang vektor**, bukan hanya arahnya.

Misalnya vektor kata "kucing" panjangnya 1.0, dan vektor kata "hewan" panjangnya 10.0 (lebih banyak fitur/konteks). Dot product "kucing"·"hewan" akan kecil bukan karena mereka tidak mirip, tapi karena "kucing" vektornya pendek.

**Cosine Similarity** menyelesaikan ini dengan membagi dot product dengan panjang kedua vektor:

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
```

Hasilnya selalu antara -1 dan 1, terlepas dari panjang vektor. Kita hanya mengukur **arah**, bukan magnitude.

> **Insight kritis**: Di dalam Transformer, operasi *Attention* pada dasarnya adalah cosine similarity yang dikomputasi antara jutaan pasang vektor, secara paralel, untuk menentukan "kata mana yang paling relevan dengan kata ini."

---

### 📖 Ruang Vektor Berdimensi Tinggi: Kenapa 768 Dimensi?

Model seperti BERT merepresentasikan setiap token sebagai vektor **768 dimensi**. GPT-4 mungkin menggunakan 12.288 dimensi. Kenapa sebesar itu?

**Analogi: Koordinat GPS vs Deskripsi Lengkap**

GPS menggunakan 2 dimensi (latitude, longitude) untuk menemukan lokasi. Tapi 2 dimensi tidak cukup untuk mendeskripsikan sebuah kota secara lengkap — tidak ada informasi tentang populasi, iklim, budaya, ekonomi, dsb.

Kata-kata jauh lebih kompleks dari koordinat GPS. Kata "bank" punya dimensi makna: institusi keuangan, tepian sungai, menyimpan data, kemiringan pesawat... Setiap dimensi dalam vektor embedding menangkap *satu aspek* dari makna kata tersebut.

**768 dimensi bukan angka ajaib** — ia dipilih karena memberikan kapasitas yang cukup untuk memisahkan semua nuansa makna yang ada dalam bahasa, sekaligus masih bisa diproses secara efisien oleh GPU.

#### Fenomena Menarik di Ruang Berdimensi Tinggi

Di ruang 2D atau 3D, intuisi kita tentang "jarak" dan "kerapatan" masih bekerja. Tapi di ruang 768 dimensi, hal-hal aneh terjadi yang disebut **"curse of dimensionality"**:

- Hampir semua titik menjadi sama jauhnya satu sama lain
- Volume ruang bertambah secara eksponensial → data menjadi sangat "jarang" (sparse)
- Konsep "tetangga terdekat" menjadi kurang bermakna

Ini adalah salah satu alasan kenapa model perlu dilatih dengan **sangat banyak data** — agar titik-titik data (kata-kata dengan konteks berbeda) cukup padat mengisi ruang berdimensi tinggi ini.

---

### 📖 Batasan Matematis: Mengapa Satu Vektor Tidak Cukup untuk Reasoning

Ini adalah insight riset terbaru yang wajib kamu tahu.

**Masalah**: Kata "bank" dalam kalimat "Saya pergi ke **bank** untuk menabung" dan "Saya duduk di **bank** sungai" — idealnya harus punya representasi vektor yang berbeda. Contextual embeddings sudah mengatasi ini (dibahas di modul berikutnya).

Tapi ada masalah yang lebih dalam: **single-vector representation tidak bisa menangkap relasi yang kompleks secara bersamaan.**

Misalnya, untuk menjawab pertanyaan multi-hop seperti:
*"Siapa presiden negara tempat lahirnya pencipta teori relativitas?"*

Model perlu melakukan rangkaian reasoning:
1. Teori relativitas → Einstein
2. Einstein lahir di → Jerman
3. Presiden Jerman → Olaf Scholz

Satu vektor statis tidak bisa menyimpan seluruh rantai inferensi ini sekaligus. Inilah mengapa arsitektur modern bergerak ke arah **multi-step reasoning** (Chain-of-Thought), **external memory**, dan **agentic systems** — topik yang akan kita bahas di Fase 3 dan 4.

---

### 💻 Kode: Rasakan Langsung Dot Product & Cosine Similarity

```python
import numpy as np

# --- BAGIAN 1: Dot Product Intuitif ---
# Kita representasikan "makna" kata dalam 4 dimensi khayalan:
# [makhluk_hidup, bergerak, punya_kaki, hidup_di_air]

vektor_kucing  = np.array([1.0, 1.0, 1.0, 0.0])
vektor_anjing  = np.array([1.0, 1.0, 1.0, 0.1])
vektor_ikan    = np.array([1.0, 1.0, 0.0, 1.0])
vektor_mobil   = np.array([0.0, 1.0, 0.0, 0.0])

# Dot product mentah
print("=== Dot Product ===")
print(f"kucing · anjing = {np.dot(vektor_kucing, vektor_anjing):.2f}")   # harusnya tinggi
print(f"kucing · ikan   = {np.dot(vektor_kucing, vektor_ikan):.2f}")    # sedang (sama-sama makhluk hidup)
print(f"kucing · mobil  = {np.dot(vektor_kucing, vektor_mobil):.2f}")   # rendah

# --- BAGIAN 2: Cosine Similarity ---
# Perhatikan perbedaannya dengan dot product!

def cosine_similarity(a, b):
    # Rumus: (a · b) / (|a| × |b|)
    # np.linalg.norm() menghitung panjang/magnitude vektor
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n=== Cosine Similarity ===")
print(f"kucing ↔ anjing: {cosine_similarity(vektor_kucing, vektor_anjing):.4f}")
print(f"kucing ↔ ikan:   {cosine_similarity(vektor_kucing, vektor_ikan):.4f}")
print(f"kucing ↔ mobil:  {cosine_similarity(vektor_kucing, vektor_mobil):.4f}")

# Output yang diharapkan:
# kucing ↔ anjing: ~0.9987  (sangat mirip)
# kucing ↔ ikan:   ~0.7071  (lumayan mirip, sama-sama makhluk hidup)
# kucing ↔ mobil:  ~0.4082  (tidak mirip)
```

> **Yang perlu kamu perhatikan setelah menjalankan kode ini**: Bukan hasilnya — tapi *apakah intuisimu tentang kemiripan terbukti secara matematis?* Coba ubah nilai vektornya dan lihat bagaimana hasilnya berubah. Eksperimen ini lebih berharga dari membaca teorinya.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Cosine similarity selalu antara 0 dan 1"**
Salah. Ia antara -1 dan 1. Nilai negatif muncul ketika dua vektor "berlawanan arah" — dalam NLP, ini bisa berarti kata-kata yang bermakna kontradiktif (antonym).

**Jebakan 2: "Dimensi embedding yang lebih besar selalu lebih baik"**
Tidak. Ada trade-off: lebih banyak dimensi = lebih ekspresif, tapi juga lebih lambat dan butuh lebih banyak data training untuk mengisi dimensi tersebut dengan bermakna. GPT-2 small (117M parameter) menggunakan 768 dimensi dan masih berguna untuk banyak task.

**Jebakan 3: "Kalau cosine similarity tinggi, artinya maknanya sama"**
Hati-hati. Model yang buruk atau dilatih pada domain sempit mungkin memberi similarity tinggi pada kata-kata yang sebenarnya berbeda. Kualitas embedding sangat bergantung pada kualitas dan skala data training.

---

### 🧩 Latihan

**Level 1 — Recall:**
Tanpa membuka materi, jelaskan dalam 3 kalimat: mengapa kita menggunakan *cosine similarity* dan bukan *dot product mentah* untuk mengukur kemiripan kata?

**Level 2 — Aplikasi:**
Modifikasi kode di atas. Tambahkan vektor untuk kata "paus" dan "lumba-lumba". Berdasarkan pemikiranmu, berapa nilai cosine similarity-nya terhadap "ikan" vs "anjing"? Periksa hasilnya. Apakah sesuai intuisimu? Jika tidak, kenapa?

**Level 3 — Eksplorasi:**
Cari tahu: apa itu **"hubungan paralel" dalam embedding** (contoh klasik: king - man + woman ≈ queen). Secara matematis, operasi apa yang terjadi di sini? Mengapa ini bisa bekerja?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Dot Product | Mengukur "searah"-nya dua vektor; makin besar = makin mirip |
| Cosine Similarity | Versi dot product yang adil — hanya ukur arah, bukan magnitude |
| Ruang Berdimensi Tinggi | Lebih ekspresif tapi butuh lebih banyak data; ada "curse of dimensionality" |
| Batasan Single Vector | Tidak cukup untuk multi-step reasoning — inilah kenapa kita perlu Agents |

> **Takeaway utama**: Seluruh NLP modern adalah seni mengubah teks menjadi vektor yang bermakna — dan menggunakan operasi vektor sederhana untuk melakukan hal-hal yang tampak "cerdas."

**Selanjutnya → Modul 1.2: Tokenization** — Sebelum jadi vektor, teks harus dipotong-potong dulu. Dan cara pemotongannya ternyata jauh lebih berpengaruh dari yang kamu kira.

---
---

## MODUL 1.2 — Tokenization Modern: BPE & SentencePiece

> **Estimasi waktu**: 3–4 jam
> **Prerequisite**: Modul 1.1

---

### 🎯 Tujuan Belajar

- Menjelaskan mengapa tokenization berbasis kata utuh tidak skalabel
- Memahami mekanisme Byte-Pair Encoding (BPE) dari nol
- Mengerti implikasi tokenization terhadap performa model pada bahasa non-Inggris

---

### 🤔 Kenapa Ini Penting?

Ini pertanyaan yang kelihatannya sepele: *"Bagaimana LLM membaca teks?"*

Jawabannya bukan "huruf per huruf" dan bukan "kata per kata." Jawabannya adalah **token per token** — dan definisi "token" ini bukan sembarangan. Pilihan tokenization mempengaruhi:
- Berapa banyak "langkah" yang dibutuhkan model untuk memproses sebuah kalimat
- Apakah model bisa menangani kata baru / kata asing / typo
- Kenapa model bahasa Inggris *jauh lebih efisien* dari model bahasa Indonesia pada infrastruktur yang sama

---

### 📖 Masalah dengan Tokenization Naif

#### Pendekatan 1: Character-level (huruf per huruf)
"Halo" → ["H", "a", "l", "o"]

**Masalah**: Kalimat 100 karakter menghasilkan 100 token. Model harus "berpikir" 100 langkah. Hubungan jarak jauh (antara kata pertama dan terakhir) jadi sangat sulit dipelajari karena jaraknya terlalu panjang.

#### Pendekatan 2: Word-level (kata per kata)
"Saya pergi ke sekolah" → ["Saya", "pergi", "ke", "sekolah"]

**Masalah**: Kosakata bahasa Inggris saja sudah lebih dari 170.000 kata. Bahasa Indonesia dengan afiksasi-nya (me-, -kan, ber-, per-...) bisa menghasilkan ratusan ribu variasi. Ukuran vocabulary table-nya akan raksasa — dan kata baru / nama orang / istilah teknis yang tidak ada di training data akan menjadi **OOV (Out-of-Vocabulary)** dan tidak bisa diproses sama sekali.

**Solusi yang dicari**: Sesuatu di antara keduanya — potongan yang cukup pendek agar fleksibel, tapi cukup panjang agar efisien.

---

### 📖 Byte-Pair Encoding (BPE): Algoritma yang Mengubah Segalanya

BPE awalnya adalah algoritma kompresi data dari tahun 1994. Pada 2015, paper "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al.) mengadopsinya untuk NLP — dan hasilnya revolusioner.

#### Cara Kerja BPE: Step by Step

**Langkah 0 — Mulai dari karakter individual.**
Anggap corpus kita adalah: "low lower lowest newer newest"

Representasi awal (setiap kata dipecah ke karakter + penanda akhir kata `</w>`):
```
l o w </w>         (frekuensi: 1)
l o w e r </w>     (frekuensi: 1)
l o w e s t </w>   (frekuensi: 1)
n e w e r </w>     (frekuensi: 1)
n e w e s t </w>   (frekuensi: 1)
```

**Langkah 1 — Hitung semua pasangan karakter yang berdekatan.**
```
(l,o): 3 kali
(o,w): 3 kali
(w,</w>): 1 kali
(w,e): 4 kali   ← pasangan paling sering!
(e,r): 2 kali
(e,s): 2 kali
...
```

**Langkah 2 — Gabungkan pasangan paling sering menjadi simbol baru.**
`(w,e)` → `we` menjadi simbol baru. Update semua kata:
```
l o we </w>
l o we r </w>
l o we s t </w>
n we r </w>
n we s t </w>
```

**Langkah 3 — Ulangi.** Sekarang cari pasangan paling sering lagi:
`(l,o)` muncul 3 kali → gabungkan jadi `lo`:
```
lo we </w>
lo we r </w>
lo we s t </w>
n we r </w>
n we s t </w>
```

**Lanjutkan** sampai kamu mencapai ukuran vocabulary yang diinginkan (misalnya 50.000 token).

Hasilnya: kata-kata umum menjadi satu token ("low" = `low</w>`), kata-kata jarang tetap dipecah jadi subwords. Kata baru / typo / nama asing bisa tetap diproses dengan memecahnya ke subwords yang dikenal.

#### Mengapa Ini Elegan?

BPE **belajar dari data** — bukan dari aturan linguistik yang dikodekan manual. Ia secara otomatis menemukan unit-unit bermakna: "un", "ing", "tion", "re" untuk bahasa Inggris; "me", "ber", "kan" untuk bahasa Indonesia. Ini terjadi tanpa pernah memberi tahu algoritma apa itu prefiks atau sufiks.

---

### 📖 SentencePiece: BPE yang Lebih Modern

**Masalah BPE standar**: Ia bergantung pada spasi untuk memisahkan kata. Ini rusak untuk:
- Bahasa Mandarin / Jepang (tidak ada spasi antar kata)
- Teks informal dengan typo atau tanpa spasi
- Code (yang punya karakter khusus)

**SentencePiece** (digunakan oleh BERT multilingual, T5, Llama) mengatasi ini dengan memperlakukan teks sebagai **aliran byte mentah** — tanpa asumsi tentang spasi atau batasan kata. Spasi sendiri menjadi karakter yang bisa digabungkan (direpresentasikan sebagai `▁`).

Hasilnya:
- "Hello world" → ["▁Hello", "▁world"]
- "Helloworld" → ["▁Hello", "world"] *(perhatikan: tidak ada `▁` di "world" karena tidak ada spasi sebelumnya)*

---

### 📖 Implikasi Penting yang Sering Diabaikan

#### Token ≠ Kata
Kalimat "Saya belajar NLP" dalam tokenizer GPT-2:
- "Saya" → `[26501]` (1 token)
- " belajar" → `[29260]` (1 token, termasuk spasi)
- " NLP" → `[45", "LP"]` → mungkin 2 token! (huruf kapital jarang, dipecah)

Implikasinya: **biaya API LLM dihitung per token, bukan per kata.** Teks dalam bahasa Indonesia rata-rata menggunakan **20-40% lebih banyak token** dari teks bahasa Inggris yang setara maknanya — karena tokenizer dilatih dominan pada data Inggris.

#### Sensitivitas Kapitalisasi
"kucing" dan "Kucing" sering menghasilkan token ID yang berbeda. Ini berarti model bisa berperilaku berbeda hanya karena huruf kapital di awal kalimat. Ini bukan bug — ini konsekuensi dari cara BPE belajar dari teks asli.

---

### 💻 Kode: Visualisasi Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer GPT-2 (representatif untuk model-model modern berbasis BPE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

kalimat_list = [
    "Saya belajar NLP",
    "saya belajar nlp",          # huruf kecil semua
    "SayaBelajarNLP",             # tanpa spasi
    "I am learning NLP",          # bandingkan dengan Inggris
]

for kalimat in kalimat_list:
    tokens = tokenizer.tokenize(kalimat)
    token_ids = tokenizer.encode(kalimat)
    print(f"\nKalimat  : '{kalimat}'")
    print(f"Tokens   : {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Jumlah   : {len(tokens)} token")

# Perhatikan: kalimat Inggris lebih sedikit tokennya dari kalimat Indonesia yang setara!
```

> **Yang perlu kamu amati**: Bandingkan jumlah token kalimat Bahasa Indonesia vs Inggris. Lalu coba tambahkan kalimat dengan kata typo ("blaajr") dan lihat bagaimana tokenizer tetap bisa memprosesnya.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Token = kata, jadi context window = jumlah kata"**
Tidak. Context window dihitung dalam token. "Gemini 1.5 punya context window 1 juta token" ≠ 1 juta kata. Dalam bahasa Indonesia, mungkin hanya ~700.000 kata.

**Jebakan 2: "Tokenizer tidak perlu dipahami dalam produksi"**
Sangat perlu. Kalau kamu melakukan chunking dokumen untuk RAG dan memotong di 512 *karakter*, kamu bisa memotong di tengah-tengah token (bahkan di tengah karakter multi-byte seperti emoji atau aksara non-Latin), yang menghasilkan output garbage.

**Jebakan 3: "Semua model punya tokenizer yang sama"**
Tidak. GPT-4, Llama, Gemini, Claude — semuanya punya vocabulary dan tokenizer yang berbeda. Kamu tidak bisa mengukur "panjang" teks dalam token tanpa mengetahui tokenizer model yang akan kamu gunakan.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan dalam kalimatmu sendiri: mengapa BPE lebih baik dari tokenization berbasis kata utuh (word-level)? Sebutkan minimal 2 alasan konkret.

**Level 2 — Aplikasi:**
Jalankan kode di atas dengan tokenizer yang berbeda: coba `"bert-base-uncased"` dan `"xlm-roberta-base"` (multilingual). Berapa jumlah token untuk kalimat Indonesia yang sama? Tokenizer mana yang paling efisien untuk Bahasa Indonesia?

**Level 3 — Eksplorasi:**
Cari tahu tentang **"tokenizer fertility"** — metrik yang mengukur rata-rata token per kata untuk bahasa tertentu. Temukan data fertility tokenizer GPT-4 untuk beberapa bahasa. Apa implikasinya terhadap biaya dan fairness penggunaan LLM di negara non-Inggris?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| BPE | Algoritma kompresi yang belajar memecah kata ke subwords dari data |
| SentencePiece | BPE yang tidak bergantung pada spasi, lebih universal |
| Token ≠ Kata | Bahasa Indonesia lebih "mahal" secara token dari bahasa Inggris |
| Implikasi Produksi | Tokenizer berpengaruh pada biaya API, chunking, dan perilaku model |

> **Takeaway utama**: Tokenization bukan langkah teknis yang bisa diabaikan — ia adalah "bahasa" yang digunakan model untuk membaca dunia, dan pilihannya punya konsekuensi nyata.

**Selanjutnya → Modul 1.3: Embeddings** — Setelah teks jadi token, setiap token diubah menjadi vektor. Tapi tidak semua vektor diciptakan sama.

---
---

## MODUL 1.3 — Embeddings: Static vs Contextual

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
- "**Apel** meluncurkan iPhone 16 kemarin."

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

Untuk produksi (RAG, pencarian semantik), **selalu gunakan SBERT atau model yang didesain untuk sentence embedding** — bukan BERT standar.

---

### 💻 Kode: Melihat Contextual Embeddings Bekerja

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# --- SETUP ---
# Model ini dilatih khusus untuk sentence similarity (bukan BERT biasa!)
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
kalimat_buah    = "Saya makan apel merah yang sangat manis."
kalimat_tech    = "Apel meluncurkan produk baru di konferensi tahunan mereka."
kalimat_samsung = "Samsung merilis smartphone flagship terbaru bulan ini."
kalimat_mangga  = "Mangga dan apel adalah buah favoritku untuk dijus."

emb_buah    = get_embedding(kalimat_buah)
emb_tech    = get_embedding(kalimat_tech)
emb_samsung = get_embedding(kalimat_samsung)
emb_mangga  = get_embedding(kalimat_mangga)

# Hitung similarity (karena sudah dinormalisasi, dot product = cosine similarity)
def sim(a, b):
    return (a @ b.T).item()

print("=== Kemiripan Semantik ===")
print(f"'apel buah' ↔ 'apel tech'    : {sim(emb_buah, emb_tech):.4f}")   # harusnya RENDAH
print(f"'apel tech' ↔ 'Samsung'       : {sim(emb_tech, emb_samsung):.4f}") # harusnya TINGGI
print(f"'apel buah' ↔ 'mangga & apel': {sim(emb_buah, emb_mangga):.4f}")  # harusnya TINGGI
print(f"'apel buah' ↔ 'Samsung'       : {sim(emb_buah, emb_samsung):.4f}")# harusnya RENDAH
```

> **Yang perlu kamu amati**: Apakah model berhasil memisahkan "apel buah" dari "apel perusahaan"? Jika similarity antara `kalimat_buah` dan `kalimat_mangga` lebih tinggi dari `kalimat_buah` dan `kalimat_tech` — model berhasil memahami konteks.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Embedding model terbaik untuk semua domain"**
Tidak ada. Model `all-MiniLM-L6-v2` bagus untuk teks umum Bahasa Inggris. Untuk Bahasa Indonesia, kamu perlu model yang dilatih pada data Indonesia. Untuk domain medis atau hukum, model umum seringkali performa buruk — butuh domain-specific embedding.

**Jebakan 2: "Menggunakan BERT biasa untuk sentence similarity"**
BERT standar dilatih untuk *masked language modeling*, bukan untuk *sentence similarity*. Embedding [CLS] dari BERT tidak serta-merta bermakna secara semantik untuk perbandingan. Gunakan **SBERT / Sentence-Transformers** untuk task ini.

**Jebakan 3: "Embedding sekali jalan untuk semua input"**
Di sistem RAG, kamu meng-embed dokumen *satu kali* dan menyimpannya di vector database. Tapi query pengguna harus di-embed *setiap kali*. Pastikan kamu menggunakan **model embedding yang sama** untuk dokumen dan query — mencampur model yang berbeda akan menghasilkan vektor yang tidak sebanding.

---

### 🧩 Latihan

**Level 1 — Recall:**
Apa perbedaan fundamental antara static embedding (Word2Vec) dan contextual embedding (BERT)? Berikan contoh konkret di mana static embedding akan gagal tapi contextual embedding berhasil.

**Level 2 — Aplikasi:**
Buat fungsi `cari_paling_mirip(query, daftar_dokumen)` yang: (1) embed query dan semua dokumen, (2) hitung cosine similarity query terhadap setiap dokumen, (3) return dokumen yang paling mirip. Test dengan 5 kalimat tentang topik berbeda. Ini adalah inti dari sistem RAG paling sederhana.

**Level 3 — Eksplorasi:**
Cari tahu tentang **"embedding drift"** atau **"domain shift"** dalam embedding. Mengapa model embedding yang dilatih pada Wikipedia bisa perform buruk pada data e-commerce atau medis? Apa solusinya di industri?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Static Embedding | Satu kata = satu vektor tetap; tidak bisa menangani ambiguitas |
| Contextual Embedding | Vektor berubah sesuai konteks; dihasilkan oleh Transformer |
| Sentence Embedding | Satu kalimat = satu vektor; gunakan SBERT, bukan BERT biasa |
| Domain Spesifik | Embedding terbaik adalah yang dilatih pada domain yang sama dengan use case |

> **Takeaway utama**: Embedding adalah "representasi pikiran" model tentang sebuah teks — dan kualitasnya bergantung pada konteks pelatihan. Pilih embedding model yang tepat untuk domainmu.

**Selanjutnya → Fase 2: Transformer** — Sekarang kita sudah punya token dan embedding. Saatnya memahami *mesin* yang mengolahnya.

---
---

# 📦 FASE 2: Arsitektur NLP Modern

> *"Kamu tidak perlu mengimplementasikan Transformer dari nol. Tapi kamu HARUS bisa menjelaskannya di atas kertas."*

---

## MODUL 2.1 — Self-Attention & Transformer: Teori Mendalam

> **Estimasi waktu**: 6–8 jam (fase paling padat)
> **Prerequisite**: Seluruh Fase 1

---

### 🎯 Tujuan Belajar

- Menjelaskan secara tepat *mengapa* Self-Attention lebih baik dari RNN untuk memahami konteks
- Memahami mekanisme Query-Key-Value (QKV) secara intuitif dan matematis
- Mampu menjelaskan Multi-Head Attention dan Positional Encoding dari first principles

---

### 🤔 Kenapa Ini Penting?

Transformer adalah fondasi dari hampir semua model language AI modern: GPT-4, Gemini, Claude, Llama, Mistral — semuanya Transformer atau variannya. Kalau kamu melamar kerja sebagai AI Engineer dan tidak bisa menjelaskan cara kerja Self-Attention, kamu akan gugur di technical interview.

Lebih penting dari itu: memahami Transformer akan membuat semua konsep lanjutan — fine-tuning, RAG, attention patterns, context window — masuk akal secara intuitif, bukan sekadar "tool yang digunakan."

---

### 📖 Masalah yang Dipecahkan Transformer

Sebelum 2017, arsitektur dominan untuk pemrosesan teks berurutan adalah RNN (Recurrent Neural Network) dan LSTM. Mereka punya masalah fundamental:

**Masalah 1 — Sequential Processing (tidak bisa paralel)**
RNN memproses teks satu token per waktu: token₁ → token₂ → token₃ → ... Ini berarti token ke-100 baru bisa diproses setelah token ke-1 sampai ke-99 selesai. Tidak bisa diparalelkan di GPU — pelatihan sangat lambat.

**Masalah 2 — Vanishing Gradient untuk konteks jauh**
Bayangkan kalimat: "The cat that my neighbor's dog chased **was** very scared."
Untuk memahami bahwa "was" harus singular (karena subjeknya adalah "cat"), model harus mengingat informasi dari 7 kata yang lalu. Semakin panjang teks, semakin sulit RNN mempertahankan informasi dari konteks jauh karena gradients "menghilang" saat backpropagation.

**Solusi Transformer**: Setiap token langsung "melihat" semua token lain secara bersamaan — tidak ada urutan, tidak ada jarak yang membatasi. Dan semuanya bisa dikomputasi secara paralel.

---

### 📖 Self-Attention: Intuisi dari Nol

**Pertanyaan inti**: Ketika memproses kata "dia" dalam kalimat "Budi pergi ke toko karena **dia** butuh beras" — bagaimana model tahu bahwa "dia" merujuk ke "Budi"?

Jawabannya adalah **Self-Attention**: setiap kata memberi "perhatian" (attention) yang berbeda-beda kepada semua kata lain, dan menggunakan perhatian tersebut untuk memperbarui representasi dirinya sendiri.

**Analogi: Konferensi Meja Bundar**

Bayangkan setiap kata dalam kalimat adalah seseorang yang duduk di meja bundar. Setiap orang punya:
- Sebuah **pertanyaan (Query)**: "Siapa yang relevan denganku?"
- Sebuah **papan nama (Key)**: "Inilah identitasku"
- Sebuah **informasi (Value)**: "Inilah yang bisa aku bagikan"

Proses attention: Setiap orang membandingkan *pertanyaannya* dengan *papan nama* semua orang lain. Semakin cocok pertanyaan dengan papan nama, semakin banyak perhatian yang diberikan, dan semakin besar pengaruh *informasi* orang tersebut terhadap representasi akhir si penanya.

Kata "dia" akan "bertanya" ke semua kata lain. Papan nama kata "Budi" akan cocok dengan pertanyaan ini — sehingga "dia" akan sangat memperhatikan "Budi" dan mengambil informasinya untuk memperbarui representasinya sendiri.

---

### 📖 Query, Key, Value: Mekanisme Matematis

Setiap embedding input **x** (vektor dari embedding layer) diproyeksikan menjadi tiga vektor:

```
Q = x · W_Q    (Query — "apa yang saya cari?")
K = x · W_K    (Key   — "apa identitas saya?")
V = x · W_V    (Value — "informasi apa yang saya punya?")
```

Di mana W_Q, W_K, W_V adalah matriks bobot yang *dipelajari saat training*. Dimensinya biasanya lebih kecil dari embedding asli (misalnya d_model=768 → d_k=64 untuk setiap head).

**Menghitung Attention Score:**

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Mari kita bongkar ini step by step:

**Step 1: `Q · K^T` — Hitung relevansi setiap pasangan**
Untuk setiap token, hitung dot product query-nya dengan key *semua* token lain. Hasilnya adalah matriks berukuran `[n_tokens × n_tokens]` yang berisi "skor relevansi" setiap pasangan token.

**Step 2: `/ √d_k` — Stabilisasi skala**
Kenapa dibagi √d_k? Karena semakin besar dimensi vektor, semakin besar nilai dot product — ini membuat gradients menjadi sangat kecil saat backprop. Pembagian √d_k menjaga nilai dalam range yang stabil.

**Step 3: `softmax(...)` — Normalisasi jadi probabilitas**
Softmax mengubah skor relevansi menjadi distribusi probabilitas yang berjumlah 1. Ini adalah "bobot perhatian" — seberapa besar perhatian yang diberikan ke setiap token.

**Step 4: `· V` — Ambil nilai berbobot**
Kalikan bobot perhatian dengan Value vectors semua token, lalu jumlahkan. Hasilnya adalah representasi baru untuk token ini — sebuah campuran informasi dari semua token, berbobot sesuai relevansinya.

---

### 📖 Multi-Head Attention: Memperhatikan dari Banyak Perspektif

Satu attention head memperhatikan satu jenis relasi. Tapi bahasa punya banyak jenis relasi:
- Relasi sintaktis (subjek-predikat-objek)
- Relasi semantik (sinonim, antonim)
- Relasi koreference ("dia" = "Budi")
- Relasi posisional (kata sebelum/sesudah)

**Multi-Head Attention** menjalankan beberapa attention head secara paralel, masing-masing dengan W_Q, W_K, W_V yang berbeda (parameter independen). Hasilnya dari semua head digabungkan (concatenate) dan diproyeksikan kembali ke dimensi asli.

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · W_O
di mana headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
```

BERT-base menggunakan **12 heads**, BERT-large menggunakan **16 heads**. Setiap head belajar memperhatikan relasi yang berbeda secara otomatis melalui training.

---

### 📖 Positional Encoding: Karena Transformer Tidak Tahu Urutan

Ada masalah besar: Self-Attention adalah operasi yang **tidak sensitif terhadap urutan**. "Budi membeli beras" dan "Beras membeli Budi" akan menghasilkan attention yang persis sama karena keduanya mengandung kata yang sama.

**Positional Encoding** menyelesaikan ini dengan menambahkan vektor posisi ke setiap embedding:

```
input_ke_transformer = embedding_token + positional_encoding(posisi)
```

Positional encoding versi asli (dari paper "Attention Is All You Need") menggunakan fungsi sinus-kosinus:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Kenapa fungsi trigonometri?** Karena:
1. Nilainya selalu antara -1 dan 1 (tidak mendominasi embedding)
2. Setiap posisi punya vektor yang unik
3. Model bisa mengekstrapolasi ke posisi yang lebih panjang dari training (dalam batas tertentu)

Model modern (Llama, Mistral) menggunakan **Rotary Positional Embedding (RoPE)** yang lebih elegan dan bisa menangani context window yang lebih panjang.

---

### 📖 Arsitektur Lengkap Transformer Block

Satu "block" Transformer terdiri dari:

```
Input Embedding (+ Positional Encoding)
        ↓
┌─────────────────────────────────┐
│  Multi-Head Self-Attention      │
│  + Residual Connection          │  ← x = x + MultiHead(x)
│  + Layer Normalization          │  ← x = LayerNorm(x)
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Feed-Forward Network (FFN)     │  ← 2 linear layers + aktivasi
│  + Residual Connection          │  ← x = x + FFN(x)
│  + Layer Normalization          │  ← x = LayerNorm(x)
└─────────────────────────────────┘
        ↓
    Output ke block berikutnya
```

Block ini **diulang N kali**: BERT-base = 12 block, GPT-3 = 96 block, GPT-4 (estimasi) = ~120 block.

**Kenapa Residual Connection penting?**
Tanpanya, sinyal dari input layer "hilang" setelah melewati banyak block — disebut *vanishing gradient*. Residual connection memastikan gradients bisa mengalir langsung dari output ke input selama backpropagation. Ini memungkinkan pelatihan model yang sangat dalam (ratusan layer).

**Kenapa FFN setelah Attention?**
Attention mengumpulkan informasi dari token lain — ini operasi "komunikasi" antar token. FFN kemudian memproses informasi tersebut secara independen untuk setiap token — ini operasi "komputasi" internal. Keduanya diperlukan untuk model yang powerful.

---

### 💻 Kode: Visualisasi Attention Weights

```python
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Gunakan BERT kecil agar lebih cepat
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

# Kalimat yang menarik untuk divisualisasi (ambiguitas referensi)
kalimat = "The cat sat on the mat because it was tired"
inputs = tokenizer(kalimat, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple of (n_layers,) tensors
# Setiap tensor: shape [batch, n_heads, seq_len, seq_len]
attentions = outputs.attentions

# Ambil layer terakhir, head pertama, sebagai contoh
# (setiap head belajar relasi yang berbeda — eksplorasi sendiri!)
last_layer_attn = attentions[-1][0]  # shape: [n_heads, seq_len, seq_len]
head_0_attn = last_layer_attn[0].numpy()  # ambil head ke-0

# Visualisasi heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    head_0_attn,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap='Blues',
    annot=False
)
plt.title("Attention Weights — Layer Terakhir, Head 0")
plt.xlabel("Key (token yang diperhatikan)")
plt.ylabel("Query (token yang memperhatikan)")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
print("Gambar disimpan: attention_heatmap.png")

# TANTANGAN: Perhatikan baris untuk kata "it"
# Apakah attention weights-nya tinggi untuk "cat"? Jika iya, model berhasil!
```

> **Yang perlu kamu amati**: Perhatikan baris untuk token "it". Token mana yang mendapat attention weight paling tinggi? Apakah itu "cat"? Coba kalimat lain dan lihat apakah pola attention-nya masuk akal secara linguistik.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Attention weight tinggi = model 'mengerti' relasi tersebut"**
Tidak sesederhana itu. Penelitian tentang "attention interpretability" masih aktif diperdebatkan. Attention weights tinggi menunjukkan *korelasi*, bukan kausalitas. Model bisa mencapai jawaban benar dengan pola attention yang terlihat "salah" secara linguistik.

**Jebakan 2: "Context window = memori model"**
Transformer tidak punya "memori" dalam arti tradisional. Ia memproses seluruh context window *sekaligus* setiap kali ada input baru. Ini berbeda dari memori manusia yang bersifat episodik dan jangka panjang.

**Jebakan 3: "Lebih banyak layer = lebih baik"**
Ada trade-off. Layer awal cenderung menangkap fitur sintaktis (struktur kalimat), layer akhir cenderung menangkap fitur semantik (makna). Tapi terlalu banyak layer tanpa data dan regularisasi yang cukup akan menghasilkan overfitting.

---

### 🧩 Latihan

**Level 1 — Recall:**
Gambar diagram sederhana (di kertas) yang menunjukkan flow sebuah token melalui satu Transformer block: mulai dari embedding input, melalui Multi-Head Attention, residual, layer norm, FFN, residual, layer norm, sampai ke output.

**Level 2 — Aplikasi:**
Modifikasi kode di atas untuk memvisualisasikan head yang berbeda (index 0 sampai 11 untuk BERT-base). Apakah setiap head menunjukkan pola yang berbeda? Coba identifikasi head mana yang mungkin menangkap relasi posisional vs. relasi semantik.

**Level 3 — Eksplorasi:**
Cari dan baca ringkasan paper "Are Sixteen Heads Really Better than One?" (Michel et al., 2019). Apa yang ditemukan tentang pentingnya setiap attention head? Apa implikasinya untuk efisiensi model?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Self-Attention | Setiap token memperhatikan semua token lain; tidak ada "jarak" yang membatasi |
| Q-K-V | Query = "apa yang kucari", Key = "identitasku", Value = "informasiku" |
| Multi-Head | Beberapa attention head paralel; masing-masing menangkap jenis relasi berbeda |
| Positional Encoding | Ditambahkan agar model tahu urutan token (karena attention sendiri tidak sensitif urutan) |
| Residual + LayerNorm | Memungkinkan training model yang sangat dalam tanpa vanishing gradient |

> **Takeaway utama**: Transformer memungkinkan setiap token langsung "berkomunikasi" dengan semua token lain dalam satu operasi paralel — inilah kenapa ia jauh lebih powerful dan efisien dari RNN.

**Selanjutnya → Modul 2.2: Arsitektur Modern** — Transformer standar sudah berumur 8 tahun. Industri sudah jauh berkembang.

---
---

## MODUL 2.2 — Arsitektur Mutakhir: MoE, Efficient Attention & Mamba

> **Estimasi waktu**: 3–4 jam
> **Prerequisite**: Modul 2.1

---

### 🎯 Tujuan Belajar

- Memahami mengapa Transformer standar tidak skalabel untuk model triliunan parameter
- Menjelaskan intuisi Mixture of Experts (MoE) dan trade-off-nya
- Mengenal kelemahan quadratic complexity Attention dan pendekatan solusinya
- Memahami mengapa SSM/Mamba muncul sebagai alternatif

---

### 🤔 Kenapa Ini Penting?

Model-model state-of-the-art di 2026 — Mixtral, Gemini, Grok — semua menggunakan variasi dari arsitektur ini. Kalau kamu hanya tahu Transformer "vanilla", kamu akan bingung membaca paper dan dokumentasi model terbaru. Dan kalau kamu bekerja di tim yang memilih atau mengevaluasi model, kamu perlu tahu trade-off arsitektural ini.

---

### 📖 Masalah Skalabilitas Transformer

**Masalah 1 — Biaya Komputasi Quadratic**

Self-Attention punya kompleksitas **O(n²)** terhadap panjang sequence, di mana n = jumlah token.

Artinya: doubling panjang context = 4x biaya komputasi. Untuk context window 1 juta token (seperti Gemini 1.5), attention standar secara harfiah tidak bisa dikomputasi dalam waktu yang wajar.

**Masalah 2 — Semua Parameter Aktif untuk Setiap Token**

Model 70 miliar parameter seperti Llama-3-70B mengaktifkan *semua* 70 miliar parameternya untuk memproses *setiap token*. Ini sangat boros — apakah benar-benar semua "pengetahuan" model diperlukan untuk memproses kata "dan"?

---

### 📖 Mixture of Experts (MoE): Otak yang Terspesialisasi

**Ide dasarnya sederhana**: Daripada satu jaringan besar yang selalu aktif, gunakan banyak jaringan kecil (*experts*) dan sebuah *router* yang memilih hanya sebagian experts untuk setiap token.

```
Input Token
     ↓
┌─────────────────────┐
│  ROUTER (Gating)    │  ← "Expert mana yang paling cocok untuk token ini?"
│  Pilih Top-K Expert │
└─────────────────────┘
     ↓ (hanya aktivasi 2 dari, misalnya, 8 experts)
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Expert1│ │Expert2│ │Expert3│ │Expert4│ │Expert5│ │Expert6│ │Expert7│ │Expert8│
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
  (aktif)  (aktif)   (skip)    (skip)    (skip)    (skip)    (skip)    (skip)
     ↓
Output = weighted sum dari output Expert1 dan Expert2
```

**Hasilnya**:
- Model Mixtral 8x7B punya **46.7 miliar total parameter**, tapi hanya **12.9 miliar** yang aktif untuk setiap token (2 dari 8 experts, masing-masing 7B)
- Performa mendekati model 70B, dengan biaya komputasi mendekati model 13B

**Trade-off yang perlu kamu ketahui**:
- Performa tinggi saat inferensi (hanya sebagian parameter aktif)
- Tapi training jauh lebih kompleks dan tidak stabil (router bisa "collapse" — semua token diarahkan ke 1-2 expert yang sama)
- Memori total tetap besar (semua parameter harus dimuat ke GPU/RAM meski tidak semuanya aktif)

---

### 📖 Efficient Attention: Solusi untuk Quadratic Complexity

Beberapa pendekatan untuk mengatasi O(n²):

**FlashAttention** — Bukan mengubah matematikanya, tapi mengubah *cara komputasi*-nya agar lebih efisien di GPU (memanfaatkan hirarki memori GPU). Hasilnya secara matematis identik dengan Attention standar, tapi 2-4x lebih cepat dan hemat memori. Hampir semua framework modern sudah menggunakannya.

**Sparse Attention** — Setiap token tidak memperhatikan *semua* token lain, hanya sebagian (token terdekat, token "landmark", dsb.). Mengurangi kompleksitas ke O(n√n) atau O(n log n). Tradeoff: bisa kehilangan informasi dari konteks jauh yang tidak masuk dalam subset.

**Linear Attention / Linformer** — Menggunakan aproksimasi matematis untuk menurunkan kompleksitas ke O(n). Dalam praktik, aproksimasi ini seringkali menghasilkan kualitas yang lebih rendah pada task-task yang butuh presisi tinggi.

---

### 📖 State Space Models (SSM) & Mamba: Pesaing Transformer

**Konteks historis**: Sebelum Transformer, ada model *State Space* dari kontrol sistem dan signal processing. Pada 2022-2023, peneliti mulai mengadaptasinya untuk NLP dengan hasil yang mengejutkan.

**Intuisi SSM**: Alih-alih "memperhatikan semua token sekaligus" (seperti Attention), SSM mempertahankan sebuah *hidden state* yang terus diperbarui saat memproses token satu per satu — mirip RNN, tapi dengan matematika yang jauh lebih stabil dan parallelizable.

**Mamba** (Gu & Dao, 2023) adalah SSM yang paling berpengaruh. Inovasinya: *selective state spaces* — model bisa memilih secara adaptif informasi mana yang perlu dipertahankan di hidden state berdasarkan input.

**Keunggulan Mamba vs Transformer**:
- Kompleksitas **O(n)** terhadap panjang sequence — linear, bukan quadratic
- Efisien untuk context yang sangat panjang (jutaan token)
- Inferensi lebih cepat karena tidak perlu menyimpan seluruh KV-cache

**Kelemahan Mamba**:
- Pada benchmark bahasa standar (≤ 4K token), masih kalah dari Transformer dengan ukuran yang sama
- "Recall" informasi dari konteks sangat jauh tidak sebaik Attention
- Ekosistem dan tooling masih jauh lebih kecil dari Transformer

**Status 2026**: Arsitektur hybrid (Transformer + SSM) seperti Jamba sudah mulai muncul, mencoba mengambil keunggulan keduanya.

---

### 📖 Base Model vs Instruct Model

Ini adalah konsep yang sering disalahpahami dan kritis untuk dipahami saat memilih model.

**Base Model** (contoh: Llama-3-70B-Base, GPT-3):
- Dilatih dengan *pre-training* pada teks internet yang sangat besar
- Tujuan satu-satunya: **menebak token berikutnya**
- Jika kamu beri prompt "Cara membuat nasi goreng:", ia mungkin melanjutkan dengan teks acak dari internet yang temanya memasak — bisa jadi resep, bisa jadi review restoran, bisa jadi apa saja
- Tidak "mengerti" instruksi manusia, tidak punya "kepribadian"

**Instruct/Chat Model** (contoh: Llama-3-70B-Instruct, ChatGPT):
- Base model yang kemudian di-*fine-tune* menggunakan data instruksi manusia
- Proses: **Supervised Fine-Tuning (SFT)** → **RLHF (Reinforcement Learning from Human Feedback)** atau **DPO (Direct Preference Optimization)**
- Mampu mengikuti instruksi, berdialog, dan menghindari output berbahaya
- Punya "kepribadian" yang konsisten

**Kapan pakai yang mana?**
- Research, pre-training experiments, atau kalau kamu mau fine-tune sendiri dari awal → Base model
- Hampir semua use case produksi (chatbot, assistant, RAG) → Instruct model

---

### ⚠️ Jebakan Umum

**Jebakan 1: "MoE berarti model lebih murah untuk di-deploy"**
Tidak selalu. Kamu tetap perlu muat semua parameter ke GPU (untuk Mixtral 8x7B = ~90GB dalam float16). Yang lebih murah adalah biaya *komputasi saat inferensi* — bukan memori.

**Jebakan 2: "Mamba akan menggantikan Transformer"**
Masih terlalu dini. Per 2026, Transformer masih mendominasi dan ekosistemnya jauh lebih matang. Mamba sangat menjanjikan untuk aplikasi yang butuh context sangat panjang, tapi belum menggantikan Transformer untuk task general.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan dengan analogi sederhana: apa perbedaan antara Dense Model (Transformer biasa) dan Sparse Model (MoE)? Mengapa ini penting untuk efisiensi?

**Level 2 — Eksplorasi:**
Cari perbandingan benchmark Mixtral 8x7B vs Llama-2-70B. Pada task apa MoE unggul? Pada task apa ia tidak unggul? Apa implikasinya untuk pemilihan model di produksi?

**Level 3 — Riset:**
Cari paper atau blogpost yang membahas arsitektur **Jamba** atau **Zamba** (hybrid Mamba-Transformer). Apa klaim keunggulannya? Bagaimana mereka menggabungkan SSM dan Attention? Apakah ada benchmark yang memvalidasi klaim tersebut?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| MoE | Banyak "expert" kecil; hanya sebagian yang aktif per token — efisiensi tinggi |
| Quadratic Attention | Masalah skalabilitas; FlashAttention mempercepat tanpa mengubah matematika |
| Mamba/SSM | Alternatif O(n) untuk sequence panjang; ekosistem masih berkembang |
| Base vs Instruct | Base = prediksi token; Instruct = ikuti instruksi; keduanya punya use case berbeda |

> **Takeaway utama**: Arsitektur NLP terus berkembang untuk mengatasi bottleneck skalabilitas. Pahami trade-off setiap pendekatan — tidak ada arsitektur yang "terbaik" untuk semua kasus.

**Selanjutnya → Fase 3: Rekayasa Sistem** — Cukup teori model. Saatnya membangun sistem AI yang nyata.

---
---

# 📦 FASE 3: Rekayasa Sistem NLP

> *"Di industri, 80% pekerjaan bukan melatih model — tapi merangkai model yang sudah ada menjadi sistem yang bekerja dengan andal."*

---

## MODUL 3.1 — Advanced RAG: Lebih dari Sekedar "Cari & Tempel"

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

Model reranker populer: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cepat), `BAAI/bge-reranker-v2-m3` (akurat, multilingual).

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

---
---

## MODUL 3.2 — Vector Database & HNSW

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
| **Qdrant** | Open source, Rust (cepat), filter metadata powerful | Self-hosted, performa tinggi |
| **Milvus** | Open source, skalabilitas enterprise, fitur lengkap | Dataset sangat besar (>100M) |
| **Pinecone** | Fully managed, mudah, scale otomatis | Prototype cepat, tidak mau urus infra |
| **Weaviate** | GraphQL API, multimodal, built-in vectorizer | Query kompleks, data multimodal |
| **ChromaDB** | Embeddable, zero setup, in-memory/persistent | Development & testing lokal |
| **pgvector** | Extension PostgreSQL | Sudah pakai Postgres, data < 1M |

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

---
---

## MODUL 3.3 — LoRA & QLoRA: Fine-Tuning Efisien

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Modul 2.1 (struktur Transformer), aljabar linear dasar

---

### 🎯 Tujuan Belajar

- Memahami *mengapa* full fine-tuning model LLM tidak praktis untuk kebanyakan use case
- Menjelaskan intuisi matematis LoRA dari konsep rank matrix
- Memahami perbedaan LoRA dan QLoRA, serta kapan menggunakan masing-masing

---

### 🤔 Kenapa Ini Penting?

Bayangkan kamu ingin membuat model LLM yang jago dalam hukum Indonesia. Kamu punya Llama-3-70B yang sudah pintar secara umum, dan kamu punya 100.000 contoh teks hukum Indonesia.

**Full fine-tuning** (melatih ulang semua 70 miliar parameter): butuh ratusan GPU A100 selama berminggu-minggu. Biaya: jutaan rupiah, tidak realistis untuk mahasiswa atau startup kecil.

**LoRA**: latih hanya ~1% dari parameter dengan satu GPU consumer-grade (RTX 3090 atau bahkan A100 satu buah). Performa mendekati full fine-tuning. Biaya: puluhan ribu rupiah di Google Colab Pro.

---

### 📖 Mengapa Kita Bisa Melatih Hanya 1% Parameter?

**Hipotesis Low-Rank**: Ketika model besar dilatih pada task baru, perubahan pada weight matrix W selama fine-tuning ternyata punya **intrinsic rank yang rendah**.

Artinya: meski W adalah matriks 4096×4096 (16+ juta angka), *perubahan* ΔW yang perlu dipelajari saat fine-tuning bisa direpresentasikan dengan akurat oleh matriks yang jauh lebih kecil.

**Analogi: Editing Foto**

Bayangkan sebuah foto resolusi 4K (8 juta piksel). Kalau kamu hanya mengubah warna foto (tambah kehangatan, kurangi saturasi biru), perubahan tersebut bisa dideskripsikan secara kompak — tidak perlu menyimpan nilai baru untuk semua 8 juta piksel. Cukup simpan "aturan transformasi warna"-nya.

LoRA memanfaatkan properti serupa: perubahan "kepribadian" model saat fine-tuning bisa dikompresi ke representasi yang jauh lebih kecil.

---

### 📖 Matematika LoRA (Intuitif)

Dalam Transformer, setiap operasi linear menggunakan matriks bobot W, misalnya W ∈ ℝ^(d×d) untuk attention projection.

**Fine-tuning penuh**: Update W → W + ΔW. Tapi ΔW punya ukuran yang sama dengan W — mahal.

**LoRA**: Aproksimasi ΔW dengan dekomposisi rank-rendah:
```
ΔW ≈ B × A
di mana:
  A ∈ ℝ^(r×d)   — matriks kecil, r << d
  B ∈ ℝ^(d×r)   — matriks kecil
```

Output yang dimodifikasi menjadi:
```
h = W·x + B·A·x
```

Di mana `r` adalah **rank** — hyperparameter yang menentukan kapasitas LoRA. Biasanya r = 4, 8, 16, atau 32.

**Mengapa ini menghemat?**

Tanpa LoRA: simpan ΔW berukuran d×d = 4096×4096 = **16.7 juta parameter** per layer
Dengan LoRA (r=8): simpan A (8×4096) + B (4096×8) = **65.536 parameter** per layer

Penghematan: **256x lebih sedikit parameter** yang perlu dilatih!

Selama inference, kamu bisa **merge** B×A kembali ke W:
```
W_new = W + B×A
```
Hasilnya adalah model dengan ukuran yang sama tapi sudah "dimodifikasi" — tanpa overhead saat inference.

---

### 📖 QLoRA: LoRA + Quantization

**Quantization** adalah teknik mengkompresi model dengan mengurangi presisi angka-angka di dalamnya.

Normal: setiap parameter disimpan sebagai `float32` (4 byte) atau `float16` (2 byte)
Quantized: setiap parameter disimpan sebagai `int8` (1 byte) atau `int4` (0.5 byte)

Hasilnya:
- Model 70B dalam float16: ~140 GB VRAM → butuh 2-3 GPU A100 80GB
- Model 70B dalam 4-bit NF4: ~35 GB VRAM → **muat di 1 GPU A100 40GB!**

**QLoRA** (Dettmers et al., 2023) menggabungkan:
1. Load model dalam presisi 4-bit (hemat memori drastis)
2. Terapkan LoRA adapters dalam presisi 16-bit (untuk menjaga kualitas training)
3. Gunakan "double quantization" dan "paged optimizer" untuk efisiensi memori lebih lanjut

Hasilnya: fine-tune model 65 miliar parameter dengan satu GPU 48GB dengan kualitas yang hampir setara full fine-tuning.

---

### 💻 Kode: Fine-Tuning dengan LoRA (PEFT)

```python
# pip install transformers peft accelerate bitsandbytes datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# === STEP 1: Load model dalam 4-bit (QLoRA) ===
model_name = "facebook/opt-1.3b"  # Pakai yang kecil dulu untuk belajar

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,              # quantize ke 4-bit
    bnb_4bit_quant_type="nf4",     # NormalFloat4 — tipe quantization terbaik untuk LLM
    bnb_4bit_compute_dtype=torch.float16,  # komputasi dalam float16
    bnb_4bit_use_double_quant=True  # double quantization untuk hemat lebih banyak
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"  # otomatis distribusi ke GPU yang tersedia
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === STEP 2: Konfigurasi LoRA ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # tipe task: causal language modeling
    r=8,                             # rank — semakin besar, semakin ekspresif tapi lebih berat
    lora_alpha=32,                   # scaling factor (biasanya 2-4x dari r)
    lora_dropout=0.1,               # regularisasi
    target_modules=["q_proj", "v_proj"],  # lapisan mana yang di-LoRA
    # Biasanya: attention projection layers (q, k, v, o)
    # Untuk model berbeda, nama modulenya berbeda — cek dokumentasi!
)

# === STEP 3: Wrap model dengan LoRA ===
model = get_peft_model(model, lora_config)

# Lihat berapa banyak parameter yang dilatih vs total
model.print_trainable_parameters()
# Output contoh: "trainable params: 2,097,152 || all params: 1,316,143,104 || trainable%: 0.16%"
# Kamu hanya melatih 0.16% dari semua parameter!

# === STEP 4: Training (outline) ===
# Di produksi, tambahkan:
# - Dataset loading (datasets library)
# - DataCollator untuk padding
# - Trainer dari HuggingFace (paling mudah) atau training loop manual
# - Evaluasi di validation set
# - Simpan adapter dengan: model.save_pretrained("my-lora-adapter")

# === STEP 5: Load dan gunakan adapter yang sudah dilatih ===
# from peft import PeftModel
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# model_dengan_lora = PeftModel.from_pretrained(base_model, "my-lora-adapter")

# Untuk merge ke dalam model (hapus overhead LoRA saat inference):
# merged_model = model_dengan_lora.merge_and_unload()
```

---

### ⚠️ Jebakan Umum

**Jebakan 1: "LoRA rank yang lebih tinggi selalu lebih baik"**
Rank lebih tinggi = lebih ekspresif, tapi juga lebih lambat dan bisa overfit pada dataset kecil. Mulai dengan r=8 atau r=16, lalu naik jika perlu.

**Jebakan 2: "Satu LoRA adapter untuk semua task"**
LoRA adapter spesifik untuk task yang ia latih. Adapter yang dilatih untuk teks hukum akan perform buruk pada teks medis. Untuk multi-task, kamu bisa train beberapa adapter terpisah dan swap-in sesuai kebutuhan.

**Jebakan 3: "Quantization 4-bit selalu menghasilkan model yang sama baiknya"**
Ada degradasi kualitas. Untuk task yang sangat butuh presisi (matematika, kode kompleks), 8-bit atau float16 mungkin lebih baik dari 4-bit meski lebih berat.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan secara matematis mengapa LoRA bisa menghemat memori drastis. Berikan contoh numerik: berapa parameter yang dilatih untuk satu attention layer berukuran 4096×4096 dengan rank r=8?

**Level 2 — Aplikasi:**
Jalankan kode di atas dengan model `facebook/opt-125m` (model terkecil yang feasible). Ubah nilai `r` menjadi 4, 8, 16, dan 64. Catat jumlah trainable parameters di setiap setting. Plot hasilnya.

**Level 3 — Eksplorasi:**
Cari paper "LIMA: Less Is More for Alignment" (Zhou et al., 2023). Apa yang ditemukan tentang jumlah data yang diperlukan untuk fine-tuning yang baik? Bagaimana ini mengubah cara kamu akan merencanakan fine-tuning proyek?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Full Fine-Tuning | Latih semua parameter — mahal, tidak realistis untuk kebanyakan orang |
| LoRA | Aproksimasi ΔW dengan matriks rank rendah; hemat 100x+ parameter |
| QLoRA | LoRA + 4-bit quantization; fine-tune 70B dengan 1 GPU |
| r (rank) | Kapasitas LoRA; mulai dari 8-16, naik jika perlu |

> **Takeaway utama**: LoRA memdemokratisasi fine-tuning LLM. Kamu tidak butuh datacenter untuk melatih model domain-specific — cukup satu GPU dan data yang bagus.

**Selanjutnya → Modul 3.4: Evaluasi** — Bagaimana tahu kalau sistemmu bagus atau tidak?

---
---

## MODUL 3.4 — Evaluasi RAG: RAGAS & LLM-as-a-Judge

> **Estimasi waktu**: 2–3 jam
> **Prerequisite**: Modul 3.1 (RAG)

---

### 🎯 Tujuan Belajar

- Memahami mengapa metrik tradisional (BLEU, ROUGE) tidak cocok untuk RAG
- Menjelaskan dimensi evaluasi RAG: Faithfulness, Answer Relevance, Context Precision
- Mengimplementasikan evaluasi sederhana menggunakan LLM-as-a-Judge

---

### 📖 Masalah Evaluasi di RAG

Pertanyaan: "Siapa presiden Indonesia saat ini?"
Jawaban RAG: "Berdasarkan data kami, presiden Indonesia adalah Joko Widodo."
Jawaban Referensi (ground truth): "Prabowo Subianto."

BLEU score: mungkin 0.1 (karena kata-katanya sangat berbeda dari referensi)
Tapi masalah sebenarnya: sistem RAG mengambil dokumen lama yang tidak ter-update. Ini masalah *retrieval*, bukan masalah *generation*.

Evaluasi RAG yang baik harus bisa **mendiagnosis di mana sistem gagal** — apakah di retrieval, di generation, atau di keduanya.

---

### 📖 Dimensi Evaluasi RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment) mendefinisikan 4 metrik:

**1. Faithfulness** — Apakah jawaban hanya berisi klaim yang didukung oleh context yang diambil?
- Nilai 0-1; 1 = semua klaim dalam jawaban bisa ditelusuri ke context
- Ini mengukur apakah model "berhalusinasi" atau tidak
- Contoh gagal: model menjawab "Harga adalah Rp 50.000" padahal context tidak menyebutkan angka

**2. Answer Relevance** — Apakah jawaban menjawab pertanyaan yang ditanyakan?
- Nilai 0-1; 1 = jawaban sangat relevan dengan pertanyaan
- Ini mengukur apakah model menjawab sesuai topik
- Contoh gagal: query "cara reset password" tapi jawaban membahas "cara daftar akun"

**3. Context Precision** — Apakah chunk yang diambil memang relevan dengan pertanyaan?
- Nilai 0-1; 1 = semua chunk yang diambil relevan
- Ini mengukur kualitas retrieval
- Contoh gagal: 3 dari 5 chunk yang diambil tidak relevan sama sekali

**4. Context Recall** — Apakah semua informasi yang diperlukan ada dalam chunk yang diambil?
- Nilai 0-1; 1 = semua informasi yang ada di ground truth answer juga ada di retrieved context
- Ini mengukur kelengkapan retrieval

---

### 📖 LLM-as-a-Judge: Evaluasi Berbasis AI

Menilai "apakah jawaban ini faithful?" secara otomatis adalah tugas yang kompleks — terlalu rumit untuk rule-based metrics. Solusi modern: gunakan LLM yang kuat (GPT-4, Claude) sebagai *penilai otomatis*.

**Cara kerjanya (untuk Faithfulness)**:
1. Berikan ke LLM-judge: context yang diambil + jawaban yang dihasilkan
2. Prompt: *"Periksa setiap klaim dalam jawaban. Apakah setiap klaim bisa didukung oleh informasi dalam context? Beri penilaian 1-5 dan jelaskan."*
3. Parse output LLM untuk mendapatkan skor

**Kelemahan yang perlu disadari**:
- LLM-judge bisa bias terhadap gaya bahasa yang formal atau panjang
- LLM yang sama dengan yang digunakan untuk generate jawaban tidak baik sebagai judge (bias positif)
- Masih ada ketidakkonsistenan — eval yang sama bisa memberi skor berbeda jika dijalankan dua kali

---

### 💻 Kode: Evaluasi Faithfulness Sederhana

```python
# Implementasi LLM-as-a-Judge tanpa framework — agar kamu benar-benar paham cara kerjanya

import json

# Simulasi output RAG
context = """
Kebijakan cuti karyawan: Setiap karyawan berhak atas 12 hari cuti per tahun.
Pengajuan cuti harus dilakukan minimal 3 hari kerja sebelumnya melalui sistem HR.
Cuti tidak bisa dikumulasikan ke tahun berikutnya.
"""

jawaban_rag = """
Karyawan berhak mendapat 12 hari cuti per tahun. Pengajuan cuti dilakukan
minimal 3 hari kerja sebelumnya. Karyawan juga bisa mendapat cuti tambahan
jika kinerja sangat baik.
"""
# Perhatikan: kalimat terakhir TIDAK ADA dalam context — ini halusinasi!

def evaluasi_faithfulness(context, jawaban, llm_client):
    """
    Gunakan LLM untuk menilai apakah setiap klaim dalam jawaban
    didukung oleh context.
    
    llm_client: function yang menerima prompt dan return string response
    """
    prompt = f"""
Kamu adalah evaluator yang teliti. Tugasmu adalah menilai apakah jawaban berikut
HANYA berisi informasi yang ada dalam context yang diberikan.

CONTEXT:
{context}

JAWABAN YANG DIEVALUASI:
{jawaban}

Instruksi:
1. Pecah jawaban menjadi klaim-klaim individual
2. Untuk setiap klaim, tentukan apakah didukung oleh context (1) atau tidak (0)
3. Hitung faithfulness = jumlah_klaim_didukung / total_klaim

Respond HANYA dengan JSON format berikut:
{{
  "klaim": [
    {{"teks": "...", "didukung": true/false, "alasan": "..."}},
    ...
  ],
  "faithfulness_score": 0.XX,
  "kesimpulan": "..."
}}
"""
    response = llm_client(prompt)
    try:
        return json.loads(response)
    except:
        return {"error": "Gagal parse JSON", "raw": response}

# Contoh penggunaan (kamu perlu replace dengan API call nyata):
def dummy_llm(prompt):
    # Ganti ini dengan: openai.chat.completions.create(...) atau anthropic.messages.create(...)
    return json.dumps({
        "klaim": [
            {"teks": "Karyawan berhak 12 hari cuti per tahun", "didukung": True, 
             "alasan": "Tercantum eksplisit di context"},
            {"teks": "Pengajuan minimal 3 hari kerja sebelumnya", "didukung": True,
             "alasan": "Tercantum di context"},
            {"teks": "Cuti tambahan jika kinerja sangat baik", "didukung": False,
             "alasan": "Informasi ini TIDAK ADA dalam context — ini halusinasi!"}
        ],
        "faithfulness_score": 0.67,
        "kesimpulan": "2 dari 3 klaim didukung context. Ada 1 halusinasi."
    })

hasil = evaluasi_faithfulness(context, jawaban_rag, dummy_llm)
print(json.dumps(hasil, indent=2, ensure_ascii=False))
```

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Faithfulness | Apakah jawaban hanya berisi info dari context? (deteksi halusinasi) |
| Answer Relevance | Apakah jawaban menjawab pertanyaan? |
| Context Precision/Recall | Kualitas dan kelengkapan retrieval |
| LLM-as-a-Judge | Gunakan LLM kuat untuk menilai output LLM lain secara otomatis |

> **Takeaway utama**: Evaluasi yang baik adalah yang bisa mendiagnosis di mana sistem gagal — bukan hanya memberikan satu angka.

**Selanjutnya → Fase 4: Agentic AI** — Dari sistem yang *menjawab* ke sistem yang *bertindak*.

---
---

# 📦 FASE 4: Agentic AI & Orkestrasi

> *"RAG menjawab pertanyaan. Agents menyelesaikan masalah."*

---

## MODUL 4.1 — AI Agents: Dari Chatbot ke Sistem Otonom

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Seluruh Fase 3

---

### 🎯 Tujuan Belajar

- Memahami perbedaan fundamental antara LLM chatbot dan AI Agent
- Menjelaskan loop ReAct (Reason + Act) yang mendasari semua agent
- Memahami cara LLM "memanggil" alat eksternal (Function Calling)
- Mengerti konsep Multi-Agent Systems dan kapan mereka diperlukan

---

### 🤔 Kenapa Ini Penting?

Chatbot: kamu tanya → model jawab → selesai. Satu putaran.

AI Agent: kamu kasih tujuan → model *merencanakan*, *mengambil alat yang tepat*, *mengeksekusi*, *melihat hasilnya*, *merevisi rencana*, *mengeksekusi lagi*... sampai tujuan tercapai atau batas tertentu.

Ini adalah pergeseran paradigma dari LLM sebagai "oracle yang menjawab" ke LLM sebagai "agen yang bekerja." Dan ini adalah arah yang sedang dan akan terus berkembang pesat.

---

### 📖 Anatomi AI Agent

Sebuah AI Agent memiliki komponen:

```
┌────────────────────────────────────────────────────────┐
│                     AI AGENT                          │
│                                                        │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐  │
│  │  OTAK    │────▶│ PLANNING │────▶│   MEMORY     │  │
│  │  (LLM)   │◀────│ & REASON │     │ (Short/Long) │  │
│  └──────────┘     └──────────┘     └──────────────┘  │
│       │                                                │
│       ▼                                                │
│  ┌────────────────────────────────────────────────┐   │
│  │               TOOLS / ACTIONS                  │   │
│  │  [Web Search] [Code Exec] [DB Query] [API Call]│   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

**Otak (LLM)**: Mengambil keputusan — apa yang harus dilakukan selanjutnya?
**Planning & Reasoning**: Memecah tujuan besar menjadi langkah-langkah kecil yang bisa dieksekusi
**Memory**: 
- *Short-term* = conversation history (dalam context window)
- *Long-term* = vector database, file, database
**Tools**: Kemampuan untuk berinteraksi dengan dunia luar

---

### 📖 ReAct: Loop Berpikir-Bertindak

**ReAct** (Reason + Act, Yao et al., 2022) adalah paradigma paling fundamental untuk AI agents. Alurnya:

```
[THINK] Apa yang harus saya lakukan untuk mencapai tujuan ini?
[ACT]   Jalankan tool X dengan parameter Y
[OBS]   Lihat hasil tool X
[THINK] Berdasarkan hasil ini, apa langkah berikutnya?
[ACT]   Jalankan tool Z...
[OBS]   ...
[FINAL] Tujuan tercapai → berikan jawaban final
```

**Contoh nyata**: "Cari tahu berapa kurs dollar hari ini, lalu hitung total belanja $150 dalam rupiah"

```
[THINK] Saya perlu dua hal: (1) kurs USD-IDR hari ini, (2) kalkulator.
[ACT]   Tool: web_search("kurs dollar hari ini")
[OBS]   "1 USD = 16.350 IDR (26 Juni 2026)"
[THINK] Dapat kursnya. Sekarang hitung: 150 × 16.350.
[ACT]   Tool: calculator("150 * 16350")
[OBS]   "2.452.500"
[THINK] Sudah punya semua informasi. Susun jawaban.
[FINAL] "Dengan kurs 1 USD = Rp 16.350, belanja senilai $150 setara Rp 2.452.500"
```

Ini terlihat sederhana, tapi sangat powerful. Model bisa menjalankan loop ini ratusan kali untuk task yang kompleks.

---

### 📖 Function Calling / Tool Use: Bagaimana LLM "Memegang" Alat

Secara teknis, LLM tidak bisa langsung "menjalankan kode" atau "searching internet." Yang terjadi adalah:

1. Kamu mendefinisikan **daftar tool yang tersedia** beserta skema input/output-nya (dalam format JSON)
2. Ketika LLM memutuskan perlu menggunakan tool, ia **mengeluarkan output terstruktur** yang berisi: nama tool + parameter yang akan dikirimkan
3. **Runtime (kode Python kamu)** yang benar-benar memanggil tool tersebut
4. Hasil tool dikembalikan ke LLM sebagai observation
5. LLM melanjutkan reasoning

```
Kamu → [definisi tools] → LLM
LLM  → [tool call: {"name": "web_search", "args": {"query": "..."}}] → Kamu
Kamu → [jalankan web_search() di Python] → [hasil]
Kamu → [hasil] → LLM
LLM  → [lanjutkan reasoning] → ...
```

LLM tidak punya agency sejati — ia hanya "menulis instruksi" dan kamu yang mengeksekusinya. Tapi dari sudut pandang hasil, efeknya sama: model bisa "berinteraksi dengan dunia luar."

---

### 💻 Kode: AI Agent Sederhana dengan Function Calling

```python
import json
import math
from anthropic import Anthropic  # atau openai

client = Anthropic()

# === STEP 1: Definisikan tools yang tersedia ===
tools = [
    {
        "name": "kalkulator",
        "description": "Hitung operasi matematika. Gunakan untuk semua perhitungan numerik.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ekspresi": {
                    "type": "string",
                    "description": "Ekspresi matematika yang valid, contoh: '150 * 16350' atau 'sqrt(144)'"
                }
            },
            "required": ["ekspresi"]
        }
    },
    {
        "name": "konversi_suhu",
        "description": "Konversi suhu antar skala (Celsius, Fahrenheit, Kelvin)",
        "input_schema": {
            "type": "object",
            "properties": {
                "nilai": {"type": "number", "description": "Nilai suhu yang akan dikonversi"},
                "dari": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]},
                "ke": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]}
            },
            "required": ["nilai", "dari", "ke"]
        }
    }
]

# === STEP 2: Implementasi tools (ini kode Python biasa, bukan LLM!) ===
def jalankan_tool(nama_tool, args):
    if nama_tool == "kalkulator":
        try:
            # Eval ekspresi matematika (hati-hati di produksi — ini unsafe untuk input sembarang!)
            hasil = eval(args["ekspresi"], {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
            return f"Hasil: {hasil}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif nama_tool == "konversi_suhu":
        nilai, dari, ke = args["nilai"], args["dari"], args["ke"]
        # Konversi ke Celsius dulu
        if dari == "fahrenheit": celsius = (nilai - 32) * 5/9
        elif dari == "kelvin": celsius = nilai - 273.15
        else: celsius = nilai
        # Konversi dari Celsius ke target
        if ke == "fahrenheit": hasil = celsius * 9/5 + 32
        elif ke == "kelvin": hasil = celsius + 273.15
        else: hasil = celsius
        return f"{nilai}° {dari.capitalize()} = {hasil:.2f}° {ke.capitalize()}"

# === STEP 3: ReAct Loop ===
def jalankan_agent(pertanyaan_user):
    print(f"\n🎯 Task: {pertanyaan_user}\n")
    messages = [{"role": "user", "content": pertanyaan_user}]
    
    while True:
        # Kirim ke LLM
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )
        
        # Cek apakah LLM mau pakai tool atau sudah selesai
        if response.stop_reason == "end_turn":
            # Model sudah selesai, tidak perlu tool lagi
            jawaban = response.content[0].text
            print(f"✅ Jawaban Final: {jawaban}")
            return jawaban
        
        # Model mau pakai tool
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"🔧 Tool: {block.name}({block.input})")
                hasil = jalankan_tool(block.name, block.input)
                print(f"   Hasil: {hasil}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": hasil
                })
        
        # Kembalikan hasil tool ke model
        messages.append({"role": "user", "content": tool_results})

# Test
jalankan_agent("Berapa akar kuadrat dari 2025, dan berapa suhu 37 Celsius dalam Fahrenheit?")
```

---

### 📖 Multi-Agent Systems: Kenapa Satu Agent Tidak Cukup

Satu agent bisa menyelesaikan banyak task. Tapi ada skenario di mana beberapa agent yang berkolaborasi lebih baik:

**Skenario 1: Task yang terlalu panjang untuk satu context window**
Penelitian 100 halaman tidak muat dalam satu context window. Pecah: Agent Extractor (ekstrak fakta kunci per bab), Agent Synthesizer (gabungkan temuan), Agent Writer (tulis laporan).

**Skenario 2: Specialized expertise**
Tidak ada satu agent yang jago di segalanya. Agent Researcher (search & summarize), Agent Critic (temukan kelemahan argumen), Agent Coder (implementasi) — masing-masing dengan system prompt yang berbeda.

**Skenario 3: Parallelism**
Task yang bisa dikerjakan secara paralel. Agent A riset tentang topik 1, Agent B riset topik 2, Agent C riset topik 3 — semua bersamaan, lalu hasilnya digabungkan.

**Framework yang sering digunakan**:
- **LangGraph**: Definisikan agents sebagai nodes dalam graph, edges adalah aliran informasi/kontrol
- **AutoGen** (Microsoft): Lebih dekat ke "agen yang berdialog satu sama lain"
- **CrewAI**: Abstraksi tingkat tinggi untuk "tim" agent dengan peran yang jelas

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Lebih banyak agent = lebih baik"**
Agent overhead adalah nyata: setiap agent menambah latency, biaya, dan kompleksitas debugging. Mulai dengan satu agent, tambah hanya jika ada bottleneck yang jelas.

**Jebakan 2: "Agent bisa dipercaya sepenuhnya untuk mengeksekusi aksi permanen"**
Agent masih bisa membuat kesalahan. Untuk aksi yang tidak bisa di-undo (kirim email ke klien, hapus data, transaksi finansial), selalu ada konfirmasi manusia (human-in-the-loop) sebelum eksekusi.

**Jebakan 3: "Prompt injection tidak relevan untuk internal tools"**
Jika agent kamu membaca dokumen dari internet atau dari user lain, dokumen tersebut bisa mengandung instruksi tersembunyi yang "membajak" agent — ini disebut **prompt injection**. Ini adalah ancaman keamanan nyata yang perlu dimitigasi.

---

### 🧩 Latihan

**Level 1 — Recall:**
Gambar diagram ReAct loop untuk skenario ini: "Cari harga saham Apple hari ini, bandingkan dengan harga 1 tahun lalu, dan tentukan persentase perubahannya." Berapa langkah Thought-Action-Observation yang dibutuhkan?

**Level 2 — Aplikasi:**
Tambahkan tool baru ke agent di atas: `konversi_mata_uang(jumlah, dari, ke)` yang menggunakan nilai tukar hardcoded. Buat task yang memaksa agent menggunakan kombinasi tools: kalkulator + konversi mata uang.

**Level 3 — Eksplorasi:**
Cari dan baca tentang **"LLM Agents Benchmark"** seperti SWE-bench (agent yang solve GitHub issues) atau AgentBench. Bagaimana performa model terbaik saat ini? Apa task yang masih sulit bagi agent AI? Apa implikasinya untuk pengembangan agent di masa depan?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| AI Agent | LLM + Loop + Tools; dari "menjawab" ke "menyelesaikan masalah" |
| ReAct | Think → Act → Observe → Think → ...; loop fundamental semua agent |
| Function Calling | LLM "menulis instruksi" tool, kode Python yang mengeksekusinya |
| Multi-Agent | Beberapa agent spesialis berkolaborasi; gunakan hanya jika satu agent tidak cukup |

> **Takeaway utama**: AI Agent adalah LLM yang diberi kemampuan untuk "bertindak" di dunia nyata — tapi tetap membutuhkan desain yang cermat agar aman dan andal.

---
---

# 🏁 Penutup & Roadmap Selanjutnya

## Apa yang Sudah Kamu Pelajari

```
✅ FASE 1: Fondasi
   ├── Math: Dot Product, Cosine Similarity, Ruang Berdimensi Tinggi
   ├── Tokenization: BPE, SentencePiece, implikasi token ≠ kata
   └── Embeddings: Static vs Contextual, Sentence Embeddings

✅ FASE 2: Arsitektur  
   ├── Transformer: Self-Attention, QKV, Multi-Head, Positional Encoding
   ├── Modern: MoE, FlashAttention, Mamba/SSM
   └── Base vs Instruct Model

✅ FASE 3: Sistem
   ├── Advanced RAG: Hybrid Search, Semantic Chunking, Reranking
   ├── Vector Database: HNSW, Qdrant
   ├── Fine-Tuning: LoRA, QLoRA
   └── Evaluasi: RAGAS, LLM-as-a-Judge

✅ FASE 4: Agentic AI
   ├── AI Agent: Anatomy, Memory, Tools
   ├── ReAct: Loop Think-Act-Observe
   ├── Function Calling: Cara teknis tool use
   └── Multi-Agent: Kolaborasi agen spesialis
```

## Langkah Berikutnya yang Disarankan

**Bulan 1-2 — Implementasikan proyek mini dari setiap fase:**
- Fase 1: Visualisasi embedding similarity pada 100 kalimat Bahasa Indonesia
- Fase 2: Visualisasi attention patterns pada kalimat ambigu
- Fase 3: RAG pipeline untuk dokumen PDF dengan hybrid search + reranker
- Fase 4: Agent sederhana yang bisa search web + execute code

**Bulan 3-4 — Pilih satu proyek portfolio:**
Pilih salah satu dari tiga ide proyek di study plan originalmu (Fact-Checking Pipeline, Legal Assistant dengan QLoRA, atau Academic Researcher). Bangun end-to-end.

**Resource Lanjutan:**
- Paper: "Attention Is All You Need" (Vaswani et al., 2017) — baca setidaknya sekali
- Blog: The Illustrated Transformer (Jay Alammar) — visualisasi terbaik yang ada
- Course: Stanford CS224N (YouTube) — untuk yang mau lebih dalam ke riset
- Praktik: HuggingFace Spaces — deploy model dan lihat feedback nyata

---

*Selamat belajar, Fahmi. Dunia NLP bergerak cepat — tapi fondasi yang kuat akan membuatmu bisa mengikuti perkembangannya dengan mudah.*
