# MODUL 1.1 — Matematika Terapan untuk NLP Modern

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

Seluruh kemampuan LLM — dari Gemini sampai Claude — berakar pada satu operasi matematika sederhana: **perkalian vektor**. Kalau kamu tidak paham ini di level intuisi, kamu hanya akan jadi *pengguna* library, bukan *insinyur* yang bisa men-debug ketika sesuatu salah.

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

Model seperti BERT merepresentasikan setiap token sebagai vektor **768 dimensi**. Model frontier seperti Gemini 3.5 Pro atau GPT-5.5 bisa menggunakan lebih dari 16.000 dimensi. Kenapa sebesar itu?

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
3. Presiden Jerman → Frank-Walter Steinmeier (2026)

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
Tidak. Ada trade-off: lebih banyak dimensi = lebih ekspresif, tapi juga lebih lambat dan butuh lebih banyak data training untuk mengisi dimensi tersebut dengan bermakna. Model embedding ringan seperti `BAAI/bge-m3` atau `Qwen3-Embedding` dengan dimensi 768–1024 sudah sangat berguna untuk banyak task pencarian semantik.

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
