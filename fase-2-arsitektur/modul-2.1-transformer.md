# MODUL 2.1 — Self-Attention & Transformer: Teori Mendalam

> **Estimasi waktu**: 6–8 jam (fase paling padat)
> **Prerequisite**: Seluruh Fase 1

---

## 🎯 Tujuan Belajar

- Menjelaskan secara tepat *mengapa* Self-Attention lebih baik dari RNN untuk memahami konteks
- Memahami mekanisme Query-Key-Value (QKV) secara intuitif dan matematis
- Mampu menjelaskan Multi-Head Attention dan Positional Encoding dari first principles

---

## 🤔 Kenapa Ini Penting?

Transformer adalah fondasi dari hampir semua model language AI modern: Gemini, Claude, Llama, Mistral, Qwen, DeepSeek — semuanya Transformer atau variannya. Kalau kamu melamar kerja sebagai AI Engineer dan tidak bisa menjelaskan cara kerja Self-Attention, kamu akan gugur di technical interview.

Lebih penting dari itu: memahami Transformer akan membuat semua konsep lanjutan — fine-tuning, RAG, attention patterns, context window — masuk akal secara intuitif, bukan sekadar "tool yang digunakan."

---

## 📖 Masalah yang Dipecahkan Transformer

Sebelum 2017, arsitektur dominan untuk pemrosesan teks berurutan adalah RNN (Recurrent Neural Network) dan LSTM. Mereka punya masalah fundamental:

### Masalah 1 — Sequential Processing (tidak bisa paralel)

RNN memproses teks satu token per waktu:

```
token₁ → token₂ → token₃ → ... → token₁₀₀
```

Token ke-100 baru bisa diproses **setelah** token ke-1 sampai ke-99 selesai. Tidak ada cara untuk memparalelkan proses ini di GPU — pelatihan menjadi sangat lambat, terutama untuk teks panjang.

### Masalah 2 — Vanishing Gradient untuk konteks jauh

Bayangkan kalimat:

> "The cat that my neighbor's dog chased **was** very scared."

Untuk memahami bahwa "was" harus singular (karena subjeknya "cat", bukan "dog"), model harus mengingat informasi dari 7 kata sebelumnya. Pada RNN, informasi ini harus "diteruskan" lewat setiap sel secara berurutan. Semakin panjang jaraknya, semakin lemah sinyal yang sampai — inilah *vanishing gradient*: gradients mengecil secara eksponensial saat backpropagation melewati banyak langkah.

### Solusi Transformer

Setiap token langsung "melihat" semua token lain **secara bersamaan** — tidak ada urutan, tidak ada jarak yang membatasi. Dan semuanya bisa dikomputasi secara paralel di GPU.

```
RNN:   token₁ → token₂ → token₃ → token₄   (berurutan, lambat)
       
Transformer:
       token₁ ↔ token₂
       token₁ ↔ token₃  (semua ke semua, sekaligus, paralel)
       token₁ ↔ token₄
       token₂ ↔ token₃
       ... dst
```

---

## 📖 Self-Attention: Intuisi dari Nol

### Pertanyaan inti

Ketika memproses kata "dia" dalam kalimat:

> "Budi pergi ke toko karena **dia** butuh beras"

...bagaimana model tahu bahwa "dia" merujuk ke "Budi"?

Model harus bisa mengukur: *seberapa relevan setiap kata lain terhadap kata yang sedang diproses?* Itulah Self-Attention — setiap kata memberi "perhatian" (attention) yang berbeda-beda kepada semua kata lain, lalu menggunakan bobot perhatian tersebut untuk memperbarui representasi dirinya sendiri.

### Analogi: Konferensi Meja Bundar

Bayangkan setiap kata dalam kalimat adalah seseorang yang duduk di meja bundar. Setiap orang membawa tiga hal:

| Peran | Analogi | Makna teknis |
|-------|---------|--------------|
| **Query (Q)** | Pertanyaan yang dibawa: *"Siapa yang relevan denganku?"* | Apa yang kata ini sedang cari dari konteks |
| **Key (K)** | Papan nama di dada: *"Inilah identitasku"* | Penanda identitas kata ini agar bisa dikenali |
| **Value (V)** | Informasi di amplop: *"Inilah yang bisa aku bagikan"* | Konten sesungguhnya yang akan diambil jika relevan |

Proses attention berjalan seperti ini:

1. Kata "dia" mengeluarkan **Query**-nya: *"Siapa subjek yang dirujuk oleh kata ganti?"*
2. Ia membandingkan Query-nya dengan **Key** setiap kata lain di meja
3. **Key** milik kata "Budi" cocok dengan pertanyaan tersebut — skor relevansinya tinggi
4. Maka "dia" mengambil sebagian besar **Value** dari "Budi" untuk memperbarui representasinya sendiri
5. Representasi baru "dia" kini mengandung informasi kontekstual dari "Budi"

Ini terjadi untuk **setiap kata secara bersamaan**, bukan satu per satu.

---

## 📖 Query, Key, Value: Mekanisme Matematis

### Dari embedding ke Q, K, V

Setiap token dalam kalimat sudah direpresentasikan sebagai vektor (dari embedding layer). Sebut saja vektor ini **x** dengan dimensi `d_model` (misalnya 768 untuk BERT-base).

Langkah pertama: proyeksikan **x** menjadi tiga vektor berbeda menggunakan tiga matriks bobot yang *dipelajari saat training*:

```
Q = x · W_Q    # Query  — "apa yang saya cari?"
K = x · W_K    # Key    — "apa identitas saya?"
V = x · W_V    # Value  — "informasi apa yang saya punya?"
```

`W_Q`, `W_K`, `W_V` adalah matriks berukuran `[d_model × d_k]`. Biasanya `d_k` lebih kecil dari `d_model` — misalnya dari 768 menjadi 64 per head. Ketiga matriks ini **berbeda satu sama lain** dan memiliki nilai awal acak yang kemudian dioptimalkan lewat training.

> **Poin penting**: Q, K, V dari token yang sama berasal dari vektor embedding yang sama (**x**), tapi diproyeksikan ke "ruang" yang berbeda. Ini yang membuat mereka bisa memainkan peran berbeda.

### Formula Attention

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Mari kita bongkar ini step by step dengan contoh konkret.

Misalkan kalimat kita: `"Budi pergi ke toko"` — 4 token.  
Setelah proyeksi, kita punya:
- `Q`: matriks 4×64 (Query untuk 4 token)
- `K`: matriks 4×64 (Key untuk 4 token)
- `V`: matriks 4×64 (Value untuk 4 token)

---

### Step 1: `Q · Kᵀ` — Hitung relevansi setiap pasangan

Kalikan matriks Query dengan transpose matriks Key.

```
Q · Kᵀ  →  matriks berukuran [4 × 4]
```

Setiap sel `[i, j]` berisi dot product antara Query token ke-i dan Key token ke-j. Nilai tinggi berarti: *"Query token i merasa Key token j sangat relevan."*

```
         Budi   pergi    ke    toko
Budi   [ 8.2    1.1    0.3    2.1 ]
pergi  [ 7.5    6.8    0.4    1.2 ]
ke     [ 0.8    1.2    5.5    3.4 ]
toko   [ 2.1    0.9    3.1    7.9 ]
```

*(Angka ini ilustratif, bukan hasil nyata)*

Baca per baris: baris "pergi" menunjukkan bahwa token "pergi" paling memperhatikan "Budi" (7.5) dan dirinya sendiri (6.8).

---

### Step 2: `/ √d_k` — Stabilisasi skala

Bagi seluruh matriks dengan `√d_k` (akar dari dimensi Key, misalnya √64 = 8).

**Kenapa perlu ini?** Ketika dimensi vektor besar, nilai dot product cenderung sangat besar. Nilai yang sangat besar membuat softmax pada step berikutnya menghasilkan distribusi yang hampir seluruhnya terpusat pada satu nilai (mendekati one-hot) — artinya gradients hampir nol, dan training mandek. Pembagian `√d_k` menjaga nilai dalam range yang stabil sehingga gradients bisa mengalir dengan baik.

```
         Budi   pergi    ke    toko
Budi   [ 1.03   0.14   0.04   0.26 ]
pergi  [ 0.94   0.85   0.05   0.15 ]
ke     [ 0.10   0.15   0.69   0.43 ]
toko   [ 0.26   0.11   0.39   0.99 ]
```

---

### Step 3: `softmax(...)` — Normalisasi jadi bobot perhatian

Terapkan softmax **per baris**. Setiap baris diubah menjadi distribusi probabilitas yang berjumlah 1.

```
         Budi   pergi    ke    toko
Budi   [ 0.61   0.13   0.09   0.17 ]   → jumlah = 1.0
pergi  [ 0.42   0.38   0.08   0.12 ]   → jumlah = 1.0
ke     [ 0.12   0.13   0.45   0.30 ]   → jumlah = 1.0
toko   [ 0.14   0.11   0.22   0.53 ]   → jumlah = 1.0
```

Inilah **attention weights** — seberapa besar perhatian yang diberikan setiap token kepada token lain. Baris "Budi" mengatakan: token "Budi" memberikan 61% perhatiannya kepada dirinya sendiri, 17% kepada "toko", dan sisanya kepada "pergi" dan "ke".

---

### Step 4: `· V` — Ambil informasi berbobot

Kalikan matriks attention weights dengan matriks Value.

```
Attention Weights [4×4]  ×  V [4×64]  =  Output [4×64]
```

Untuk setiap token, representasi barunya adalah **rata-rata berbobot dari Value semua token**. Token yang mendapat attention weight tinggi menyumbang lebih besar ke representasi akhir.

Representasi baru token "pergi" = 0.42 × V("Budi") + 0.38 × V("pergi") + 0.08 × V("ke") + 0.12 × V("toko")

Representasi baru ini bukan lagi sekadar "kata pergi" — ia kini mengandung konteks dari seluruh kalimat, berbobot sesuai relevansi.

---

### Ringkasan alur Q-K-V

```
Token "Budi"   ──→  Q_budi, K_budi, V_budi
Token "pergi"  ──→  Q_pergi, K_pergi, V_pergi
Token "ke"     ──→  Q_ke,    K_ke,    V_ke
Token "toko"   ──→  Q_toko,  K_toko,  V_toko

                  ↓
    Hitung Q·Kᵀ  →  skor relevansi semua pasangan
    Bagi √d_k    →  stabilkan skala
    Softmax      →  bobot perhatian (attention weights)
    Kalikan ×V   →  representasi baru yang kaya konteks
```

---

## 📖 Multi-Head Attention: Memperhatikan dari Banyak Perspektif

### Masalah dengan satu attention head

Satu attention head hanya bisa "fokus" pada satu jenis pola relasi dalam satu waktu. Tapi bahasa punya banyak jenis relasi yang perlu dipahami secara bersamaan:

- **Relasi sintaktis**: "pergi" adalah predikat dari "Budi"
- **Relasi koreference**: "dia" merujuk ke "Budi"
- **Relasi semantik**: "toko" berkaitan dengan "beli", "barang", "uang"
- **Relasi posisional**: kata ini muncul setelah kata itu

Jika hanya ada satu set W_Q, W_K, W_V, model harus memilih: pola mana yang paling penting? Ia tidak bisa menangkap semuanya sekaligus.

### Solusi: jalankan banyak head secara paralel

**Multi-Head Attention** menjalankan `h` attention head sekaligus. Setiap head punya W_Q, W_K, W_V sendiri yang **independen** — sehingga setiap head bisa belajar memperhatikan jenis relasi yang berbeda.

```
head₁ = Attention(x·W_Q₁, x·W_K₁, x·W_V₁)   # mungkin belajar relasi sintaktis
head₂ = Attention(x·W_Q₂, x·W_K₂, x·W_V₂)   # mungkin belajar koreference
head₃ = Attention(x·W_Q₃, x·W_K₃, x·W_V₃)   # mungkin belajar relasi semantik
...
headₕ = Attention(x·W_Qₕ, x·W_Kₕ, x·W_Vₕ)
```

Semua head dijalankan **paralel** (bukan berurutan). Hasilnya digabungkan (concatenate) lalu diproyeksikan kembali ke dimensi asal:

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · W_O
```

`W_O` adalah matriks proyeksi output berukuran `[h·d_k × d_model]` yang juga dipelajari saat training.

### Dimensi dalam praktik

Untuk **BERT-base** (`d_model` = 768, `h` = 12 heads):
- Setiap head menggunakan `d_k = 768 / 12 = 64`
- 12 head × 64 dimensi = 768 dimensi setelah concat
- Proyeksi W_O mengembalikannya ke 768 dimensi

Jadi total parameter tidak bertambah eksponensial — head yang lebih banyak hanya berarti setiap head bekerja di ruang dimensi yang lebih kecil.

> **Catatan**: Model tidak diberitahu head mana yang harus belajar relasi apa — ini muncul secara otomatis dari training. Penelitian interpretabilitas menunjukkan beberapa head memang secara konsisten memperhatikan relasi tertentu (misalnya subjek-verba), tapi tidak semua head punya peran sesempit itu.

---

## 📖 Positional Encoding: Karena Transformer Tidak Tahu Urutan

### Masalah

Self-Attention adalah operasi yang **tidak sensitif terhadap urutan**. Perhatikan:

- "Budi membeli beras"
- "Beras membeli Budi"

Kedua kalimat mengandung kata yang sama. Matriks attention weights yang dihasilkan akan **identik** karena dot product tidak peduli pada posisi — hanya pada nilai vektor. Tanpa informasi posisi, Transformer tidak bisa membedakan kalimat "kucing mengejar anjing" dari "anjing mengejar kucing."

### Solusi: tambahkan vektor posisi ke embedding

Sebelum masuk ke Transformer block pertama, tambahkan **positional encoding** ke setiap embedding token:

```
input_ke_transformer = embedding_token + positional_encoding(posisi)
```

Positional encoding adalah vektor berukuran sama dengan embedding (`d_model`) yang unik untuk setiap posisi.

### Versi asli: fungsi sinus-kosinus

Paper "Attention Is All You Need" (2017) menggunakan fungsi trigonometri:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

Di mana `pos` adalah indeks posisi token (0, 1, 2, ...) dan `i` adalah indeks dimensi (0, 1, 2, ..., d_model/2).

**Mengapa fungsi trigonometri?**

1. **Nilai terbatas**: sin dan cos selalu antara -1 dan 1, sehingga tidak mendominasi embedding token
2. **Setiap posisi unik**: kombinasi sin/cos di berbagai frekuensi menghasilkan vektor yang berbeda untuk setiap posisi
3. **Relasi antar posisi bisa dipelajari**: karena ada hubungan trigonometri yang konsisten antar posisi, model bisa belajar menghitung "jarak" antara dua posisi
4. **Bisa ekstrapolasi**: model dapat menangani panjang teks yang sedikit melampaui panjang training (dalam batas tertentu)

### Visualisasi intuitif

Bayangkan setiap posisi punya "tanda tangan frekuensi" yang berbeda. Posisi awal mengubah banyak dimensi, posisi akhir mengubah dimensi yang berbeda. Gabungan ini membentuk sidik jari yang unik per posisi:

```
Posisi 0:  [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
            = [0.000, 1.000, 0.000, 1.000, ...]

Posisi 1:  [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
            = [0.841, 0.540, 0.010, 0.9999, ...]

Posisi 2:  [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
            = [0.909, -0.416, 0.020, 0.9998, ...]
```

### RoPE: standar modern

Model modern (Llama, Mistral, Qwen, Gemma) menggunakan **Rotary Positional Embedding (RoPE)** — pendekatan yang lebih elegan: alih-alih menambahkan vektor posisi ke embedding, RoPE *merotasi* vektor Query dan Key berdasarkan posisinya sebelum dot product dihitung. Efeknya adalah dot product Q·K secara otomatis mengandung informasi jarak relatif antara dua token.

Keunggulan RoPE: performa lebih baik untuk context window panjang, dan lebih mudah di-extend. Beberapa model 2025–2026 menggunakan variasi RoPE (YaRN, NTK-aware scaling) untuk mendukung context window hingga 128K–1M token.

---

## 📖 Arsitektur Lengkap Transformer Block

Satu "block" Transformer menggabungkan semua komponen di atas menjadi satu unit yang bisa ditumpuk:

```
Input x (embedding + positional encoding)
        │
        ▼
┌───────────────────────────────────────┐
│         Multi-Head Self-Attention     │
│                                       │
│  Q, K, V ← proyeksi dari x           │
│  Attention weights ← softmax(QKᵀ/√d) │
│  Output ← weights × V                │
└───────────────────────────────────────┘
        │
        ▼  (tambahkan input asli x — Residual Connection)
        x = x + attention_output
        │
        ▼  (normalisasi — Layer Normalization)
        x = LayerNorm(x)
        │
        ▼
┌───────────────────────────────────────┐
│       Feed-Forward Network (FFN)      │
│                                       │
│  Linear(d_model → d_ff)              │
│  Aktivasi (ReLU / GELU / SwiGLU)     │
│  Linear(d_ff → d_model)              │
└───────────────────────────────────────┘
        │
        ▼  (tambahkan input sebelum FFN — Residual Connection)
        x = x + ffn_output
        │
        ▼  (normalisasi lagi)
        x = LayerNorm(x)
        │
        ▼
   Output → masuk ke block berikutnya
```

Block ini **diulang N kali**: BERT-base = 12 block, GPT-3 = 96 block, Llama 4 Scout = 80 block. Semakin banyak block, semakin dalam "pemikiran" yang bisa dilakukan model.

### Mengapa Residual Connection diperlukan?

Tanpa residual connection, sinyal dari layer awal "hilang" setelah melewati banyak transformasi — inilah *vanishing gradient*. Residual connection menambahkan input langsung ke output setiap sub-layer:

```
output = LayerNorm(x + sub_layer(x))
```

Ini menciptakan "jalan pintas" untuk gradients mengalir langsung dari output ke input selama backpropagation, tanpa harus melewati semua transformasi. Tanpa ini, melatih model dengan 12+ layer hampir mustahil.

### Mengapa ada FFN setelah Attention?

Dua peran yang berbeda:

- **Multi-Head Attention** = operasi "komunikasi" antar token. Setiap token mengumpulkan informasi dari token lain.
- **FFN** = operasi "komputasi" internal per token. Setiap token memproses informasi yang sudah terkumpul secara independen.

Tanpa FFN, model hanya bisa menggabungkan informasi antar token tanpa bisa mengolahnya lebih lanjut. Penelitian menunjukkan FFN bertindak seperti "memori" yang menyimpan pengetahuan faktual dari training data.

Dimensi FFN (`d_ff`) biasanya 4× `d_model`: BERT-base punya `d_ff` = 3072 untuk `d_model` = 768.

---

## 💻 Kode: Visualisasi Attention Weights

```python
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Gunakan BERT-base agar cepat dan gratis
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

# Kalimat dengan ambiguitas koreference — "it" merujuk ke apa?
kalimat = "The cat sat on the mat because it was tired"
inputs = tokenizer(kalimat, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple berisi tensor attention dari setiap layer
# Setiap tensor: shape [batch_size, n_heads, seq_len, seq_len]
attentions = outputs.attentions

print(f"Jumlah layer: {len(attentions)}")          # 12 untuk BERT-base
print(f"Shape per layer: {attentions[0].shape}")   # [1, 12, 11, 11]
print(f"Tokens: {tokens}")

# Ambil layer terakhir (representasi paling abstrak/semantik)
# Head 0 sebagai contoh — coba ganti indeks head untuk eksplorasi
last_layer_attn = attentions[-1][0]  # shape: [12, seq_len, seq_len]
head_idx = 0
head_attn = last_layer_attn[head_idx].numpy()  # shape: [seq_len, seq_len]

# Visualisasi heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    head_attn,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap='Blues',
    annot=True,
    fmt='.2f',
    annot_kws={"size": 7}
)
plt.title(f"Attention Weights — Layer 12, Head {head_idx}")
plt.xlabel("Key (token yang diperhatikan)")
plt.ylabel("Query (token yang memperhatikan)")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
print("Gambar disimpan: attention_heatmap.png")
```

**Yang perlu kamu amati setelah menjalankan kode:**

1. Cari baris untuk token `"it"` pada heatmap. Token mana yang mendapat warna paling gelap (attention weight tertinggi)?
2. Apakah itu `"cat"`? Jika iya, head ini berhasil menangkap relasi koreference.
3. Ganti `head_idx` dari 0 ke 1, 2, ..., 11. Apakah setiap head menunjukkan pola berbeda?
4. Coba perhatikan diagonal (setiap token memperhatikan dirinya sendiri) — apakah selalu dominan?

> **Ekspektasi realistis**: Tidak semua head akan menunjukkan pola yang mudah diinterpretasi. Beberapa head menunjukkan pola yang tampak acak atau mendominasi pada token `[CLS]` dan `[SEP]`. Ini normal — interpretabilitas attention masih area riset aktif.

---

## ⚠️ Jebakan Umum

### Jebakan 1 — "Attention weight tinggi = model 'mengerti' relasi tersebut"

Tidak sesederhana itu. Penelitian tentang *attention interpretability* masih aktif diperdebatkan. Attention weights tinggi menunjukkan *korelasi*, bukan kausalitas. Model bisa mencapai jawaban benar dengan pola attention yang terlihat "salah" secara linguistik — dan sebaliknya, pola attention yang "masuk akal" belum tentu yang benar-benar mendorong prediksi.

### Jebakan 2 — "Context window = memori model"

Transformer tidak punya "memori" dalam arti tradisional. Ia memproses seluruh context window *sekaligus* setiap kali ada input baru — tidak ada rekam jejak dari percakapan sebelumnya kecuali dimasukkan eksplisit ke dalam context. Ini berbeda dari memori manusia yang bersifat episodik dan persisten.

### Jebakan 3 — "Lebih banyak layer = selalu lebih baik"

Ada trade-off. Layer awal cenderung menangkap fitur lokal dan sintaktis (struktur frasa), layer tengah menangkap relasi semantik, layer akhir paling task-specific. Tapi menambah layer tanpa menambah data training dan regularisasi yang proporsional akan menghasilkan overfitting — model menghafal data training, bukan belajar generalisasi.

### Jebakan 4 — "Self-Attention = O(n²) selalu jadi bottleneck"

Kompleksitas komputasi Self-Attention memang O(n²) terhadap panjang sequence — untuk 1000 token, ada 1.000.000 pasangan yang dihitung. Tapi dalam praktik modern, bottleneck sering ada di tempat lain (FFN, memory bandwidth). Ada juga varian efficient attention (FlashAttention, Sparse Attention, Linear Attention) yang mengurangi kompleksitas ini untuk sequence sangat panjang.

---

## 🧩 Latihan

### Level 1 — Recall

Gambar diagram di kertas yang menunjukkan flow satu token melalui satu Transformer block:
embedding → (+positional encoding) → Multi-Head Attention → residual → LayerNorm → FFN → residual → LayerNorm → output.

Tandai di mana Q, K, V terbentuk, dan di mana hasil attention digabungkan.

### Level 2 — Aplikasi

Modifikasi kode di atas untuk memvisualisasikan **semua 12 head sekaligus** dalam satu figure (gunakan `plt.subplot`). Perhatikan:
- Head mana yang paling memperhatikan token `"it"` ke `"cat"`?
- Head mana yang paling memperhatikan token ke dirinya sendiri (diagonal)?
- Head mana yang polanya paling "acak" atau sulit diinterpretasi?

### Level 3 — Eksplorasi

Cari dan baca ringkasan paper **"Are Sixteen Heads Really Better than One?"** (Michel et al., 2019). Pertanyaan panduan:
- Apa yang terjadi jika sebagian besar attention head di-prune (dihilangkan) saat inference?
- Head mana yang paling "penting" dan head mana yang redundan?
- Apa implikasinya untuk efisiensi model dan asumsi kita tentang Multi-Head Attention?

---

## 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|----------------|
| **Self-Attention** | Setiap token memperhatikan semua token lain secara bersamaan; tidak ada "jarak" yang membatasi; bisa dikomputasi paralel |
| **Q (Query)** | Proyeksi "apa yang token ini cari" dari konteks sekitarnya |
| **K (Key)** | Proyeksi "identitas token ini" agar bisa dikenali oleh Query token lain |
| **V (Value)** | Proyeksi "informasi sesungguhnya" yang akan dibagikan jika Key-nya relevan dengan Query |
| **Softmax(QKᵀ/√d_k)** | Menghasilkan attention weights — distribusi probabilitas seberapa besar perhatian ke setiap token |
| **Multi-Head Attention** | Beberapa attention head paralel; masing-masing menangkap jenis relasi berbeda secara otomatis |
| **Positional Encoding** | Ditambahkan ke embedding agar model tahu urutan token — tanpanya, anagram dan kalimat normal terlihat identik |
| **Residual + LayerNorm** | Memungkinkan training model sangat dalam tanpa vanishing gradient |
| **FFN** | Komputasi internal per token setelah attention; sering dianggap sebagai "memori" faktual model |

> **Takeaway utama**: Transformer memungkinkan setiap token langsung "berkomunikasi" dengan semua token lain dalam satu operasi paralel — tanpa urutan, tanpa jarak. Q-K-V adalah mekanisme untuk mengukur relevansi dan mengambil informasi yang relevan tersebut. Inilah kenapa Transformer jauh lebih powerful dan efisien dari RNN untuk teks.

---

**Selanjutnya → Modul 2.2: Arsitektur Modern** — Transformer standar sudah berumur hampir 9 tahun. Industri sudah jauh berkembang: Flash Attention, Mixture of Experts, GQA, SwiGLU, dan banyak lagi.
