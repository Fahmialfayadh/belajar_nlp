# MODUL 2.1 — Self-Attention & Transformer: Teori Mendalam

> **Estimasi waktu**: 6–8 jam (fase paling padat)
> **Prerequisite**: Seluruh Fase 1

---

### 🎯 Tujuan Belajar

- Menjelaskan secara tepat *mengapa* Self-Attention lebih baik dari RNN untuk memahami konteks
- Memahami mekanisme Query-Key-Value (QKV) secara intuitif dan matematis
- Mampu menjelaskan Multi-Head Attention dan Positional Encoding dari first principles

---

### 🤔 Kenapa Ini Penting?

Transformer adalah fondasi dari hampir semua model language AI modern: Gemini, Claude, Llama, Mistral, Qwen, DeepSeek — semuanya Transformer atau variannya. Kalau kamu melamar kerja sebagai AI Engineer dan tidak bisa menjelaskan cara kerja Self-Attention, kamu akan gugur di technical interview.

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

Model modern (Llama, Mistral, Qwen, Gemma) menggunakan **Rotary Positional Embedding (RoPE)** yang lebih elegan dan bisa menangani context window yang lebih panjang. Beberapa model 2025-2026 juga mulai menggunakan variasi RoPE yang di-extend (YaRN, NTK-aware scaling) untuk mendukung context window 128K–1M token.

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

Block ini **diulang N kali**: BERT-base = 12 block, Llama 4 Scout = 80 block, DeepSeek-V4-Pro = ratusan block. Semakin besar model, semakin banyak block — dan semakin dalam "pemikiran" yang bisa dilakukan.

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

**Selanjutnya → Modul 2.2: Arsitektur Modern** — Transformer standar sudah berumur hampir 9 tahun. Industri sudah jauh berkembang.
