# MODUL 1.2 — Tokenization Modern: BPE & SentencePiece

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

**SentencePiece** (digunakan oleh Gemini, Llama, Mistral, dan sebagian besar model modern) mengatasi ini dengan memperlakukan teks sebagai **aliran byte mentah** — tanpa asumsi tentang spasi atau batasan kata. Spasi sendiri menjadi karakter yang bisa digabungkan (direpresentasikan sebagai `▁`).

Hasilnya:
- "Hello world" → ["▁Hello", "▁world"]
- "Helloworld" → ["▁Hello", "world"] *(perhatikan: tidak ada `▁` di "world" karena tidak ada spasi sebelumnya)*

#### Deep-Dive: SentencePiece Punya Dua Mode

Ini sering dilewatkan — SentencePiece bukan cuma "BPE tanpa spasi":

**Mode 1 — BPE**: Sama seperti BPE standar yang sudah kamu pelajari, tapi dijalankan pada aliran byte mentah tanpa asumsi spasi. Dipakai oleh GPT-series, Llama.

**Mode 2 — Unigram Language Model**: Pendekatan yang *sepenuhnya berbeda* dari BPE:

```
BPE:         Bottom-up  → mulai dari karakter, GABUNG pasangan tersering
Unigram LM:  Top-down   → mulai dari vocabulary besar, HAPUS token yang paling sedikit
                           mengurangi likelihood corpus
```

Cara kerja Unigram LM:
1. Mulai dengan vocabulary besar (~1 juta kandidat token)
2. Untuk setiap token, hitung: *"Berapa besar penurunan likelihood corpus jika token ini dihapus?"*
3. Hapus token yang paling sedikit mengurangi likelihood (token yang paling "tidak berguna")
4. Ulangi sampai mencapai ukuran vocabulary target

**Mengapa ini menarik?** Unigram LM bisa menghasilkan **multiple valid tokenizations** untuk satu kata yang sama, sehingga lebih robust terhadap noise. Model seperti Gemini dan T5 menggunakan mode ini.

---

### 📖 Update 2026: Byte Latent Transformer (BLT) — Masa Depan Tanpa Tokenizer

Ini adalah *breakthrough* terbaru yang patut kamu ketahui, meskipun belum mainstream.

**Masalah fundamental semua tokenizer**: Vocabulary tetap (fixed vocabulary) selalu punya batasan:
- Kata baru yang tidak ada di training data → dipecah secara sub-optimal
- Bahasa non-Latin (Arab, Thai, Jepang) sering diperlakukan kurang efisien
- Typo kecil bisa menghasilkan tokenisasi yang sangat berbeda

**Byte Latent Transformer (BLT)** dari Meta membuang konsep tokenizer tetap sepenuhnya:

```
Tokenizer tradisional:
  "Halo dunia" → [fixed_token_1, fixed_token_2]  (vocabulary statis)

BLT:
  "Halo dunia" → [72, 97, 108, 111, 32, 100, ...]  (raw UTF-8 bytes!)
                  → [patch_1, patch_2, ...]  (dynamic grouping berdasarkan entropy)
```

Cara BLT mengelompokkan bytes menjadi patches:
- Bagian teks yang **mudah diprediksi** (kata umum, pola berulang) → dikompresi jadi patch besar
- Bagian teks yang **sulit diprediksi** (nama, angka, kode, kata asing) → patch kecil, mendapat lebih banyak compute

**Keunggulan BLT**:
- **Tidak ada OOV** — semua teks bisa diproses karena bekerja di level byte
- **Sempurna untuk multilingual** — semua bahasa diperlakukan setara
- **Adaptive compute** — model mengalokasikan lebih banyak "pemikiran" untuk bagian yang sulit

**Status**: Masih riset, belum di-adopt di model produksi utama. Tapi arahnya jelas — masa depan mungkin tanpa tokenizer tetap.

#### Parity-Aware BPE: Mengurangi "Token Tax" Multilingual

Masalah: tokenizer yang dilatih dominan pada data Inggris membuat bahasa lain "lebih mahal":

```
Kalimat setara:
  English:   "I love programming"  → 3 tokens
  Indonesia: "Saya suka programming" → 5 tokens  ← 67% lebih banyak!
```

**Parity-Aware BPE** (2025-2026) secara eksplisit mengoptimasi merge rules agar **panjang token seimbang antar bahasa** — bukan cuma memaksimalkan kompresi pada bahasa mayoritas. Ini masalah fairness: pengguna berbahasa Indonesia seharusnya tidak membayar 40% lebih mahal untuk API LLM.

---

### 📖 Implikasi Penting yang Sering Diabaikan

#### Token ≠ Kata
Kalimat "Saya belajar NLP" dalam tokenizer Llama 4:
- "Saya" → mungkin 1-2 token
- " belajar" → mungkin 1-2 token
- " NLP" → mungkin 1-2 token (akronim sering dipecah)

Implikasinya: **biaya API LLM dihitung per token, bukan per kata.** Teks dalam bahasa Indonesia rata-rata menggunakan **20-40% lebih banyak token** dari teks bahasa Inggris yang setara maknanya — karena tokenizer dilatih dominan pada data Inggris. Meski gap ini sudah berkurang di model-model 2025-2026 yang dilatih lebih multilingual.

#### Sensitivitas Kapitalisasi
"kucing" dan "Kucing" sering menghasilkan token ID yang berbeda. Ini berarti model bisa berperilaku berbeda hanya karena huruf kapital di awal kalimat. Ini bukan bug — ini konsekuensi dari cara BPE belajar dari teks asli.

---

### 💻 Kode: Visualisasi Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer Llama 3.x (representatif untuk model-model modern berbasis BPE/SentencePiece)
# Alternatif jika tidak punya akses: gunakan "gpt2" sebagai fallback
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

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
Tidak. Context window dihitung dalam token. "Gemini 3.5 Flash punya context window 1 juta token" ≠ 1 juta kata. Dalam bahasa Indonesia, mungkin hanya ~600.000–700.000 kata.

**Jebakan 2: "Tokenizer tidak perlu dipahami dalam produksi"**
Sangat perlu. Kalau kamu melakukan chunking dokumen untuk RAG dan memotong di 512 *karakter*, kamu bisa memotong di tengah-tengah token (bahkan di tengah karakter multi-byte seperti emoji atau aksara non-Latin), yang menghasilkan output garbage.

**Jebakan 3: "Semua model punya tokenizer yang sama"**
Tidak. Gemini, Llama, Claude, Mistral — semuanya punya vocabulary dan tokenizer yang berbeda. Kamu tidak bisa mengukur "panjang" teks dalam token tanpa mengetahui tokenizer model yang akan kamu gunakan.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan dalam kalimatmu sendiri: mengapa BPE lebih baik dari tokenization berbasis kata utuh (word-level)? Sebutkan minimal 2 alasan konkret.

**Level 2 — Aplikasi:**
Jalankan kode di atas dengan tokenizer yang berbeda: coba `"bert-base-uncased"` dan `"xlm-roberta-base"` (multilingual). Berapa jumlah token untuk kalimat Indonesia yang sama? Tokenizer mana yang paling efisien untuk Bahasa Indonesia?

**Level 3 — Eksplorasi:**
Cari tahu tentang **"tokenizer fertility"** — metrik yang mengukur rata-rata token per kata untuk bahasa tertentu. Temukan data fertility tokenizer Llama 4 atau Gemini 3.5 untuk beberapa bahasa. Apa implikasinya terhadap biaya dan fairness penggunaan LLM di negara non-Inggris?

---

### 🔬 Eksperimen Google Colab: Tokenizer Battle Royale

Copy-paste ke Google Colab — tidak butuh GPU.

```python
# ============================================================
# EKSPERIMEN: Tokenizer Battle — 5 Tokenizer Head-to-Head
# ============================================================
# Bandingkan efisiensi tokenizer untuk Bahasa Indonesia vs Inggris
# Ukur "token tax" — berapa % lebih mahal bahasa Indonesia?

!pip install -q transformers tiktoken sentencepiece protobuf

from transformers import AutoTokenizer
import tiktoken

# Kalimat test paralel (makna setara Indonesia ↔ Inggris)
test_pairs = [
    ("Pemerintah Indonesia mengumumkan kebijakan baru tentang energi terbarukan.",
     "The Indonesian government announced new policies on renewable energy."),
    ("Mahasiswa itu sedang mengerjakan tugas akhirnya di perpustakaan kampus.",
     "The student was working on their final thesis at the campus library."),
    ("Kecerdasan buatan telah mengubah cara kita berinteraksi dengan teknologi.",
     "Artificial intelligence has changed how we interact with technology."),
    ("Dia mempertanggungjawabkan perbuatannya di depan pengadilan negeri.",
     "He was held accountable for his actions before the district court."),
]

# Load tokenizers (masing-masing mewakili strategi tokenisasi berbeda)
tokenizers = {
    "GPT-4o (tiktoken)": tiktoken.encoding_for_model("gpt-4o"),
    "Llama-3.2": AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
    "Qwen2.5": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
    "Gemma-2": AutoTokenizer.from_pretrained("google/gemma-2-2b"),
    "BERT multilingual": AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
}

print("=" * 90)
print(f"{'Tokenizer':<22} {'ID Tokens':<12} {'EN Tokens':<12} {'Rasio ID/EN':<12} {'Token Tax %'}")
print("=" * 90)

for tok_name, tok in tokenizers.items():
    total_id, total_en = 0, 0
    for id_text, en_text in test_pairs:
        if isinstance(tok, tiktoken.Encoding):
            id_tokens = len(tok.encode(id_text))
            en_tokens = len(tok.encode(en_text))
        else:
            id_tokens = len(tok.encode(id_text))
            en_tokens = len(tok.encode(en_text))
        total_id += id_tokens
        total_en += en_tokens
    
    ratio = total_id / total_en
    tax = (ratio - 1) * 100
    bar = "🟢" if tax < 15 else "🟡" if tax < 30 else "🔴"
    print(f"{tok_name:<22} {total_id:<12} {total_en:<12} {ratio:<12.2f} {bar} {tax:+.1f}%")

print("\n💡 INSIGHT: Token Tax = berapa persen lebih mahal Bahasa Indonesia")
print("   dibanding Inggris pada tokenizer yang sama.")
print("   Model yang dilatih lebih multilingual → tax lebih rendah.")
print("\n🔬 COBA: Tambahkan kalimat dengan typo, emoji, atau kode program!")
```

> **Yang perlu kamu amati**: Perhatikan tokenizer mana yang paling "adil" untuk Bahasa Indonesia. Apakah model yang lebih baru (Qwen, Gemma) lebih baik dari yang lama (BERT)? Ini relevan karena menentukan berapa biaya API yang kamu bayar.

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| BPE | Algoritma kompresi yang belajar memecah kata ke subwords dari data |
| SentencePiece | BPE/Unigram LM yang tidak bergantung pada spasi, lebih universal |
| Unigram LM | Top-down approach: mulai besar, hapus token yang tidak berguna |
| BLT (2026) | Tanpa tokenizer tetap — proses raw bytes dengan dynamic patching |
| Token ≠ Kata | Bahasa Indonesia lebih "mahal" secara token dari bahasa Inggris |
| Parity-Aware BPE | Optimasi BPE untuk keadilan multilingual (kurangi token tax) |

> **Takeaway utama**: Tokenization bukan langkah teknis yang bisa diabaikan — ia adalah "bahasa" yang digunakan model untuk membaca dunia, dan pilihannya punya konsekuensi nyata. Masa depan mungkin tanpa tokenizer tetap (BLT), tapi untuk sekarang, pahami cara BPE/SentencePiece bekerja.

**Selanjutnya → Modul 1.3: Embeddings** — Setelah teks jadi token, setiap token diubah menjadi vektor. Tapi tidak semua vektor diciptakan sama.
