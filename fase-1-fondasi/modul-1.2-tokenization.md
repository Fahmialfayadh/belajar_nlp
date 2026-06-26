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

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| BPE | Algoritma kompresi yang belajar memecah kata ke subwords dari data |
| SentencePiece | BPE yang tidak bergantung pada spasi, lebih universal |
| Token ≠ Kata | Bahasa Indonesia lebih "mahal" secara token dari bahasa Inggris |
| Implikasi Produksi | Tokenizer berpengaruh pada biaya API, chunking, dan perilaku model |

> **Takeaway utama**: Tokenization bukan langkah teknis yang bisa diabaikan — ia adalah "bahasa" yang digunakan model untuk membaca dunia, dan pilihannya punya konsekuensi nyata.

**Selanjutnya → Modul 1.3: Embeddings** — Setelah teks jadi token, setiap token diubah menjadi vektor. Tapi tidak semua vektor diciptakan sama.
