# MODUL 3.3 — LoRA & QLoRA: Fine-Tuning Efisien

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Modul 2.1 (struktur Transformer), aljabar linear dasar

---

### 🎯 Tujuan Belajar

- Memahami *mengapa* full fine-tuning model LLM tidak praktis untuk kebanyakan use case
- Menjelaskan intuisi matematis LoRA dari konsep rank matrix
- Memahami perbedaan LoRA dan QLoRA, serta kapan menggunakan masing-masing

---

### 🤔 Kenapa Ini Penting?

Bayangkan kamu ingin membuat model LLM yang jago dalam hukum Indonesia. Kamu punya Llama 4 Scout yang sudah pintar secara umum, dan kamu punya 100.000 contoh teks hukum Indonesia.

**Full fine-tuning** (melatih ulang semua parameter): butuh ratusan GPU A100/H100 selama berminggu-minggu. Biaya: puluhan juta rupiah, tidak realistis untuk mahasiswa atau startup kecil.

**LoRA**: latih hanya ~1% dari parameter dengan satu GPU consumer-grade (RTX 4090 atau satu buah A100). Performa mendekati full fine-tuning. Biaya: puluhan ribu rupiah di Google Colab Pro atau modal pribadi.

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

### 📖 Perkembangan Terbaru: DoRA, rsLoRA, dan LoRA+

Sejak LoRA asli (2021), banyak variasi yang meningkatkan performanya:

**DoRA (Weight-Decomposed Low-Rank Adaptation, 2024)** — Memisahkan magnitude dan arah pada weight matrix sebelum menerapkan LoRA, menghasilkan fine-tuning yang lebih stabil dan mendekati full fine-tuning.

**rsLoRA (Rank-Stabilized LoRA)** — Menyesuaikan scaling factor agar performa tetap stabil pada rank yang lebih tinggi.

**LoRA+ (2024)** — Menggunakan learning rate yang berbeda untuk matriks A dan B, meningkatkan kecepatan konvergensi.

Framework seperti **Unsloth** (2024-2025) menyediakan implementasi yang dioptimasi 2-5x lebih cepat dari PEFT standar dengan penggunaan memori yang lebih rendah — sangat berguna untuk fine-tuning pada GPU consumer.

---

### 💻 Kode: Fine-Tuning dengan LoRA (PEFT)

```python
# pip install transformers peft accelerate bitsandbytes datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# === STEP 1: Load model dalam 4-bit (QLoRA) ===
model_name = "Qwen/Qwen3-0.6B"  # Pakai yang kecil dulu untuk belajar

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
Jalankan kode di atas dengan model `Qwen/Qwen3-0.6B` (model kecil yang feasible). Ubah nilai `r` menjadi 4, 8, 16, dan 64. Catat jumlah trainable parameters di setiap setting. Plot hasilnya.

**Level 3 — Eksplorasi:**
Cari paper "LIMA: Less Is More for Alignment" (Zhou et al., 2023). Apa yang ditemukan tentang jumlah data yang diperlukan untuk fine-tuning yang baik? Bagaimana ini mengubah cara kamu akan merencanakan fine-tuning proyek?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Full Fine-Tuning | Latih semua parameter — mahal, tidak realistis untuk kebanyakan orang |
| LoRA | Aproksimasi ΔW dengan matriks rank rendah; hemat 100x+ parameter |
| QLoRA | LoRA + 4-bit quantization; fine-tune model besar dengan 1 GPU |
| DoRA/rsLoRA | Variasi LoRA terbaru yang lebih stabil dan mendekati full fine-tuning |
| r (rank) | Kapasitas LoRA; mulai dari 8-16, naik jika perlu |

> **Takeaway utama**: LoRA memdemokratisasi fine-tuning LLM. Kamu tidak butuh datacenter untuk melatih model domain-specific — cukup satu GPU dan data yang bagus.

**Selanjutnya → Modul 3.4: Evaluasi** — Bagaimana tahu kalau sistemmu bagus atau tidak?
