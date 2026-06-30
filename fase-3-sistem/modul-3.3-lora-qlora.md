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

#### Deep-Dive: Rank Selection ($r$) & Scaling Factor ($\alpha$)
Saat melakukan konfigurasi LoRA Config, kamu wajib memahami parameter berikut:
- **Rank ($r$)**: Mengatur lebar matriks dekomposisi low-rank.
  - Heuristik: $r=8$ atau $16$ cocok untuk task instruksi umum/formatting. $r=32$ atau $64$ dibutuhkan untuk pembelajaran domain spesifik yang sangat mendalam (seperti istilah medis baru atau bahasa pemrograman baru).
- **Alpha ($\alpha$)**: Scaling factor untuk menyeimbangkan pengaruh adaptor LoRA terhadap model dasar.
  - Heuristik: Selalu atur $\alpha$ sebesar **$2\times r$** (misal $r=8, \alpha=16$). Menjaga $\alpha$ konstan membantu kestabilan gradient dan kestabilan nilai learning rate saat kamu melakukan tuning rank $r$ yang berbeda.

#### Target Modules: Di Mana Harus Memasang Adapter?
LLM terdiri dari banyak matriks linear. Di layer mana kita harus menempelkan matriks $A$ dan $B$?
- **Attention Modules saja (`q_proj`, `v_proj`)**: Standar LoRA asli (2021). Menghemat parameter paling ekstrem, namun performa kurang maksimal untuk task penalaran berat.
- **All Linear Modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)**: Standar industri modern. Menempelkan adapter ke modul Attention dan FFN sekaligus. Meskipun trainable parameter bertambah sedikit, performa model meningkat signifikan dan mendekati full fine-tuning.

#### Update 2026: DoRA, rsLoRA, LoRA+, dan Unsloth

Beberapa inovasi optimasi adapter yang wajib diketahui:

- **DoRA (Weight-Decomposed Low-Rank Adaptation, 2024)**: Memisahkan perubahan bobot menjadi dua komponen independen: **magnitude** (panjang/skalar) dan **direction** (arah/matriks low-rank). Hal ini secara dramatis meningkatkan stabilitas training dan akurasi model pada task logika.
- **rsLoRA (Rank-Stabilized LoRA)**: Menskalakan adaptor menggunakan $\frac{1}{\sqrt{r}}$ alih-alih $\frac{1}{r}$. Menjaga stabilitas model ketika di-train pada rank $r$ yang besar.
- **LoRA+ (2024)**: Menerapkan learning rate yang $4\times$ lebih besar pada matriks $B$ dibandingkan matriks $A$ untuk mempercepat konvergensi parameter.
- **Unsloth (Framework Produksi Utama)**:
  - Menyediakan kernel CUDA buatan tangan (handwritten CUDA kernels) yang dioptimalkan khusus untuk backpropagation model Llama, Mistral, dan Qwen.
  - Mengurangi pemakaian memori hingga **60%** dan mempercepat training **2x - 5x lebih cepat** daripada PEFT standar tanpa mengurangi akurasi sedikit pun. Sangat direkomendasikan untuk training di GPU lokal/Google Colab gratis.

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

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan secara matematis mengapa dekomposisi low-rank $W_0 + B \times A$ menghemat memori GPU secara signifikan dibandingkan melatih ulang matriks $W_0$ secara utuh.

**Level 2 — Aplikasi:**
Jalankan eksperimen Colab di bawah. Ubah rank `$r$` menjadi `1`, `4`, `16`, dan `32`. Catat bagaimana perubahan rank memengaruhi **Reconstruction Error** (kesalahan aproksimasi) dan jumlah **trainable parameter**. Plot datanya.

**Level 3 — Eksplorasi:**
Mengapa pada model modern kita disarankan memilih opsi **All Linear Modules** untuk penempelan adaptor LoRA alih-alih hanya **Attention Modules (`q_proj`, `v_proj`)**? Bagaimana pengaruhnya terhadap stabilitas model pada tugas reasoning kompleks?

---

### 🔬 Eksperimen Google Colab: Visualisasi LoRA Rank Ablation

Copy-paste kode ini ke Google Colab (CPU runtime) untuk memvisualisasikan bagaimana dekomposisi rank rendah mengaproksimasi perubahan matriks dan menghitung parameter efisiensi secara visual:

```python
# ============================================================
# EKSPERIMEN: Visualisasi Low-Rank Approximation LoRA
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
d = 64  # dimensi model (simulasi)

# Buat perubahan weight delta_W ber-rank rendah asli (misal rank-4)
true_rank = 4
A_true = np.random.randn(d, true_rank) * 0.1
B_true = np.random.randn(true_rank, d) * 0.1
delta_W = A_true @ B_true

# Hitung SVD untuk mendapatkan aproksimasi terbaik berbagai rank
U, S, Vt = np.linalg.svd(delta_W, full_matrices=True)

ranks_to_test = [1, 2, 4, 8, 16]
errors = []

fig, axes = plt.subplots(1, len(ranks_to_test) + 1, figsize=(20, 4))

for i, r in enumerate(ranks_to_test):
    delta_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    error = np.linalg.norm(delta_W - delta_approx, 'fro') / np.linalg.norm(delta_W, 'fro')
    errors.append(error)
    
    # Hitung rasio parameter
    lora_params = 2 * d * r
    full_params = d * d
    compression = (lora_params / full_params) * 100
    
    ax = axes[i]
    ax.imshow(delta_approx, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    ax.set_title(f"Rank {r}\nError: {error:.3f}\nParams: {compression:.1f}%")
    ax.axis('off')

# Plot matriks original
axes[-1].imshow(delta_W, cmap='coolwarm', vmin=-0.2, vmax=0.2)
axes[-1].set_title("Original ΔW\n(True Rank: 4)")
axes[-1].axis('off')

plt.suptitle("LoRA Low-Rank Reconstruction Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| **Full Fine-Tuning** | Melatih seluruh parameter model; sangat mahal dan membutuhkan VRAM GPU raksasa. |
| **LoRA** | Menambahkan adaptor berupa dekomposisi matriks rank rendah ($B \times A$) untuk menguji perubahan bobot. |
| **QLoRA** | Memadukan LoRA dengan pemuatan base model dalam presisi 4-bit (NF4) untuk efisiensi memori tingkat ekstrem. |
| **Rank Heuristics** | Pilih rank `$r=8$` atau `$16$` untuk instruksi umum; perbesar ke `$r=32$` atau `$64$` untuk domain spesifik. |
| **DoRA** | Memisahkan magnitude dan arah weight update untuk menghasilkan performa yang lebih stabil mendekati full fine-tuning. |
| **Unsloth** | Framework optimal dengan CUDA kernel kustom yang mempercepat training hingga 2-5x lebih cepat di GPU. |

> **Takeaway utama**: LoRA mendemokratisasi proses kustomisasi LLM. Kamu tidak perlu memiliki supercomputer untuk melatih model domain-spesifik; cukup satu GPU consumer, teknik QLoRA, dan framework Unsloth.

---

**Selanjutnya → Modul 3.4: Evaluasi** — Bagaimana tahu kalau sistemmu bagus atau tidak?

