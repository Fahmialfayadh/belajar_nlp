# MODUL 2.2 — Arsitektur Mutakhir: MoE, Efficient Attention & Mamba

> **Estimasi waktu**: 3–4 jam  
> **Prerequisite**: Modul 2.1

---

## 🎯 Tujuan Belajar

- Memahami mengapa Transformer standar tidak skalabel untuk model triliunan parameter
- Menjelaskan intuisi Mixture of Experts (MoE) dan trade-off-nya
- Mengenal kelemahan quadratic complexity Attention dan pendekatan solusinya
- Memahami mengapa SSM/Mamba muncul sebagai alternatif dan bagaimana arsitektur hybrid berkembang

---

## 🤔 Kenapa Ini Penting?

Model-model state-of-the-art di 2026 — Gemini 3.5, Claude Fable 5 / Opus 4.8, GPT-5.5, Llama 4 Maverick (MoE), DeepSeek-V4, Qwen3.7 — semua menggunakan variasi dari arsitektur ini. Kalau kamu hanya tahu Transformer "vanilla", kamu akan bingung membaca paper dan dokumentasi model terbaru. Dan kalau kamu bekerja di tim yang memilih atau mengevaluasi model, kamu perlu tahu trade-off arsitektural ini.

---

## 📖 Masalah Skalabilitas Transformer

### Masalah 1 — Biaya Komputasi Quadratic

Self-Attention punya kompleksitas **O(n²)** terhadap panjang sequence, di mana n = jumlah token.

Artinya: doubling panjang context = 4x biaya komputasi. Untuk context window 1 juta token (seperti Gemini 3.5 Flash atau Claude Opus 4.8), attention standar secara harfiah tidak bisa dikomputasi dalam waktu yang wajar tanpa optimasi.

### Masalah 2 — Semua Parameter Aktif untuk Setiap Token

Model dense 70 miliar parameter seperti Llama 3.1-70B mengaktifkan *semua* 70 miliar parameternya untuk memproses *setiap token*. Ini sangat boros — apakah benar-benar semua "pengetahuan" model diperlukan untuk memproses kata "dan"?

---

## 📖 Mixture of Experts (MoE): Otak yang Terspesialisasi

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

### Cara Kerja Router (Gating Network)

Router adalah **jaringan neural kecil** (biasanya linear layer + softmax) yang:
1. Menerima input embedding dari token
2. Menghasilkan probabilitas untuk setiap expert (berapa "cocok" expert tersebut untuk token ini)
3. Memilih top-K expert (biasanya K=2)
4. Output token adalah weighted sum dari output expert yang dipilih

**Contoh code sederhana router:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        # Pilih top-K experts untuk setiap token
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Normalisasi weights untuk top-K saja
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return top_k_indices, top_k_weights
```

### Contoh nyata dari 2024-2026

- **Llama 4 Maverick**: MoE dengan 128 experts, hanya 2 aktif per token — total ~400B parameter tapi hanya ~17B aktif per token
- **DeepSeek-V4-Pro**: 1.6T total parameter, ~49B aktif per token (MoE terbaru April 2026 dengan hybrid attention architecture)
- **Qwen3.7-Max**: Versi MoE dari seri Qwen 3.7, fokus pada agentic workflows

### Hasilnya

- Performa mendekati model dense yang jauh lebih besar
- Biaya komputasi per token jauh lebih rendah

### Trade-off yang perlu kamu ketahui

- Performa tinggi saat inferensi (hanya sebagian parameter aktif)
- Tapi training jauh lebih kompleks dan tidak stabil (router bisa "collapse" — semua token diarahkan ke 1-2 expert yang sama). DeepSeek memperkenalkan teknik *auxiliary-loss-free load balancing* untuk mengatasi ini.
- Memori total tetap besar (semua parameter harus dimuat ke GPU/RAM meski tidak semuanya aktif)

---

## 📖 Efficient Attention: Solusi untuk Quadratic Complexity

Beberapa pendekatan untuk mengatasi O(n²):

### FlashAttention (v1, v2, v3)

Bukan mengubah matematikanya, tapi mengubah *cara komputasi*-nya agar lebih efisien di GPU (memanfaatkan hirarki memori GPU). FlashAttention-3 (2024) bahkan memanfaatkan Tensor Cores pada GPU Hopper (H100). Hasilnya secara matematis identik dengan Attention standar, tapi 2-4x lebih cepat dan hemat memori. Hampir semua framework modern sudah menggunakannya secara default.

**Contoh penggunaan FlashAttention:**

```python
# FlashAttention biasanya sudah built-in di PyTorch 2.2+
import torch.nn.functional as F

# Standard attention (lambat untuk context panjang)
def standard_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

# FlashAttention (cepat dan hemat memori)
def flash_attention(q, k, v):
    # PyTorch 2.2+ secara otomatis menggunakan FlashAttention
    # saat menggunakan scaled_dot_product_attention
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Sparse Attention

Setiap token tidak memperhatikan *semua* token lain, hanya sebagian (token terdekat, token "landmark", dsb.). Mengurangi kompleksitas ke O(n√n) atau O(n log n). Tradeoff: bisa kehilangan informasi dari konteks jauh yang tidak masuk dalam subset.

### Ring Attention & Sequence Parallelism

Mendistribusikan attention computation ke banyak GPU secara efisien, memungkinkan context window yang sangat panjang (1M+ token) tanpa satu GPU pun harus menyimpan seluruh KV-cache.

### Multi-head Latent Attention (MLA)

Diperkenalkan oleh DeepSeek-V2 (2024), MLA mengompresi Key dan Value ke dimensi yang jauh lebih kecil melalui *low-rank joint compression*, mengurangi ukuran KV-cache drastis tanpa mengorbankan kualitas. Ini memungkinkan inferensi yang jauh lebih efisien pada context panjang.

---

## 📖 State Space Models (SSM) & Mamba: Pesaing Transformer

### Konteks historis

Sebelum Transformer, ada model *State Space* dari kontrol sistem dan signal processing. Pada 2022-2023, peneliti mulai mengadaptasinya untuk NLP dengan hasil yang mengejutkan.

### Intuisi SSM

**Analogi sederhana**: Bayangkan kamu membaca buku. 
- **Attention**: Kamu berhenti, melihat kembali semua halaman sebelumnya untuk mencari konteks
- **SSM**: Kamu terus membaca sambil menyimpan "ringkasan" di kepala, yang kamu update setiap membaca kata baru

Alih-alih "memperhatikan semua token sekaligus" (seperti Attention), SSM mempertahankan sebuah *hidden state* yang terus diperbarui saat memproses token satu per satu — mirip RNN, tapi dengan matematika yang jauh lebih stabil dan parallelizable.

### Mamba (Gu & Dao, 2023)

Mamba adalah SSM yang paling berpengaruh. Inovasinya: *selective state spaces* — model bisa memilih secara adaptif informasi mana yang perlu dipertahankan di hidden state berdasarkan input. **Mamba-2** (2024) menyederhanakan arsitektur dan menunjukkan koneksi mendalam antara SSM dan Attention.

**Contoh code sederhana SSM (ilustratif):**

```python
import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Matrix A: how state evolves
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        # Matrix B: how input affects state
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        # Matrix C: how to read out state
        self.C = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        hidden_state = torch.zeros(batch_size, 1, x.size(-1), device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            # Update hidden state: h_{t+1} = A*h_t + B*x_t
            hidden_state = torch.matmul(hidden_state, self.A) + torch.matmul(x_t, self.B)
            # Output: y_t = C*h_t
            y_t = torch.matmul(hidden_state, self.C)
            outputs.append(y_t)
            
        return torch.cat(outputs, dim=1)
```

### Keunggulan Mamba vs Transformer

- Kompleksitas **O(n)** terhadap panjang sequence — linear, bukan quadratic
- Efisien untuk context yang sangat panjang (jutaan token)
- Inferensi lebih cepat karena tidak perlu menyimpan seluruh KV-cache

### Kelemahan Mamba

- Pada benchmark bahasa standar (≤ 4K token), masih kalah dari Transformer dengan ukuran yang sama
- "Recall" informasi dari konteks sangat jauh tidak sebaik Attention
- Ekosistem dan tooling masih lebih kecil dari Transformer

### Status 2026

Arsitektur hybrid (Transformer + SSM) seperti **Jamba 1.5** (AI21), **Zamba** (Zyphra), dan model-model riset lainnya sudah menunjukkan hasil menjanjikan. Beberapa perusahaan juga mengeksplorasi **RWKV-6** dan **Griffin** sebagai alternatif linear-time. Trend utamanya: bukan "Mamba vs Transformer" tapi "bagaimana menggabungkan keunggulan keduanya."

---

## 📖 Base Model vs Instruct Model

Ini adalah konsep yang sering disalahpahami dan kritis untuk dipahami saat memilih model.

### Base Model

**Contoh**: Llama 4 Scout Base, Qwen3-72B-Base

- Dilatih dengan *pre-training* pada teks internet yang sangat besar
- Tujuan satu-satunya: **menebak token berikutnya**
- Jika kamu beri prompt "Cara membuat nasi goreng:", ia mungkin melanjutkan dengan teks acak dari internet yang temanya memasak — bisa jadi resep, bisa jadi review restoran, bisa jadi apa saja
- Tidak "mengerti" instruksi manusia, tidak punya "kepribadian"

### Instruct/Chat Model

**Contoh**: Llama 4 Scout Instruct, Claude Opus 4.8, Gemini 3.5 Flash

- Base model yang kemudian di-*fine-tune* menggunakan data instruksi manusia

### Pipeline: Dari Base ke Instruct

```
Base Model (Pre-training)
         ↓
┌─────────────────────────┐
│  1. SFT                 │  ← Fine-tuning dengan dataset instruksi
│  (Supervised            │     manusia (contoh: "Jelaskan X", "Tulis Y")
│   Fine-Tuning)          │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  2. RLHF / DPO / GRPO   │  ← Alignment dengan preferensi manusia
│  (Reinforcement         │     (pilih output yang lebih baik, lebih aman,
│   Learning from         │     lebih membantu)
│   Human Feedback)       │
└─────────────────────────┘
         ↓
Instruct Model (Siap digunakan)
```

**Penjelasan setiap tahap**:

1. **SFT (Supervised Fine-Tuning)**: Model belajar format instruksi-respon. Dataset berisi ribuan contoh "User: pertanyaan → Assistant: jawaban". Model belajar bahwa format tertentu harus diikuti.

2. **RLHF/DPO/GRPO**: Model belajar *kualitas* output. Manusia menilai beberapa respon, model belajar preferensi (misal: jawaban yang lebih detail lebih baik, hindari konten berbahaya). GRPO adalah teknik baru dari DeepSeek yang lebih efisien.

**Hasil akhir**: Mampu mengikuti instruksi, berdialog, dan menghindari output berbahaya. Punya "kepribadian" yang konsisten.

### Kapan pakai yang mana?

- Research, pre-training experiments, atau kalau kamu mau fine-tune sendiri dari awal → Base model
- Hampir semua use case produksi (chatbot, assistant, RAG) → Instruct model

---

## ⚠️ Jebakan Umum

### Jebakan 1: "MoE berarti model lebih murah untuk di-deploy"

**Kenapa salah**: Kamu tetap perlu muat semua parameter ke GPU (untuk Llama 4 Maverick MoE = ratusan GB). Yang lebih murah adalah biaya *komputasi saat inferensi* — bukan memori. Teknik seperti **expert offloading** (muat expert ke GPU hanya saat dibutuhkan) membantu, tapi menambah latency.

### Jebakan 2: "Mamba akan menggantikan Transformer"

**Kenapa salah**: Masih terlalu dini. Per 2026, Transformer masih mendominasi dan ekosistemnya jauh lebih matang. Mamba dan SSM sangat menjanjikan untuk aplikasi yang butuh context sangat panjang, tapi model-model terbaik (Gemini, Claude, GPT) masih berbasis Transformer (atau hybrid).

### Jebakan 3: "Base model lebih fleksibel jadi lebih baik untuk produksi"

**Kenapa salah**: Base model tidak tahu cara mengikuti instruksi. Kamu bisa fine-tune sendiri, tapi butuh dataset berkualitas tinggi dan eksperimen panjang. Untuk hampir semua use case, Instruct model sudah di-align dan siap pakai. Gunakan Base model hanya jika kamu punya use case yang sangat spesifik.

### Jebakan 4: "FlashAttention mengubah matematika attention"

**Kenapa salah**: FlashAttention hanya mengubah *cara komputasi* untuk efisiensi hardware (tiling, memanfaatkan SRAM GPU). Outputnya secara matematis identik dengan attention standar. Ini bukan arsitektur baru, ini optimasi engineering.

### Jebakan 5: "Context window besar = model lebih pintar"

**Kenapa salah**: Context window besar memungkinkan model memproses lebih banyak informasi sekaligus, tapi bukan berarti model lebih memahami atau lebih akurat. Bahkan, banyak model mengalami "lost in the middle" — informasi di tengah context panjang sering diabaikan.

### Jebakan 6: "MoE tidak pernah collapse"

**Kenapa salah**: Router di MoE bisa mengalami *load imbalance* — beberapa expert terlalu sering dipakai, yang lain tidak pernah. Ini bikin training tidak stabil. Teknik seperti *auxiliary loss* atau *expert dropout* dibutuhkan untuk memaksa distribusi yang merata.

---

## 🧩 Latihan

### Level 1 — Recall

Jelaskan dengan analogi sederhana: apa perbedaan antara Dense Model (Transformer biasa) dan Sparse Model (MoE)? Mengapa ini penting untuk efisiensi?

### Level 2 — Eksplorasi

Cari perbandingan benchmark DeepSeek-V4-Pro vs Llama 4 Maverick. Pada task apa MoE unggul? Pada task apa ia tidak unggul? Apa implikasinya untuk pemilihan model di produksi?

### Level 3 — Riset

Cari paper atau blogpost yang membahas arsitektur **Jamba 1.5** atau **Mamba-2**. Apa klaim keunggulannya? Bagaimana mereka menggabungkan SSM dan Attention? Apakah ada benchmark yang memvalidasi klaim tersebut?

---

## 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| MoE | Banyak "expert" kecil; hanya sebagian yang aktif per token — efisiensi tinggi |
| Quadratic Attention | Masalah skalabilitas; FlashAttention & MLA mempercepat tanpa mengubah output |
| Mamba/SSM | Alternatif O(n) untuk sequence panjang; hybrid dengan Transformer paling menjanjikan |
| Base vs Instruct | Base = prediksi token; Instruct = ikuti instruksi; keduanya punya use case berbeda |

> **Takeaway utama**: Arsitektur NLP terus berkembang untuk mengatasi bottleneck skalabilitas. Pahami trade-off setiap pendekatan — tidak ada arsitektur yang "terbaik" untuk semua kasus.

---

**Selanjutnya → Fase 3: Rekayasa Sistem** — Cukup teori model. Saatnya membangun sistem AI yang nyata.
```

- ✅ Menggunakan line breaks yang jelas antar section

Markdown ini siap dipakai untuk dokumentasi atau materi pembelajaran!
