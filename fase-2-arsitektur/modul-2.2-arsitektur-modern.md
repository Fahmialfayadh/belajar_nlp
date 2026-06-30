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

### Deep-Dive: Expert Collapse & Load Balancing

Membangun router MoE bukan sekadar membuat linear layer biasa. Masalah terbesar dalam melatih MoE adalah **Expert Collapse**:

```
Mula-mula: Router sedikit condong ke Expert A.
  ↓
Expert A menerima lebih banyak gradient update karena sering dipanggil.
  ↓
Expert A menjadi lebih pintar dibandingkan expert lain.
  ↓
Router semakin condong mengarahkan token ke Expert A.
  ↓
Expert collapse: Expert A terbebani penuh, expert lain menganggur (idle).
```

Jika ini terjadi, model MoE akan berperilaku seperti model *dense* kecil, menyia-nyiakan kapasitas dari parameter expert lainnya.

#### Solusi Klasik: Auxiliary Loss
Secara tradisional (seperti pada Mixtral), peneliti menambahkan fungsi loss tambahan (**auxiliary loss**) selama training. Loss ini menghukum router jika distribusinya tidak merata:
- Menghitung frekuensi pemilihan setiap expert dalam satu batch.
- Menghitung entropi dari probabilitas perutean.
- Menambahkan penalty ke loss total jika router memilih expert yang sama secara berlebihan.
*Trade-off*: Auxiliary loss yang terlalu besar dapat menurunkan performa model karena mengorbankan kualitas perutean demi keadilan (fairness).

#### Update 2026: Auxiliary-Loss-Free Load Balancing (DeepSeek-V3/V4)
DeepSeek memperkenalkan terobosan penting untuk melatih model skala triliunan parameter tanpa degradasi performa:
- **Bias Dinamis**: Alih-alih menambahkan auxiliary loss ke fungsi optimasi utama, DeepSeek menggunakan bias dinamis pada logits router.
- Jika sebuah expert terpilih melebihi kapasitas idealnya, bias pengurang ditambahkan pada logits expert tersebut untuk menurunkan probabilitas pemilihannya pada iterasi berikutnya.
- Teknik ini menjaga beban expert tetap seimbang (load balancing) tanpa mengganggu representasi gradient model utama, sehingga performa tetap optimal.

### Trade-off MoE yang Perlu Kamu Ketahui

- **Kecepatan Inferensi Tinggi**: Hanya sebagian kecil parameter yang diaktivasi per token (misal: 2 dari 8 expert).
- **RAM/VRAM Raksasa**: Meskipun parameter aktif kecil, **seluruh parameter expert harus tetap dimuat di memori (VRAM/RAM)**. Deployment model MoE triliunan parameter membutuhkan GPU cluster yang sangat besar.
- **Komunikasi Latensi Tinggi**: Jika expert tersebar di beberapa GPU berbeda, terjadi overhead jaringan yang signifikan (all-to-all communication) untuk memindahkan representasi token ke GPU tempat expert tersebut berada.

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

Diperkenalkan oleh DeepSeek-V2/V3/V4, **MLA** mengatasi batasan memori dari **KV-cache** yang sangat membengkak saat melayani jutaan token secara bersamaan.

#### Konteks: Masalah KV-Cache (MQA vs GQA)
- **MHA (Multi-Head Attention)**: Setiap Query head memiliki Key head dan Value head sendiri. VRAM KV-cache tersedot sangat cepat.
- **MQA (Multi-Query Attention)**: Semua Query head berbagi 1 Key head dan 1 Value head. Menghemat VRAM hingga $\approx 8\times$, tapi kualitas model turun drastis karena representasi Key/Value menjadi terlalu sempit.
- **GQA (Grouped-Query Attention)**: Query head dikelompokkan (misal 8 head per group), dan setiap kelompok berbagi 1 Key dan Value head (standar Llama-3). Ini adalah jalan tengah yang baik.

#### Solusi MLA: Low-Rank Joint Compression
MLA memotong kebutuhan KV-cache lebih ekstrem daripada GQA **tanpa menurunkan kualitas representasi model** melalui kompresi joint rank-rendah (low-rank projection):

1. **Kompresi Key dan Value**: Alih-alih menyimpan vektor $K$ dan $V$ asli yang berdimensi besar untuk setiap token, MLA memproyeksikan keduanya secara bersamaan ke dalam satu ruang latent berdimensi sangat kecil ($d_c \ll d_{\text{model}}$, misal 512 dimensi):
   $$\mathbf{c}_t^{KV} = \text{Linear}_{\text{down}}(\mathbf{h}_t)$$
   Di mana $\mathbf{c}_t^{KV}$ adalah representasi terkompresi yang disimpan dalam KV-cache.
2. **Rekonstruksi saat Inferensi**: Saat model melakukan komputasi attention, model melakukan up-projection secara instan untuk merekonstruksi Key dan Value dari latent vector tersebut:
   $$\mathbf{k}_t = \mathbf{c}_t^{KV} W_{UK}, \quad \mathbf{v}_t = \mathbf{c}_t^{KV} W_{UV}$$
3. **Optimasi RoPE**: Karena RoPE (Rotary PE) tidak kompatibel dengan kompresi matriks linear di atas (rotasi merusak properti dekomposisi matriks), MLA memisahkan sebagian kecil dimensi Query dan Key untuk diberi RoPE secara independen, kemudian digabungkan (concatenated) kembali.

#### Hasil Akhir MLA
MLA menghemat ukuran KV-cache hingga **93%** dibandingkan MHA standar, bahkan melampaui efisiensi GQA, namun dengan performa representasi yang tetap setara dengan MHA penuh. Hal inilah yang memungkinkan model DeepSeek melayani context window 128K+ token pada traffic produksi yang sangat padat secara murah.

---

## 📖 State Space Models (SSM) & Mamba: Pesaing Transformer

### Konteks Historis
Sebelum Transformer mendominasi, recurrent neural networks (RNN) adalah standar untuk sequence modeling. RNN memiliki sifat komputasi sekuensial yang sangat lambat dilatih. Pada 2022-2023, peneliti mengadopsi kembali **State Space Models (SSM)** dari bidang teori kontrol dan signal processing untuk NLP, menciptakan model dengan efisiensi RNN namun bisa dilatih secara paralel.

### Intuisi SSM vs Attention
- **Attention**: Saat memproses token baru, model melihat kembali *seluruh* token masa lalu di KV-cache. (Kompleksitas memori $\mathcal{O}(n^2)$).
- **SSM**: Model memproses token satu per satu secara linear, namun memperbarui satu ringkasan state internal (**hidden state**) yang komprehensif. (Kompleksitas memori $\mathcal{O}(n)$).

### Mamba-1, Mamba-2, dan Mamba-3 (Masa Depan Hybrid)

- **Mamba-1 (2023)**: Memperkenalkan *Selective State Spaces* — kemampuan matriks transisi SSM untuk berubah secara dinamis berdasarkan konten input token. Hal ini memungkinkannya "memilih" informasi mana yang perlu diingat dan mana yang perlu dibuang.
- **Mamba-2 (2024)**: Memperkenalkan **Structured State Space Duality (SSD)**. Mamba-2 menyederhanakan struktur matriks transisi agar secara matematis setara dengan attention semi-sparse. Perubahan ini memungkinkan Mamba-2 diparalelkan secara penuh menggunakan GPU Tensor Cores, meningkatkan kecepatan training hingga $2\times$ sampai $8\times$.
- **Mamba-3 (Maret 2026 Update)**: Dirilis untuk menyempurnakan kelemahan klasik Mamba dalam hal *associative recall* (kemampuan mengingat asosiasi fakta acak yang sangat jauh) yang biasanya menjadi keunggulan mutlak Transformer. Mamba-3 mengoptimalkan interaksi selective state dan meminimalkan parameter redundancy.

#### Keunggulan Mamba vs Transformer
- **Kompleksitas Linear $\mathcal{O}(n)$**: Biaya komputasi dan memori bertambah secara linear terhadap panjang teks, bukan kuadratis.
- **KV-Cache Free**: Tidak perlu menyimpan KV-cache yang besar. Hanya butuh satu hidden state berdimensi tetap untuk melakukan generasi, menjadikannya sangat hemat memori pada deployment skala besar.
- **Kecepatan Inferensi Konstan**: Latensi per token baru tetap sama meskipun memproses jutaan token.

#### Kelemahan Mamba
- **In-Context Learning (ICL) Lebih Lemah**: Dibandingkan Transformer dengan ukuran parameter yang sama, Mamba kurang efisien dalam belajar dari contoh-contoh di dalam prompt (few-shot learning).
- **Akurasi Asosiasi Detail Rendah**: Untuk tugas penalaran logika yang sangat ketat (seperti kompilasi kode pemrograman atau matematika multi-langkah), Mamba murni sering kehilangan fokus detail.

### Update 2026: Arsitektur Hybrid & Teknik "Priming"
Melihat trade-off di atas, industri di 2026 tidak lagi mempertentangkan Mamba vs Transformer. Standar industri beralih ke **Hybrid SSM-Transformer** (seperti seri Jamba 1.5, Bamba, dan IBM Granite):
- Model menumpuk block secara selang-seling: misalnya, 4 block SSM diikuti 1 block Attention.
- Block Attention menjaga kemampuan penalaran global dan in-context learning.
- Block SSM mengurangi KV-cache total hingga 80%, memungkinkan model berjalan cepat pada sequence jutaan token.

#### Priming (Metode Inisialisasi Hybrid Terbaru)
Membangun model hybrid dari nol sangat mahal. Pada Mei 2026, teknik **Priming** diperkenalkan untuk mengatasi ini:
- Kita mengambil model Transformer murni yang sudah di-pretrain (misal Llama-3).
- Sebagian block Transformer di-distill dan dikonversi menjadi block SSM.
- Model hybrid baru ini kemudian di-fine-tune singkat.
- Teknik ini memangkas biaya training model hybrid baru hingga **90%** karena mendaur ulang pengetahuan dari model Transformer yang sudah matang.

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

1. **SFT (Supervised Fine-Tuning)**: Model belajar format instruksi-respon. Dataset berisi ribuan contoh "User: pertanyaan → Assistant: jawaban" buatan kurator manusia. Model belajar bahwa prompt harus diikuti dengan format tertentu (bukan sekadar meneruskan teks web acak).

2. **Alignment (RLHF / DPO / GRPO)**: Model belajar menyelaraskan outputnya dengan nilai-nilai manusia (berguna, jujur, aman):
   - **RLHF (Reinforcement Learning from Human Feedback)**: Menggunakan model reward terpisah untuk menilai skor output, lalu melatih model utama dengan algoritma PPO (Proximal Policy Optimization). Sangat rumit karena melibatkan 4 model sekaligus di VRAM (Policy, Reference, Reward, Value).
   - **DPO (Direct Preference Optimization)**: Membuang model reward terpisah. DPO langsung mengoptimasi model menggunakan data preferensi biner (pasangan respon "accepted" vs "rejected") melalui fungsi loss matematis yang elegan.
   - **GRPO (Group Relative Policy Optimization - DeepSeek Breakthrough)**: 
     Algoritma RLHF modern yang merevolusi cara melatih model *reasoning* (seperti DeepSeek-R1):
     - Alih-alih memelihara model **Value** yang besar di memori untuk menghitung estimasi keuntungan (baseline/advantage), GRPO melakukan **sampling kelompok**.
     - Untuk setiap prompt, model memproduksi sekelompok respon (misal $G = 8$ respon).
     - Reward untuk setiap respon dihitung (misalnya melalui test suite untuk coding, atau jawaban biner benar/salah untuk matematika).
     - **Advantage relatif** dihitung secara instan dengan menormalisasi reward di dalam kelompok tersebut (z-score relative to the group).
     - Model di-update berdasarkan keunggulan relatif respon tersebut terhadap rekan-rekannya di kelompok yang sama.
     - **Keuntungan**: Menghemat memori GPU secara ekstrem (karena membuang model Value dan model Reward) dan melatih kemampuan *self-correction* model dengan sangat cepat pada tugas logika terstruktur.

**Hasil akhir**: Model siap digunakan untuk dialog, mengikuti instruksi kompleks, dan bernalar otonom tanpa mengalami kegagalan alignment atau bias respon monoton.

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

### 🔬 Eksperimen Google Colab: Visualisasi MoE Router & Expert Collapse

Copy-paste kode ini ke Google Colab (cukup gunakan CPU) untuk mensimulasikan bagaimana router memilih expert, memvisualisasikan data affinity, dan memahami fenomena "expert collapse":

```python
# ============================================================
# EKSPERIMEN: MoE Router & Expert Selection Visualization
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class MiniMoE(nn.Module):
    """Simplified MoE layer untuk visualisasi."""
    def __init__(self, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)
        
    def forward(self, x):
        batch, seq_len, hidden = x.shape
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return router_logits, top_k_indices, top_k_weights

# Inisialisasi
hidden_dim = 64
num_experts = 8
moe = MiniMoE(hidden_dim, num_experts, top_k=2)

# Generate 200 token random
torch.manual_seed(42)
x = torch.randn(1, 200, hidden_dim)

with torch.no_grad():
    logits, indices, weights = moe(x)

# Hitung beban per expert
expert_counts = torch.zeros(num_experts)
for k in range(2):
    for e in range(num_experts):
        expert_counts[e] += (indices[0, :, k] == e).sum().item()

# Visualisasi
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Load imbalance
ax = axes[0]
colors = plt.cm.Set3(np.linspace(0, 1, num_experts))
ax.bar(range(num_experts), expert_counts.numpy(), color=colors)
ax.axhline(y=200*2/num_experts, color='red', linestyle='--', label='Ideal Balanced')
ax.set_xlabel('Expert ID'); ax.set_ylabel('Beban Token')
ax.set_title('Distribusi Beban Expert')
ax.legend()

# Plot 2: Heatmap Logits Router (50 Token Pertama)
ax = axes[1]
im = ax.imshow(logits[0, :50, :].numpy().T, aspect='auto', cmap='viridis')
ax.set_xlabel('Token Position'); ax.set_ylabel('Expert ID')
ax.set_title('Logits Router (Afinitas Expert)')
plt.colorbar(im, ax=ax)

# Plot 3: Expert Selection Pattern (50 Token Pertama)
ax = axes[2]
selection = indices[0, :50, :].numpy()
ax.scatter(range(50), selection[:, 0], c='blue', s=30, label='Top-1 Expert', alpha=0.7)
ax.scatter(range(50), selection[:, 1], c='orange', s=30, label='Top-2 Expert', alpha=0.7)
ax.set_xlabel('Token Position'); ax.set_ylabel('Expert ID')
ax.set_title('Expert Pilihan Router')
ax.legend()

plt.suptitle('Analisis Mekanisme Perutean MoE (Mixture of Experts)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Imbalance score
imbalance = expert_counts.max().item() / (expert_counts.min().item() + 1e-9)
print(f"Rasio Imbalance Beban: {imbalance:.2f}x")
```

---

## 🧩 Latihan

### Level 1 — Recall
Jelaskan dengan analogi sederhana: apa perbedaan antara Dense Model (Transformer biasa) dan MoE (Sparse Model)? Mengapa total memori yang dibutuhkan MoE tetap besar meskipun komputasinya hemat?

### Level 2 — Aplikasi
Jalankan eksperimen MoE di atas di Google Colab.
- Lihat grafis **Distribusi Beban Expert**. Apakah rata atau ada expert tertentu yang mendapatkan token jauh lebih banyak daripada expert lainnya?
- Jika rasio imbalance-nya di atas $3\times$, ini adalah indikasi awal dari **expert collapse**. Bagaimana DeepSeek mencegah hal ini saat training model aslinya?

### Level 3 — Riset
Cari tahu tentang teknik **DPO** dan **GRPO** untuk alignment LLM. Apa keuntungan utama GRPO dibandingkan RLHF tradisional yang menggunakan PPO dalam hal penggunaan memori GPU selama training?

---

## 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| **MoE** | Arsitektur di mana token diproses secara selektif oleh subset expert, menghemat biaya komputasi per token. |
| **Expert Collapse** | Kondisi abnormal di mana router hanya memilih 1-2 expert saja; dicegah dengan auxiliary loss atau dynamic bias. |
| **MLA (Multi-head Latent Attention)** | Teknik kompresi joint key-value yang memangkas KV-cache hingga 93% tanpa degradasi performa. |
| **Mamba / SSM** | Model dengan kompleksitas linear $\mathcal{O}(n)$ yang sangat efisien untuk sequence panjang, sering digabungkan dengan Transformer (Hybrid). |
| **Priming** | Metode pemotongan biaya training model hybrid dengan mengonversi model Transformer pretrained menjadi hybrid. |
| **GRPO** | Algoritma alignment hemat memori yang membuang model Critic/Value dengan memanfaatkan advantage relative dalam kelompok output. |

> **Takeaway utama**: Mengatasi bottleneck skalabilitas Transformer di 2026 melibatkan tiga pilar: efisiensi parameter (MoE), efisiensi KV-cache (MLA), arsitektur linear-time (Mamba/Hybrid), dan alignment hemat memori (GRPO).

---

**Selanjutnya → Fase 3: Rekayasa Sistem** — Cukup teori model. Saatnya membangun sistem AI yang nyata.
