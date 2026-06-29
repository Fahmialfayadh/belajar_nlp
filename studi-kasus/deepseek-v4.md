# DeepSeek-V4: Merancang Ulang Efisiensi di Era Konteks Satu Juta Token

> **Sumber:** DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348v1, 2026.

---

## Deskripsi Model

DeepSeek-V4 adalah keluarga model bahasa besar (LLM) berbasis arsitektur **Mixture-of-Experts (MoE)** yang dirilis oleh DeepSeek-AI pada April 2026. Keluarga ini terdiri dari dua varian:

- **DeepSeek-V4-Pro** — 1.6 triliun parameter total, 49 miliar parameter aktif per token
- **DeepSeek-V4-Flash** — 284 miliar parameter total, 13 miliar parameter aktif per token

Keduanya mendukung konteks hingga **satu juta token** secara native dan efisien — sebuah pencapaian yang sebelumnya mustahil tanpa trade-off performa yang besar. Model-model ini dilatih pada lebih dari 32–33 triliun token data beragam dan berkualitas tinggi.

DeepSeek-V4 bukan sekadar iterasi tambahan di atas V3. Ia merupakan rancangan ulang arsitektur yang secara eksplisit menargetkan bottleneck paling fundamental dalam LLM modern: **kompleksitas kuadratik attention pada sekuens panjang**.

---

## The Why: Mengapa DeepSeek-V4 Dibangun?

Kemunculan *reasoning models* (seperti DeepSeek-R1 dan o1) menetapkan paradigma baru bernama **test-time scaling** — model diberi lebih banyak "waktu berpikir" saat inferensi untuk menghasilkan jawaban yang lebih baik. Namun paradigma ini membentur dinding yang sangat konkret: semakin panjang proses berpikir, semakin panjang sekuens yang harus diproses, dan attention vanila memiliki kompleksitas **O(n²)** terhadap panjang sekuens. Biaya komputasi meledak secara kuadratik.

Di sisi lain, aplikasi dunia nyata semakin membutuhkan *long-horizon reasoning* — menganalisis dokumen ribuan halaman, menjalankan agen yang harus melacak riwayat percakapan panjang, atau melakukan cross-document analysis dalam skala besar. Solusi yang ada (seperti sparse attention ad-hoc) tidak cukup sistematis dan tidak diselesaikan pada level arsitektur.

DeepSeek-V4 menjawab ini dengan membangun dari nol sebuah arsitektur hybrid yang **secara struktural efisien** untuk konteks panjang, bukan hanya menempelkan patch di atas arsitektur lama. Hasilnya: pada skenario 1 juta token, V4-Pro hanya membutuhkan **27% FLOPs inferensi** dan **10% KV cache** dibandingkan DeepSeek-V3.2.

---

## Key Insights

**1. Bottleneck sejati LLM bukan pada parameter, melainkan pada attention.**
Penambahan parameter via MoE relatif murah karena hanya sebagian kecil expert yang aktif per token. Sebaliknya, attention harus memproses seluruh konteks setiap saat. DeepSeek-V4 memfokuskan inovasinya di sini.

**2. Kompresi KV cache adalah kunci, bukan sekadar sparse attention.**
Pendekatan sebelumnya (seperti sparse attention biasa) mengurangi token yang diakses, tetapi KV cache tetap tumbuh linear. DeepSeek-V4 mengompresi representasi KV itu sendiri — mengurangi *ukuran* cache, bukan hanya *berapa banyak* yang dibaca.

**3. Residual connections bisa menjadi sumber instabilitas pada skala besar.**
Hyper-Connections biasa menunjukkan instabilitas numerik saat di-stack dalam banyak layer. Manifold-Constrained Hyper-Connections (mHC) menyelesaikan ini dengan menjamin spectral norm matrix ≤ 1, membuat propagasi sinyal lebih stabil secara matematis.

**4. Optimizer bukan detail teknis minor.**
Beralih dari Adam/AdamW ke Muon menghasilkan konvergensi lebih cepat dan stabilitas training yang lebih baik, khususnya saat bekerja bersama arsitektur baru yang lebih kompleks.

**5. Post-training yang efektif membutuhkan spesialisasi sebelum unifikasi.**
Alih-alih melatih satu model generalis dari awal, DeepSeek-V4 melatih expert khusus per domain (matematika, coding, agent, instruction following) lalu menggabungkan kemampuan mereka melalui on-policy distillation.

---

## Key Features

- Konteks natif satu juta token pada kedua varian model
- Hybrid attention: gabungan CSA dan HCA per layer
- Manifold-Constrained Hyper-Connections menggantikan residual connections standar
- Muon optimizer menggantikan AdamW
- DeepSeekMoE dengan routing berbasis `Sqrt(Softplus(·))` untuk affinity scoring
- Hash routing pada beberapa layer awal Transformer
- Multi-Token Prediction (MTP) dipertahankan dari V3
- FP4 Quantization-Aware Training untuk expert weights
- On-policy distillation sebagai tahap akhir post-training
- Infrastruktur inferensi dengan KV cache heterogen dan on-disk storage

---

## Teknologi yang Digunakan

### 1. Mixture-of-Experts (DeepSeekMoE)

DeepSeek-V4 mempertahankan kerangka DeepSeekMoE yang membagi Feed-Forward Network (FFN) menjadi banyak *expert* kecil, di mana setiap token hanya melewati subset expert yang dipilih oleh router.

**Perubahan dari V3:** Fungsi aktivasi untuk menghitung affinity score diubah dari:

$$\text{Sigmoid}(\cdot)$$

menjadi:

$$\text{Sqrt}(\text{Softplus}(\cdot))$$

Ini memberikan gradien yang lebih mulus dan distribusi score yang lebih stabil.

Untuk load balancing, digunakan **auxiliary-loss-free strategy** dengan tambahan *sequence-wise balance loss* ringan untuk mencegah imbalance ekstrem dalam satu sekuens.

---

### 2. Manifold-Constrained Hyper-Connections (mHC)

Residual connection standar adalah `x_{l+1} = x_l + F_l(x_l)`. Hyper-Connections (HC) memperluas ini dengan **memperlebar residual stream** dari dimensi `ℝ^d` menjadi `ℝ^{n_hc × d}`, memperkenalkan tiga linear mapping per layer.

**Formula HC dasar:**

$$X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l)$$

Di mana:
- $X_l \in \mathbb{R}^{n_{\text{hc}} \times d}$ — residual state berlapis
- $A_l \in \mathbb{R}^{1 \times n_{\text{hc}}}$ — input mapping
- $B_l \in \mathbb{R}^{n_{\text{hc}} \times n_{\text{hc}}}$ — residual transformation
- $C_l \in \mathbb{R}^{n_{\text{hc}} \times 1}$ — output mapping
- $\mathcal{F}_l$ — layer ke-l (misalnya MoE layer)

**Masalah HC:** Instabilitas numerik saat di-stack banyak layer karena $B_l$ tidak terkonstrain.

**Solusi mHC:** $B_l$ dikonstrain ke manifold **doubly stochastic matrices** (Birkhoff polytope):

$$B_l \in \mathcal{M} \coloneq \{M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^T M = \mathbf{1}_n^T,\ M \geq 0\}$$

Constraint ini menjamin **spectral norm** $\|B_l\|_2 \leq 1$, sehingga residual transformation bersifat *non-expansive* dan propagasi sinyal stabil secara forward maupun backward.

**Parameter dikonstrain dengan:**

$$A_l = \sigma(\tilde{A}_l), \quad C_l = 2\sigma(\tilde{C}_l)$$

Di mana $\sigma(\cdot)$ adalah fungsi Sigmoid untuk menjamin non-negativity dan boundedness.

Untuk $B_l$, digunakan **algoritma Sinkhorn-Knopp** — iterasi normalisasi kolom dan baris yang dijamin konvergen ke doubly stochastic matrix:

$$M^{(0)} = \exp(\tilde{B}_l), \quad M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)}))$$

**Dynamic Parameterization:** Parameter ketiga mapping juga dihasilkan secara dinamis (bergantung input), dikomposisi menjadi komponen dinamis dan statis:

$$\tilde{A}_l = \alpha_l^{\text{pre}} \cdot (\hat{X}_l W_l^{\text{pre}}) + S_l^{\text{pre}}$$

$$\tilde{B}_l = \alpha_l^{\text{res}} \cdot \text{Mat}(\hat{X}_l W_l^{\text{res}}) + S_l^{\text{res}}$$

$$\tilde{C}_l = \alpha_l^{\text{post}} \cdot (\hat{X}_l W_l^{\text{post}})^T + S_l^{\text{post}}$$

Di mana $\hat{X}_l = \text{RMSNorm}(\text{vec}(X_l))$ adalah flattened dan normalized input, $W$ adalah bobot yang dapat dipelajari, $S$ adalah static bias, dan $\alpha$ adalah learnable gating factor yang diinisialisasi ke nilai kecil.

---

### 3. Hybrid Attention: Compressed Sparse Attention (CSA) dan Heavily Compressed Attention (HCA)

Ini adalah inovasi inti DeepSeek-V4. Alih-alih menggunakan satu jenis attention untuk semua layer, V4 menggunakan **dua jenis yang berbeda secara strategis**:

#### 3a. Compressed Sparse Attention (CSA)

CSA mengompresi KV cache **sepanjang dimensi sekuens** (chunk-level compression), lalu melakukan sparse attention. Setiap chunk token diringkas menjadi satu pasangan KV terkompresi.

Komponen CSA:
- **Compressed Key-Value Entries:** KV entries dikompresi dari token-level ke chunk-level
- **Lightning Indexer:** Mekanisme seleksi sparse yang efisien untuk memilih chunk KV mana yang paling relevan per query
- **Shared Key-Value MQA (Multi-Query Attention):** Satu set KV shared di antara banyak query head untuk mengurangi ukuran KV cache
- **Grouped Output Projection:** Output projection dikelompokkan untuk efisiensi komputasi

#### 3b. Heavily Compressed Attention (HCA)

HCA mengompresi KV **lebih agresif** dibandingkan CSA, namun tetap menggunakan **dense attention** (bukan sparse). Ini berarti setiap token masih melihat seluruh konteks, tetapi representasi KV yang dilihat sudah sangat terkompresi.

Kompresi agresif ini memungkinkan HCA menangani konteks sangat panjang dengan FLOPs dan KV cache yang jauh lebih kecil, dengan trade-off pada granularitas informasi yang tersedia.

#### 3c. Detail Teknis Lain dalam Hybrid Attention

**Partial Rotary Positional Embedding (RoPE):** RoPE hanya diterapkan pada sebagian dimensi query/key, sementara sisanya bebas dari position encoding. Ini penting untuk kompatibilitas antara representasi terkompresi dan mekanisme posisi.

**Sliding Window Attention (SWA) Branch:** Ditambahkan sebagai branch tambahan untuk menjamin setiap token tetap memiliki akses ke tetangga lokalnya secara penuh, melengkapi long-range attention yang sparse/compressed.

**Attention Sink:** Token-token awal dalam sekuens ("sink tokens") selalu diakses oleh semua attention layer, karena secara empiris terbukti penting untuk stabilitas perhatian pada konteks sangat panjang.

**Query dan KV Normalization:** Normalisasi diterapkan pada query dan KV entries sebelum attention computation untuk stabilitas numerik.

---

### 4. Muon Optimizer

DeepSeek-V4 beralih dari AdamW ke **Muon** (Momentum + Nesterov orthogonalization), sebuah optimizer yang mengoptimalkan arah update dengan cara yang lebih ekspresif secara geometrik.

Muon mengaplikasikan **Newton-Schulz iteration** untuk mengortogonalisasi gradient update, memastikan langkah update memiliki spektrum singular yang seragam (mendekati matriks ortogonal). Ini menghasilkan konvergensi lebih cepat dan mengurangi sensitivitas terhadap learning rate.

Untuk menghindari **exploding attention logits** — yang bisa terjadi karena Muon tidak memiliki per-parameter scaling seperti Adam — diterapkan mekanisme logit clipping eksplisit pada attention scores.

**Hybrid Newton-Schulz Iterations:** Versi yang digunakan di DeepSeek-V4 mengkombinasikan beberapa langkah iterasi Newton-Schulz dengan polynomial approximation untuk efisiensi komputasi yang lebih baik.

---

### 5. Multi-Token Prediction (MTP)

Diwarisi dari DeepSeek-V3 tanpa modifikasi. Alih-alih memprediksi satu token berikutnya per forward pass, model dilatih untuk memprediksi **beberapa token sekaligus** menggunakan modul MTP tambahan.

Secara training, ini memberikan signal supervisi yang lebih padat. Secara inferensi, ini dapat dimanfaatkan untuk *speculative decoding* — menghasilkan beberapa token sekaligus dengan satu forward pass dan memverifikasinya secara paralel.

---

### 6. Post-Training Pipeline: Specialist Training + On-Policy Distillation

Pipeline post-training terdiri dari dua tahap:

**Tahap 1 — Specialist Training:**
Untuk setiap domain target (matematika, coding, agent, instruction following), dilatih model specialist secara terpisah:
1. SFT pada data domain-spesifik berkualitas tinggi
2. RL menggunakan **Group Relative Policy Optimization (GRPO)** dengan reward model yang disesuaikan per domain

GRPO mengoptimalkan policy dengan membandingkan reward relatif antar output dalam satu grup, tanpa memerlukan critic network terpisah seperti pada PPO:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]$$

Di mana $\hat{A}_i$ adalah advantage yang dihitung berdasarkan perbandingan reward dalam grup yang sama.

**Tahap 2 — On-Policy Distillation:**
Model unified (student) belajar dari ensemble model specialist (teachers) menggunakan **reverse KL loss**:

$$\mathcal{L}_{\text{OPD}} = \mathbb{D}_{\text{KL}}[\pi_{\text{student}} \| \pi_{\text{teacher}}]$$

Berbeda dengan forward KL yang cenderung mode-covering, reverse KL bersifat mode-seeking — mendorong student untuk sangat tajam pada mode yang paling dipilih teacher, menghasilkan output yang lebih presisi dan tegas.

---

### 7. FP4 Quantization-Aware Training (QAT)

Bobot expert dalam MoE menggunakan presisi **FP4** (4-bit floating point) selama post-training. Ini mengurangi ukuran model dan kebutuhan memori secara signifikan. QAT memastikan bahwa model dilatih dengan simulasi error kuantisasi, sehingga performa tidak turun drastis saat deployment.

Pada hardware saat ini, FP4 × FP8 operations memiliki peak FLOPs yang sama dengan FP8 × FP8. Namun pada hardware generasi berikutnya, operasi ini diproyeksikan menjadi 1/3 lebih efisien — menjadikan keputusan ini investment jangka panjang.

---

### 8. Multi-Query Attention (MQA) dan Grouped Query Attention (GQA)

Baik CSA maupun HCA memanfaatkan **Shared Key-Value MQA** — satu set K dan V yang digunakan bersama oleh semua atau beberapa attention head sekaligus, dibandingkan full MHA (Multi-Head Attention) yang memiliki K dan V terpisah per head. Ini mengurangi ukuran KV cache secara langsung proporsional dengan jumlah head.

---

## Ringkasan Arsitektur

```
Input Token
    │
    ▼
[Embedding + RoPE (Partial)]
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Block × N          │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Manifold-Constrained   │   │
│  │  Hyper-Connection (mHC) │   │
│  └────────────┬────────────┘   │
│               │                 │
│  ┌────────────▼────────────┐   │
│  │  Attention Layer:       │   │
│  │  CSA (sebagian layer)   │   │
│  │  HCA (sebagian layer)   │   │
│  │  + SWA branch           │   │
│  │  + Attention Sink       │   │
│  └────────────┬────────────┘   │
│               │                 │
│  ┌────────────▼────────────┐   │
│  │  DeepSeekMoE FFN        │   │
│  │  (Sqrt(Softplus) router)│   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
    │
    ▼
[Multi-Token Prediction Head]
    │
    ▼
Output Tokens
```

---

## Perbandingan Efisiensi vs DeepSeek-V3.2 (Konteks 1M Token)

| Model | FLOPs Inferensi | KV Cache |
|---|---|---|
| DeepSeek-V3.2 | 100% (baseline) | 100% (baseline) |
| DeepSeek-V4-Pro | **27%** | **10%** |
| DeepSeek-V4-Flash | **10%** | **7%** |

---

## Catatan untuk Pembaca Modul

Teknologi-teknologi dalam DeepSeek-V4 saling berinteraksi secara non-trivial. Manifold-constrained mHC tidak bisa dipahami terpisah dari fakta bahwa model ini memiliki puluhan layer yang dalam. CSA dan HCA tidak bisa diapresiasi sepenuhnya tanpa memahami mengapa KV cache menjadi bottleneck pada inferensi konteks panjang. Dan keputusan menggunakan Muon tidak terlepas dari fakta bahwa Adam secara inheren memiliki keterbatasan saat gradient landscape berubah drastis karena arsitektur baru.

Membaca paper aslinya (arXiv:2606.19348) sangat direkomendasikan, khususnya Bagian 2 (Architecture) dan Bagian 3 (General Infrastructures) untuk pemahaman yang lebih dalam tentang implementasi dan trade-off yang terlibat.
