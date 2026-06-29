# Qwen3: Thinking dan Non-Thinking

> **Sumber Utama:**
> - Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, May 2025.
> - Qwen Blog: *Qwen3.7 — The Agent Frontier.* qwen.ai/blog, 2026.

---

## Deskripsi Model

Qwen3 adalah keluarga model bahasa besar (LLM) open-weight yang dikembangkan oleh Alibaba Cloud's Qwen Team dan dirilis pada April 2025. Ia merupakan generasi ketiga dari lini model Qwen, mencakup delapan varian model yang merentang dari **0.6 miliar hingga 235 miliar parameter** — baik dalam arsitektur dense maupun Mixture-of-Experts (MoE):

**Model Dense:**
| Model | Parameter | Context |
|---|---|---|
| Qwen3-0.6B | 0.6B | 32K |
| Qwen3-1.7B | 1.7B | 32K |
| Qwen3-4B | 4B | 128K |
| Qwen3-8B | 8B | 128K |
| Qwen3-14B | 14B | 128K |
| Qwen3-32B | 32B | 128K |

**Model MoE:**
| Model | Total Params | Aktif per Token | Context |
|---|---|---|---|
| Qwen3-30B-A3B | 30B | 3B | 128K |
| Qwen3-235B-A22B | 235B | 22B | 128K |

Qwen3 dilatih pada **36 triliun token** yang mencakup **119 bahasa dan dialek** — naik signifikan dari 29 bahasa pada Qwen2.5. Seluruh model dirilis di bawah lisensi **Apache 2.0**, menjadikannya salah satu model open-weight paling permisif di kelasnya.

Penerusnya, **Qwen3.7**, merupakan model proprietari flagship yang dirancang khusus untuk era *agentic AI*, dengan context window **1 juta token** dan kemampuan eksekusi agen lintas framework yang lebih matang.

---

## The Why: Mengapa Qwen3 Dibangun?

Ekosistem LLM pada 2025 terpolarisasi ke dua kutub yang terpisah: model chat generalis (seperti GPT-4o) yang responsif tapi terbatas dalam reasoning kompleks, dan model reasoning khusus (seperti QwQ-32B, o1) yang powerful namun lambat dan tidak efisien untuk task sederhana. Pengguna harus memilih dan berpindah antar model tergantung task yang dihadapi — sebuah friction yang nyata dalam production pipeline.

Di sisi lain, test-time scaling — paradigma di mana performa model ditingkatkan dengan memberi lebih banyak "waktu berpikir" saat inferensi — semakin terbukti efektif. Namun kontrol atas seberapa dalam model harus berpikir untuk setiap query masih kasar: on atau off, tidak ada yang di antaranya.

Qwen3 menjawab kedua masalah ini sekaligus dengan **menyatukan thinking mode dan non-thinking mode dalam satu model tunggal**, dan memberikan kontrol granular melalui mekanisme **thinking budget**. Selain itu, Qwen3 secara eksplisit dibangun sebagai fondasi ekosistem agen — dengan integrasi native untuk tool calling, function calling, dan **Model Context Protocol (MCP)**.

---

## Key Insights

**1. Satu model bisa melayani dua paradigma inferensi sekaligus.**
Tidak harus memilih antara model reasoning dan model chat. Qwen3 membuktikan bahwa dengan pipeline post-training yang tepat, satu model bisa berpindah antara mode berpikir panjang dan respons cepat berdasarkan instruksi user, tanpa degradasi performa yang signifikan di keduanya.

**2. Thinking budget mengubah inferensi dari biner menjadi kontinum.**
Bukan lagi "apakah model harus berpikir?", melainkan "seberapa banyak token berpikir yang dialokasikan untuk query ini?". Ini adalah pergeseran fundamental: performa model kini menjadi fungsi dari anggaran komputasi yang diberikan, bukan hanya fungsi dari ukuran model.

**3. Distilasi dari model besar lebih efisien daripada RL pada model kecil.**
Untuk model-model kecil (0.6B–14B), Qwen3 menggunakan strong-to-weak distillation dari model flagship, bukan melatih RL secara langsung. Hasilnya lebih baik dan lebih hemat secara komputasi — pelajaran penting tentang transfer kapabilitas antar skala.

**4. Data sintetis berkualitas tinggi dari model sendiri adalah multiplier yang efektif.**
Qwen3 menggunakan Qwen2.5-VL untuk ekstraksi teks dari PDF skala besar, serta Qwen2.5-Math dan Qwen2.5-Coder untuk menghasilkan data sintetis domain-spesifik. Ini memungkinkan dataset pre-training mencapai 36T token tanpa sepenuhnya bergantung pada data web mentah.

**5. MCP adalah infrastruktur, bukan fitur.**
Qwen3 tidak sekadar mendukung tool calling — seluruh ekosistemnya (Qwen-Agent, template chat, parsing framework) dirancang dengan MCP sebagai primitive. Ini berbeda secara filosofis dari model yang menambahkan tool calling sebagai afterthought.

**6. Efisiensi MoE bisa melampaui dense pada skala yang sama.**
Qwen3-30B-A3B dengan hanya 3B parameter aktif mencapai performa yang sebanding dengan Qwen3-14B dense dan bahkan Qwen2.5-32B — dengan biaya inferensi yang jauh lebih rendah. Ini memvalidasi bahwa MoE fine-grained adalah jalur yang layak untuk skalabilitas.

---

## Key Features

- Unified thinking/non-thinking mode dalam satu model, dikontrol via chat template
- Thinking budget mechanism untuk alokasi komputasi adaptif saat inferensi
- Pre-training 36T token mencakup 119 bahasa dan dialek (Qwen2.5: 29 bahasa)
- 3-stage pre-training: General → Reasoning → Long Context
- 4-stage post-training untuk flagship: Long-CoT Cold Start → Reasoning RL → Thinking Mode Fusion → General RL
- Strong-to-weak distillation untuk model-model kecil
- Native MCP (Model Context Protocol) support via Qwen-Agent framework
- QK-Norm untuk stabilitas training (menggantikan QKV-bias dari Qwen2)
- Grouped Query Attention (GQA) untuk efisiensi KV cache
- YARN + Dual Chunk Attention untuk 4× peningkatan kapasitas context saat inferensi
- Global-batch load balancing loss untuk spesialisasi expert di model MoE
- Seluruh model open-weight di bawah lisensi Apache 2.0

---

## Teknologi yang Digunakan

### 1. Arsitektur Dasar: Dense dan MoE

Arsitektur Qwen3 dense mewarisi fondasi dari Qwen2.5, dengan komponen-komponen standar berikut yang dipertahankan:

**Grouped Query Attention (GQA):** Alih-alih setiap attention head memiliki Key dan Value tersendiri (Multi-Head Attention penuh), GQA mengelompokkan beberapa query head untuk berbagi satu pasang KV. Jika terdapat $H_q$ query heads dan $H_{kv}$ KV heads dengan $H_{kv} < H_q$, maka setiap KV head melayani $H_q / H_{kv}$ query heads. Ini mengurangi ukuran KV cache sebesar faktor $H_q / H_{kv}$ dibandingkan MHA penuh.

**SwiGLU sebagai aktivasi FFN:**

$$\text{SwiGLU}(x, W, V, b, c) = \text{Swish}_1(xW + b) \otimes (xV + c)$$

Di mana $\text{Swish}_1(x) = x \cdot \sigma(x)$ dan $\otimes$ adalah perkalian element-wise. SwiGLU secara empiris menghasilkan performa lebih baik dibandingkan ReLU atau GELU pada sebagian besar task language modeling.

**Rotary Positional Embedding (RoPE):** Posisi dikodekan langsung ke dalam dot product query-key, bukan ditambahkan sebagai vektor terpisah ke embedding. Untuk pasangan posisi $m$ dan $n$, dot product dimodulasi oleh fungsi rotasi:

$$q_m^T k_n = \text{Re}\left[\sum_{j=0}^{d/2-1} q_{m,j} k_{n,j}^* e^{i(m-n)\theta_j}\right]$$

Di mana $\theta_j = 10000^{-2j/d}$ adalah frekuensi rotasi per dimensi. Pada long context stage, base frequency dinaikkan dari 10,000 ke 1,000,000 menggunakan teknik **ABF (Adjusted Base Frequency)** untuk mendukung konteks yang jauh lebih panjang.

**RMSNorm dengan pre-normalization:**

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot g, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

Pre-normalization berarti normalisasi diterapkan pada input setiap sub-layer (sebelum attention dan FFN), bukan pada outputnya. Ini memberikan training yang lebih stabil dibandingkan post-normalization.

**Perubahan dari Qwen2 → Qwen3:**
- QKV-bias dihapus
- **QK-Norm** diperkenalkan: normalisasi diterapkan khusus pada query dan key sebelum attention computation. Ini mencegah attention logits meledak saat training skala besar, memberikan stabilitas tambahan.

**Arsitektur MoE Qwen3:** Model MoE Qwen3 menggunakan arsitektur yang sama dengan dense, dengan FFN digantikan oleh modul expert. Spesifikasinya: 128 total expert, 8 expert aktif per token. Tidak seperti Qwen2.5-MoE, Qwen3-MoE **tidak menggunakan shared experts** — semua expert bersifat routed. Load balancing dikelola melalui **global-batch load balancing loss** yang mendorong spesialisasi expert, bukan hanya meratakan beban.

---

### 2. Pre-Training: Tiga Tahap yang Progressif

Pre-training Qwen3 dibagi menjadi tiga tahap yang didesain secara progresif:

**Tahap 1 — General Stage (S1):** Seluruh model dilatih pada lebih dari **30 triliun token** dengan sequence length 4,096 token, mencakup 119 bahasa. Tujuannya adalah membangun fondasi pengetahuan dunia yang kuat dan kemampuan bahasa umum.

**Tahap 2 — Reasoning Stage (S2):** Proporsi data STEM, coding, reasoning, dan data sintetis ditingkatkan secara signifikan. Model dilatih tambahan pada sekitar **5 triliun token berkualitas tinggi** dengan learning rate decay yang dipercepat. Ini adalah tahap di mana kapabilitas reasoning pre-training dibentuk secara eksplisit.

**Tahap 3 — Long Context Stage:** Data long-context khusus digunakan untuk memperpanjang kemampuan konteks. Komposisi corpus: 75% teks antara 16,384–32,768 token, dan 25% teks antara 4,096–16,384 token. Teknik yang digunakan:

- **ABF (Adjusted Base Frequency):** Menaikkan base frequency RoPE dari 10,000 ke 1,000,000 agar model dapat menginterpolasi ke panjang sekuens yang lebih besar dari yang dilihat selama training.
- **YARN (Yet Another RoPE extensioN):** Teknik ekstrapolasi yang menyesuaikan distribusi attention untuk sekuens yang lebih panjang dari training length.
- **Dual Chunk Attention (DCA):** Membagi sekuens panjang menjadi chunk-chunk yang saling overlap, di mana setiap chunk melakukan attention secara penuh ke dalam chunk-nya sendiri dan secara global ke posisi-posisi tertentu di luar chunk. Ini memungkinkan 4× peningkatan kapasitas context saat inferensi tanpa modifikasi bobot model.

**Pembuatan Data Sintetis:** Qwen2.5-VL digunakan untuk mengekstrak teks dari dokumen PDF dalam skala besar. Teks yang diekstrak kemudian diperhalus menggunakan Qwen2.5. Selain itu, Qwen2.5-Math dan Qwen2.5-Coder menghasilkan data sintetis domain-spesifik dalam format textbook, QA, instruksi, dan code snippet.

**Instance-level Data Mixing:** Alih-alih mengoptimalkan campuran data di level sumber atau domain seperti pendekatan sebelumnya (DoReMi, DOGE, RegMix), Qwen3 mengoptimalkan campuran di **level instance** melalui eksperimen ablasi ekstensif pada proxy model kecil dengan label data granular.

---

### 3. Post-Training Pipeline: Empat Tahap untuk Flagship

Post-training Qwen3 memiliki dua objektif utama: **Thinking Control** (integrasi mode berpikir) dan **Strong-to-Weak Distillation** (transfer kapabilitas ke model kecil). Untuk model flagship, dilakukan empat tahap berurutan:

#### Tahap 1 — Long-CoT Cold Start

Ini adalah SFT (Supervised Fine-Tuning) pada dataset yang dikurasi secara ketat, terdiri dari problem matematika, coding, logical reasoning, dan STEM dengan jawaban terverifikasi. Fokusnya adalah problem-problem yang secara inheren membutuhkan multi-step Chain-of-Thought (CoT). Tujuannya: menanamkan kemampuan reasoning dasar sebelum masuk ke tahap RL, agar eksplorasi RL tidak dimulai dari nol.

#### Tahap 2 — Reasoning RL dengan GRPO

Reinforcement Learning diterapkan menggunakan **Group Relative Policy Optimization (GRPO)**. Dataset terdiri dari 3,995 pasang query-verifier yang dipilih karena: (a) tidak digunakan pada cold-start phase, (b) dapat dipelajari oleh model cold-start, (c) secekah mungkin, dan (d) mencakup berbagai sub-domain.

GRPO mengoptimalkan policy dengan membandingkan reward relatif dalam satu grup output untuk query yang sama, tanpa memerlukan critic network terpisah:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right]$$

Di mana advantage $\hat{A}_i$ dihitung berdasarkan reward relatif dalam grup:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

Pengamatan penting dari tim Qwen3: menggunakan **batch size besar**, **jumlah rollout per query yang banyak**, dan **off-policy training** untuk efisiensi sampling secara signifikan memperbaiki training. Selain itu, **entropi model dikontrol agar meningkat stabil** — ini mencegah policy collapse (model terlalu cepat konvergen ke jawaban lokal) tanpa mendorong eksplorasi yang berlebihan.

Hasilnya konkret: skor AIME'24 dari model Qwen3-235B-A22B meningkat dari 70.1 ke 85.1 dalam 170 langkah RL, tanpa intervensi manual pada hyperparameter.

#### Tahap 3 — Thinking Mode Fusion

Setelah tahap 1–2, model sudah kuat dalam thinking mode. Tahap ini mengintegrasikan **non-thinking mode** dengan cara SFT pada dataset gabungan: data dengan CoT panjang (thinking) dan data instruksi langsung tanpa CoT (non-thinking). Template chat digunakan untuk memberi sinyal mode ke model:

```
/think  →  aktifkan thinking mode (default)
/no_think  →  aktifkan non-thinking mode
```

Secara implementasi, mode dikontrol melalui parameter `enable_thinking` pada Hugging Face tokenizer. Dalam thinking mode, model menghasilkan output yang dimulai dengan blok `<think>...</think>` berisi chain-of-thought internal, sebelum memberikan jawaban akhir.

#### Tahap 4 — General RL

RL tahap akhir diterapkan pada lebih dari 20 task umum, termasuk instruction following, format adherence, dan agentic behaviors. Reward signals di tahap ini bersifat campuran: rule-based rewards (untuk task yang punya verifikasi jelas) dan model-based scoring (untuk task yang lebih subjektif seperti kualitas teks).

---

### 4. Thinking Budget Mechanism

Thinking budget adalah mekanisme yang memungkinkan pengguna mengalokasikan batas token untuk proses thinking model secara eksplisit. Dengan budget tertentu $B$, model dibatasi untuk menggunakan paling banyak $B$ token dalam blok `<think>` sebelum harus menghasilkan jawaban.

Dampaknya terukur secara empiris: semakin besar thinking budget yang diberikan, semakin baik performa model pada task reasoning yang kompleks. Ini menjadikan **performa model sebagai fungsi monoton dari anggaran komputasi inferensi**, bukan konstanta yang ditentukan semata oleh bobot model.

Secara praktis, ini berarti trade-off latensi-performa dapat dikontrol secara eksplisit di production: task sederhana bisa diperlakukan dengan budget kecil (atau tanpa thinking), sementara task kompleks mendapat budget besar — semuanya dari model yang sama.

---

### 5. Strong-to-Weak Distillation

Untuk model-model kecil (0.6B, 1.7B, 4B, dst.), Qwen3 tidak menjalankan pipeline RL empat tahap secara penuh. Sebagai gantinya, digunakan **strong-to-weak distillation**: model flagship yang telah dilatih lengkap (teacher) mentransfer kapabilitasnya ke model kecil (student).

Distilasi mencakup dua pendekatan:
- **Off-policy distillation:** Student belajar dari dataset output yang dipre-komputasi dari teacher.
- **On-policy distillation:** Teacher menghasilkan output secara real-time selama training student, memungkinkan teacher merespons terhadap state distribusi student saat itu.

Hasil empiris menunjukkan bahwa distilasi dari teacher yang lebih maju **secara konsisten mengungguli RL langsung** pada model kecil, baik dalam performa akhir maupun efisiensi komputasi training.

---

### 6. Model Context Protocol (MCP) dan Ekosistem Agentic

MCP adalah protokol standar yang memungkinkan LLM berinteraksi dengan tools, API, dan database eksternal melalui antarmuka yang seragam. Alih-alih setiap integrasi tool memerlukan parsing custom, MCP mendefinisikan format komunikasi yang konsisten antara model dan tool server.

**Qwen-Agent** adalah framework yang mengenkapsulasi seluruh ekosistem ini. Komponen utamanya:
- `BaseChatModel` — abstraksi LLM dengan function calling built-in
- `BaseTool` — abstraksi tool individual
- `Agent` — komponen high-level yang mengorkestrasi LLM dan tools

Konfigurasi MCP dilakukan melalui JSON config yang mendefinisikan MCP server:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "test.db"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

Model kemudian dapat melakukan function call, menerima hasil, dan melanjutkan reasoning — semuanya dalam satu context window, memanfaatkan thinking mode untuk planning multi-step.

Dalam **Qwen3.7**, kemampuan ini diperkuat lebih jauh: model dilatih dengan **cross-framework reinforcement learning** untuk menghindari shortcut overfitting ke benchmark atau framework tertentu. Hasilnya adalah model yang dapat beroperasi koheren lintas scaffold agen yang berbeda — sebuah properties penting untuk deployment production.

---

### 7. Byte-level Byte-Pair Encoding (BBPE) Tokenizer

Qwen3 menggunakan tokenizer yang sama dengan lini Qwen sebelumnya, mengimplementasikan **Byte-level BPE (BBPE)** dengan vocabulary size 151,669.

BBPE beroperasi pada level byte mentah sebelum karakter, menjamin bahwa setiap string Unicode yang mungkin dapat diencode tanpa simbol unknown. Proses BPE standard:

1. Mulai dari representasi byte-level semua token
2. Hitung frekuensi semua pasangan byte/token yang berdekatan
3. Merge pasangan paling sering menjadi satu token baru
4. Ulangi hingga vocabulary mencapai ukuran target

Dengan vocabulary 151,669, tokenizer ini mampu merepresentasikan teks multibahasa secara efisien sambil tetap menjaga coverage yang baik untuk bahasa-bahasa yang kurang terwakili.

---

## Arsitektur Post-Training Pipeline

```
Pre-trained Base Model
       │
       ▼
┌─────────────────────────────────────────┐
│  Stage 1: Long-CoT Cold Start (SFT)     │
│  ■ Data: Math, Code, STEM, Logic        │
│  ■ Fokus: Foundational CoT reasoning    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 2: Reasoning RL (GRPO)           │
│  ■ 3,995 query-verifier pairs           │
│  ■ Large batch + high rollout           │
│  ■ Entropy control untuk stabilitas     │
│  ■ Off-policy training                  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 3: Thinking Mode Fusion (SFT)    │
│  ■ Mix: CoT data + instruksi langsung   │
│  ■ Chat template: /think / /no_think    │
│  ■ Thinking budget control              │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 4: General RL                    │
│  ■ 20+ general tasks                   │
│  ■ Rule-based + model-based rewards     │
│  ■ Instruction, format, agentic tasks  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
         Unified Qwen3 Model
     (Thinking + Non-Thinking)

              ──────────
         (untuk model kecil)
              ──────────

Pre-trained Small Model
       │
       ▼
┌──────────────────────────┐
│ Strong-to-Weak           │
│ Distillation             │
│ ■ Off-policy transfer    │
│ ■ On-policy transfer     │
│ dari Flagship Teacher    │
└──────────────────────────┘
```

---

## Perbandingan Efisiensi: Qwen3 MoE vs Dense Sebelumnya

| Model | Total Params | Aktif per Token | Setara Performa Dengan |
|---|---|---|---|
| Qwen2.5-72B Dense | 72B | 72B | — (baseline) |
| Qwen3-32B Dense | 32B | 32B | Qwen2.5-72B (10 dari 15 benchmark) |
| Qwen3-30B-A3B MoE | 30B | 3B | Qwen3-14B, Qwen2.5-32B |
| Qwen3-235B-A22B MoE | 235B | 22B | Melampaui DeepSeek-V3 di 14/15 benchmark |

---

## Catatan untuk Pembaca Modul

Desain Qwen3 merefleksikan sebuah thesis yang kian kuat di komunitas LLM: arsitektur model yang relatif konservatif (Transformer standar dengan GQA, SwiGLU, RoPE) bisa menghasilkan kemampuan luar biasa jika dikombinasikan dengan **data pipeline yang cermat**, **pipeline post-training yang terstruktur**, dan **mekanisme kontrol inferensi yang granular**.

Inovasi utama Qwen3 bukan pada layer attention baru atau mekanisme positional encoding yang eksentrik — melainkan pada **bagaimana model diajarkan** (4-stage post-training yang terstruktur), **apa yang diajarkan** (36T token dengan mixing berbasis instansi), dan **bagaimana performa dikontrol saat deployment** (thinking budget).

Untuk memahami lebih dalam tentang GRPO dan dinamika entropy control dalam RL training, membaca paper Qwen3 Technical Report (arXiv:2505.09388) secara langsung — khususnya Section 4 Post-training — sangat dianjurkan.
