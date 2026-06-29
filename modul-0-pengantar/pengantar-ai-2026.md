# Modul 0 — Lanskap AI di Pertengahan 2026: Peta Sebelum Perjalanan

> **Estimasi waktu**: 30–45 menit (bacaan santai)
> **Prerequisite**: Tidak ada — ini titik awal sebelum semuanya

---

### 🎯 Kenapa Bab Ini Ada?

Kamu akan menghabiskan banyak waktu di modul-modul selanjutnya membedah *Self-Attention*, *MoE*, *RAG*, sampai *orkestrasi agent*. Semua itu adalah mesin internal. Sebelum bongkar mesinnya, kamu perlu tahu dulu: **mobil apa yang sedang kita bicarakan, siapa yang nyetir, dan mau ke mana?**

Bab ini bukan materi teknis — ini konteks. Anggap ini peta perjalanan sebelum kamu masuk ke hutan teori.

---

### 📖 Era Baru: Dari "Makin Besar" ke "Makin Berguna"

Sampai sekitar 2024, cara menaikkan kemampuan model itu gampang: tambah parameter, tambah data, tambah GPU. Scaling law mendikte segalanya. Sekarang? Paradigma itu sudah berubah total.

Di pertengahan 2026, kompetisi utama bukan lagi soal siapa yang punya model paling besar — tapi **siapa yang bisa bikin model paling efisien, paling bisa dipercaya, dan paling bisa "kerja" secara mandiri.**

Tiga pergeseran fundamental yang membentuk lanskap sekarang:

**1. Dari chatbot ke agent.** Model tidak lagi cuma menjawab pertanyaan. Model sekarang **bertindak** — booking meeting, update database, commit code, bahkan mengoperasikan browser dan desktop secara langsung. Google baru saja merilis fitur "Computer Use" pada Gemini 3.5 Flash, memungkinkan model berinteraksi langsung dengan browser dan mobile tanpa routing ke model khusus.

> 📌 **Koneksi modul:** Kamu akan membangun agent sendiri di [Modul 4.1 — AI Agents](../fase-4-agentic/modul-4.1-ai-agents.md), termasuk loop ReAct dan Function Calling.

**2. Dari satu model besar ke "tim" model kecil.** Tren paling dominan di industri adalah arsitektur Mixture-of-Experts (MoE): model punya parameter triliunan, tapi setiap kali "berpikir" cuma menyalakan sebagian kecil. DeepSeek-V4-Pro, misalnya, punya total parameter triliunan tapi parameter aktif per token-nya jauh lebih kecil — dan performanya setara model frontier Barat. Ini bukan kompromi. Ini *desain*.

> 📌 **Koneksi modul:** Mekanisme MoE, router, dan load balancing dibahas mendalam di [Modul 2.2 — Arsitektur Mutakhir](../fase-2-arsitektur/modul-2.2-arsitektur-modern.md). Penerapan nyatanya bisa kamu lihat di studi kasus [DeepSeek-V4](../studi-kasus/deepseek-v4.md).

**3. Dari "training mahal" ke "inferensi yang cerdas."** DeepSeek baru saja merilis **DSpark** (27 Juni 2026) — sebuah framework *speculative decoding* yang meningkatkan kecepatan generasi hingga puluhan persen di bawah traffic nyata. Caranya: model kecil ("drafter") memprediksi beberapa token ke depan, lalu model besar memverifikasi sekaligus dalam satu batch. Plus, ada mekanisme *confidence-scheduled verification* yang secara dinamis menyesuaikan panjang draft — kalau token yang diprediksi kemungkinan besar salah, jangan buang compute untuk verifikasi.

Ini bukan hanya trik engineering. Ini sinyal bahwa **arena kompetisi AI sudah bergeser dari training ke inferensi** — dari "siapa yang punya GPU paling banyak" ke "siapa yang bisa serve model paling murah dan cepat."

---

### 📖 Pemain-Pemain yang Mengubah Permainan

Yang bikin lanskap 2026 menarik bukan cuma model dari lab-lab Barat. **Pendatang dari China sudah menyamai — dan di beberapa area, melampaui — model frontier.**

#### 🟢 DeepSeek: Si Efisiensi yang Bikin Industri Panik

DeepSeek memulai sebagai lab riset kecil. Sekarang? Mereka baru saja mendapat funding ~$7 miliar dan sedang menggandakan ukuran semua divisi. Yang paling menarik:

- **DeepSeek-V4** (April 2026) menggunakan arsitektur hybrid attention yang secara arsitektural efisien untuk konteks panjang — bukan sekadar menempelkan patch di atas Transformer lama. Di skenario konteks jutaan token, V4 cuma butuh sebagian kecil FLOPs dan KV cache dibandingkan pendahulunya.
- **Tim "Harness"** — tim baru khusus untuk membangun *coding agent* yang bersaing langsung dengan Claude Code milik Anthropic.
- **DSpark + DeepSpec** — framework speculative decoding yang di-open-source di bawah lisensi MIT, dan kompatibel bukan cuma untuk model DeepSeek, tapi juga Qwen dan Gemma.

Pesan strategisnya jelas: DeepSeek sedang bertransformasi dari "lab riset" menjadi "perusahaan produk AI." Dan mereka membawa inovasi arsitektural yang tidak bisa diabaikan.

> 📌 **Koneksi modul:** Arsitektur DeepSeek-V4 — termasuk Compressed Sparse Attention, Manifold-Constrained Hyper-Connections, dan Muon optimizer — dibedah lengkap di [studi kasus DeepSeek-V4](../studi-kasus/deepseek-v4.md).

#### 🟣 Qwen (Alibaba): Dari Chat ke Robot

Qwen bukan cuma model bahasa lagi. Di 2026, ekosistem Qwen sudah mencakup:

- **Qwen3.7-Max** — model flagship proprietari yang dirancang khusus untuk *agentic AI*. Bisa menjalankan ribuan tool call berturut-turut tanpa degradasi performa — bayangkan agent yang bisa ngoding multi-hari tanpa kehilangan konteks.
- **Qwen-Robot Suite** (Juni 2026) — tiga model fondasi untuk *embodied AI*:
  - **Qwen-RobotWorld**: "World model" yang mensimulasikan bagaimana lingkungan fisik akan berubah *sebelum* robot bertindak — semacam "sandbox mental" untuk robot.
  - **Qwen-RobotNav**: Model navigasi yang bisa beradaptasi ke lingkungan baru tanpa perubahan arsitektur saat inferensi.
  - **Qwen-RobotManip**: Model manipulasi fisik yang dilatih pada puluhan ribu jam data open-source dan sudah memenangkan kompetisi robotika dunia nyata.
- **Qwen-AgentWorld** — *world model* berbasis bahasa yang bisa mensimulasikan lingkungan agen di tujuh domain (search, terminal, software engineering, GUI).
- **Strategi harga agresif**: diskon hingga 80% untuk model flagship di platform coding mereka, menargetkan developer global.

Pesan strategisnya: Alibaba tidak cuma bikin model — mereka membangun *AI factory* lengkap dari chip, model, platform cloud, sampai robotika.

> 📌 **Koneksi modul:** Arsitektur Qwen3 — termasuk thinking budget, GRPO, strong-to-weak distillation, dan integrasi MCP — dibedah di [studi kasus Qwen3](../studi-kasus/qwen-3.md).

#### 🔴 Anthropic: Gilanya Ada di Logika

Kalau DeepSeek juaranya efisiensi dan Qwen juaranya ekosistem, Anthropic juaranya **reasoning yang konsisten**. Mereka bukan yang paling cepat rilis model, tapi model mereka dikenal paling bisa "dipercaya" untuk tugas-tugas yang butuh penilaian (*judgment*), bukan cuma jawaban.

- **Claude Fable 5** (Juni 2026) — model terbaru kelas "Mythos" dengan **adaptive thinking**: model secara otomatis mengalokasikan kedalaman reasoning berdasarkan kompleksitas task. Bukan lagi "chain-of-thought manual" — model *sendiri* yang memutuskan seberapa dalam perlu berpikir.
- **Claude Opus 4.8** (Mei 2026) — standar emas untuk *agentic reliability*. Dalam workflow multi-service yang kompleks, Opus sering mengungguli kompetitor bukan karena lebih pintar, tapi karena lebih bisa **menangkap kesalahannya sendiri**, bertanya klarifikasi, dan mengeksekusi tugas jangka panjang tanpa intervensi manusia.
- **Claude Code** — bukan autocompletion biasa. Ini *agent* yang membaca seluruh codebase, mengedit file, menjalankan shell command, mengelola Git, menjalankan test — semua dalam loop otonom. Fitur terbaru: *subagents* (agen khusus dengan konteks terisolasi) dan *agent teams* (koordinasi multi-sesi).

Inovasi Anthropic bukan di arsitektur model (mereka cukup konservatif di situ), tapi di **post-training dan kontrol inferensi** — bagaimana model diajarkan *kapan* harus berpikir dalam dan *kapan* harus langsung jawab.

> 📌 **Koneksi modul:** Konsep adaptive thinking dan thinking budget berkaitan erat dengan materi [Modul 2.1 — Transformer](../fase-2-arsitektur/modul-2.1-transformer.md) (memahami *mengapa* reasoning butuh komputasi ekstra) dan [studi kasus Qwen3](../studi-kasus/qwen-3.md) (yang membahas mekanisme thinking budget secara teknis).

#### 🟡 Google: Si Full-Stack Diam-Diam

Google jarang bikin kehebohan di timeline, tapi secara strategis mereka mungkin pemain paling komplet:

- **Gemini 3.5 Flash** — model frontier yang jadi default di Google Search "AI Mode" dan baru mendapat fitur "Computer Use" native (interaksi langsung dengan browser, mobile, desktop).
- **Gemini Spark** — personal AI agent 24/7 yang terintegrasi ke Gmail, Drive, Sheets, Calendar. Bukan prototype — sudah di-roll out ke Gemini Enterprise.
- **Gemini Omni** — model multimodal yang bisa generate dan edit video berkualitas tinggi.
- Strategi full-stack: punya model sendiri, platform sendiri, chip sendiri (TPU), dan ekosistem developer sendiri — kontrol dari ujung ke ujung.

#### 🔵 Pendatang Baru yang Bikin Kaget

Gap antara model proprietari dan open-weight sudah **menyempit drastis**. Beberapa nama yang harus diperhatikan:

- **GLM-5.2 (Zhipu AI)** — model open-weight berlisensi MIT dengan context window jutaan token. Model open-source pertama yang mengalahkan model closed-source teratas di benchmark software engineering dunia nyata (SWE-Bench Pro). Sering berada di puncak leaderboard untuk coding dan agentic tasks.
- **MiniMax M3** — model open-weight frontier dengan native multimodality (teks, gambar, video) dan arsitektur Sparse Attention khusus (MSA) untuk efisiensi konteks panjang.

Pesan besarnya: **era di mana cuma 3-4 lab yang bisa bikin model frontier sudah berakhir.** Sekarang ada belasan lab yang bersaing di kisaran performa yang sangat dekat.

---

### 📖 "Tapi Benchmark Tinggi ≠ Berguna di Dunia Nyata"

Ini bagian yang sering dilewatkan di berita-berita hype:

#### Masalah Compounding Error

Kalau sebuah agent punya reliabilitas 95% per langkah — kedengarannya bagus, kan? Tapi di workflow 20 langkah, tingkat keberhasilannya jatuh ke ~36%. **Satu error kecil bisa cascade jadi kegagalan total.** Kebanyakan sistem agent belum punya mekanisme checkpoint atau recovery.

#### Masalah "Production Gap"

Sebagian besar proyek AI agent gagal di produksi — bukan karena modelnya bodoh, tapi karena masalah yang lebih "membosankan":

- **Tool chain yang rapuh** — API berubah schema, provider outage, integrasi ke sistem legacy yang ribet
- **Observability yang tidak ada** — server bisa tetap "healthy" padahal agent-nya menghasilkan data sampah
- **Arsitektur yang salah arah** — tim berasumsi reasoning model akan menutupi gap arsitektur. Padahal deployment yang sukses justru fokus pada desain *components-first*: batas keputusan, jalur eskalasi, dan gate human-in-the-loop yang di-encode eksplisit ke sistem

Bahkan perusahaan besar seperti Uber dilaporkan mengalami *budget exhaustion* karena konsumsi token yang intensif dari sub-agent otonom mereka.

#### Masalah Trust Gap

Studi terbaru dari Forbes menunjukkan bahwa bahkan model teratas sering hanya mendapat nilai "C" saat dievaluasi oleh profesional berlisensi untuk reliabilitas dunia nyata. **Mendapatkan "jawaban benar" itu berbeda dari memberikan "penilaian profesional."**

> 📌 **Koneksi modul:** Ini persis kenapa [Modul 3.4 — Evaluasi RAG](../fase-3-sistem/modul-3.4-evaluasi-rag.md) ada — karena tanpa evaluasi yang benar, kamu tidak tahu apakah sistem AI-mu benar-benar kerja atau cuma *kelihatan* kerja. Dan [Modul 4.1 — AI Agents](../fase-4-agentic/modul-4.1-ai-agents.md) membahas kenapa human-in-the-loop itu bukan opsional.

---

### 📖 Regulasi dan Geopolitik: Arena Baru yang Tidak Bisa Diabaikan

AI di 2026 bukan cuma soal teknologi. **Politik sudah masuk arena** dengan cara yang langsung berdampak ke akses model:

- **Export control AS**: Di Juni 2026, model-model teratas Anthropic (kelas Fable 5 dan Mythos 5) mengalami suspensi akses sementara untuk pengguna asing berdasarkan arahan kontrol ekspor pemerintah AS. Model standar (Opus 4.8) tetap tersedia, tapi ini sinyal jelas bahwa akses ke AI frontier bisa dibatasi berdasarkan geopolitik.
- **EU AI Act**: Uni Eropa menuntut transparansi dan akuntabilitas untuk sistem AI berisiko tinggi. Era "asal jalanin AI" sudah berakhir.
- **EU Cloud and AI Development Act (CADA)**: Regulasi baru yang sedang diajukan untuk mengatur pengembangan AI di level infrastruktur cloud.

Implikasi praktisnya: **model open-weight dari lab China (DeepSeek, GLM, Qwen) menjadi semakin strategis** sebagai alternatif bagi organisasi yang khawatir soal akses dan kedaulatan data.

---

### 📖 Lima Level AI: Kita Ada di Mana?

Sebagai kerangka berpikir, industri sering pakai "5 Level AI" untuk mengukur kemajuan:

| Level | Nama | Apa Artinya? | Status 2026 |
|---|---|---|---|
| 1 | **Chatbots** | Ngobrol natural, jawab pertanyaan | ✅ Basi — sudah jadi fitur standar |
| 2 | **Reasoners** | Bernalar, mecahin soal kompleks | 🔥 Arena utama — adaptive thinking, CoT |
| 3 | **Agents** | Bertindak mandiri, multi-langkah | 🚀 Tren terbesar — tapi banyak yang gagal di produksi |
| 4 | **Innovators** | Menemukan solusi/teori baru | 🔬 Prototipe awal |
| 5 | **Organizational** | Menjalankan seluruh fungsi organisasi | 📝 Masih teori |

Kita sekarang ada di **transisi Level 2→3**: model sudah bisa bernalar dengan baik, dan sekarang sedang belajar *bertindak* secara andal. Gap terbesar bukan di kecerdasan model, tapi di *reliabilitas sistem* yang dibangun di atasnya.

---

### 📖 Peta Koneksi: Bab Ini ↔ Modul-Modul Selanjutnya

```
┌─────────────────────────────────────────────────────────────────┐
│  MODUL 0: Lanskap AI 2026 (kamu di sini)                       │
│  MoE, Adaptive Thinking, Speculative Decoding, Agents...       │
└────────────┬──────────────┬──────────────┬──────────────────────┘
             │              │              │
     ┌───────▼───────┐  ┌──▼──────────┐  ┌▼──────────────┐
     │ FASE 1: Fondasi│  │ FASE 2:     │  │ FASE 3:       │
     │ Dot product,   │  │ Transformer,│  │ RAG, VecDB,   │
     │ Tokenization,  │  │ MoE, SSM,   │  │ LoRA,         │
     │ Embeddings     │  │ Base vs     │  │ Evaluasi      │
     │                │  │ Instruct    │  │               │
     └───────┬────────┘  └──┬──────────┘  └┬──────────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                    ┌───────▼───────┐
                    │ FASE 4:       │
                    │ AI Agents,    │
                    │ Function      │
                    │ Calling,      │
                    │ Multi-Agent   │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ STUDI KASUS:  │
                    │ DeepSeek-V4,  │
                    │ Qwen3         │
                    └───────────────┘
```

Semua hal di atas — efisiensi arsitektur, reasoning adaptif, agent yang bertindak mandiri, speculative decoding — tidak muncul dari langit. Semuanya dibangun di atas fondasi teknis yang justru akan kita bedah satu per satu:

- **Fase 1**: Gimana teks jadi angka, dan gimana angka itu diukur kemiripannya → dasar dari *semua* operasi AI
- **Fase 2**: Gimana *Self-Attention* kerja, gimana MoE bikin model efisien → mesin di balik semua model frontier
- **Fase 3**: Gimana RAG mencegah halusinasi, dan gimana mengukur kualitasnya → sistem yang membuat model *berguna*
- **Fase 4**: Gimana agent diorkestrasi untuk tugas multi-langkah → frontier terbesar saat ini

Anggap bab ini semacam "trailer" — biar pas masuk teori, kamu sudah punya gambaran kenapa hal-hal itu penting di dunia nyata.

---

*Siap? → Mulai dari [Fase 1: Fondasi Matematika & Embeddings](../fase-1-fondasi/modul-1.1-matematika-terapan.md)*
