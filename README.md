# Modul Belajar NLP Modern (2026)

> **Tentang Modul**: Modul ini disusun berdasarkan perkembangan teknologi AI & NLP terbaru hingga Juni 2026. Kurikulum dirancang secara khusus untuk mengajarkan **fundamental NLP** yang kokoh, namun tetap **difilter secara ketat** agar materi yang dipelajari tetap relevan dengan kebutuhan industri modern saat ini (seperti era LLM multi-modal, agentic AI, dan optimasi arsitektur efisien), tanpa membuang waktu pada metode-metode klasik yang sudah sepenuhnya ditinggalkan di industri.
>
> **Level**: Menengah (Python & matematika dasar (terutama aljabar linier dan kalkulus sederhana) sudah dikuasai).
>
> **Filosofi**: Kode hanyalah alat — pemahaman teori dan intuisi di balik arsitektur adalah kunci utama menguasai NLP.

---

## Road Map

```
MODUL 0: Pengantar
  └── Lanskap AI 2026 (konteks sebelum masuk teori)
       ↓
FASE 1: Fondasi
  └── Math Terapan (Dot Product, Cosine, Ruang Vektor)
  └── Tokenization Modern (BPE, SentencePiece)
  └── Embeddings (Static vs Contextual)
  └── Ekosistem: PyTorch + HuggingFace
       ↓
FASE 2: Arsitektur
  └── Transformer & Self-Attention (teori mendalam)
  └── MoE, Efficient Attention, SSM/Mamba
  └── Base Model vs Instruct Model
       ↓
FASE 3: Sistem
  └── Advanced RAG (Hybrid Search, Chunking, Reranking)
  └── Vector Database (HNSW, Qdrant/Milvus)
  └── LoRA & QLoRA (Parameter-Efficient Fine-Tuning)
  └── Evaluasi: RAGAS & LLM-as-a-Judge
       ↓
FASE 4: Agentic AI
  └── AI Agents (Anatomi, ReAct, Function Calling)
  └── MCP & A2A (Infrastruktur Standar untuk Agent)
  └── Agent di Produksi (Orkestrasi, Evaluasi, Keamanan)
       ↓
STUDI KASUS: Bedah Model Terkini
  └── DeepSeek-V4 (Hybrid Attention, MoE, mHC)
  └── Qwen3 (Thinking Budget, GRPO, Distillation)
```

Setiap fase **membangun di atas fase sebelumnya**. Jangan loncat.

---

## 📂 Struktur Modul

### [🗺️ Modul 0: Pengantar — Lanskap AI 2026](modul-0-pengantar/)
- [Pengantar — Peta Sebelum Perjalanan](modul-0-pengantar/pengantar-ai-2026.md)

### [📦 Fase 1: Fondasi Matematika & Embeddings](fase-1-fondasi/)
- [Modul 1.1 — Matematika Terapan](fase-1-fondasi/modul-1.1-matematika-terapan.md)
- [Modul 1.2 — Tokenization Modern](fase-1-fondasi/modul-1.2-tokenization.md)
- [Modul 1.3 — Embeddings: Static vs Contextual](fase-1-fondasi/modul-1.3-embeddings.md)

### [📦 Fase 2: Arsitektur NLP Modern](fase-2-arsitektur/)
- [Modul 2.1 — Self-Attention & Transformer](fase-2-arsitektur/modul-2.1-transformer.md)
- [Modul 2.2 — Arsitektur Mutakhir: MoE, Efficient Attention & Mamba](fase-2-arsitektur/modul-2.2-arsitektur-modern.md)

### [📦 Fase 3: Rekayasa Sistem NLP](fase-3-sistem/)
- [Modul 3.1 — Advanced RAG](fase-3-sistem/modul-3.1-advanced-rag.md)
- [Modul 3.2 — Vector Database & HNSW](fase-3-sistem/modul-3.2-vector-database.md)
- [Modul 3.3 — LoRA & QLoRA](fase-3-sistem/modul-3.3-lora-qlora.md)
- [Modul 3.4 — Evaluasi RAG](fase-3-sistem/modul-3.4-evaluasi-rag.md)

### [📦 Fase 4: Agentic AI & Orkestrasi](fase-4-agentic/)
- [Modul 4.1 — AI Agents: Dari Chatbot ke Sistem Otonom](fase-4-agentic/modul-4.1-ai-agents.md)
- [Modul 4.2 — MCP & A2A: Infrastruktur Standar untuk Agent](fase-4-agentic/modul-4.2-mcp-a2a.md)
- [Modul 4.3 — Agent di Produksi: Orkestrasi, Evaluasi & Keamanan](fase-4-agentic/modul-4.3-agent-produksi.md)

### [📑 Studi Kasus: Bedah Model Terkini](studi-kasus/)
- [DeepSeek-V4 — Efisiensi di Era Konteks Sejuta Token](studi-kasus/deepseek-v4.md)
- [Qwen3 — Thinking dan Non-Thinking dalam Satu Model](studi-kasus/qwen-3.md)

---

## 🏁 Roadmap Selanjutnya

### Apa yang Sudah Kamu Pelajari

```
✅ MODUL 0: Pengantar
   └── Lanskap AI 2026, 5 Level AI, tren industri

✅ FASE 1: Fondasi
   ├── Math: Dot Product, Cosine Similarity, Ruang Berdimensi Tinggi
   ├── Tokenization: BPE, SentencePiece, implikasi token ≠ kata
   └── Embeddings: Static vs Contextual, Sentence Embeddings

✅ FASE 2: Arsitektur  
   ├── Transformer: Self-Attention, QKV, Multi-Head, Positional Encoding
   ├── Modern: MoE, FlashAttention, Mamba/SSM, MLA
   └── Base vs Instruct Model

✅ FASE 3: Sistem
   ├── Advanced RAG: Hybrid Search, Semantic Chunking, Reranking
   ├── Vector Database: HNSW, Qdrant
   ├── Fine-Tuning: LoRA, QLoRA, DoRA
   └── Evaluasi: RAGAS, LLM-as-a-Judge

✅ FASE 4: Agentic AI
   ├── AI Agent: Anatomy, ReAct, Function Calling, Multi-Agent
   ├── MCP & A2A: Protokol standar agent↔tool dan agent↔agent
   ├── Orkestrasi: LangGraph, state machines, checkpointing
   ├── Evaluasi: Observability, tracing, regression testing
   └── Keamanan: Prompt injection, least privilege, guardrails

📑 STUDI KASUS
   ├── DeepSeek-V4: Hybrid Attention, mHC, Muon, CSA/HCA
   └── Qwen3: Thinking Budget, GRPO, Strong-to-Weak Distillation
```

### Langkah Berikutnya yang Disarankan

**Implementasikan proyek mini dari setiap fase:**
- Fase 1: Visualisasi embedding similarity pada 100 kalimat Bahasa Indonesia
- Fase 2: Visualisasi attention patterns pada kalimat ambigu
- Fase 3: RAG pipeline untuk dokumen PDF dengan hybrid search + reranker
- Fase 4: Agent sederhana yang bisa search web + execute code

**Pilih satu proyek portfolio:**
Pilih salah satu dari ide proyek portofolio yang relevan dengan industri (seperti Fact-Checking Pipeline, Legal Assistant dengan QLoRA, atau Academic Researcher Agent). Bangun end-to-end.

**Resource Lanjutan:**
- Paper: "Attention Is All You Need" (Vaswani et al., 2017) — baca setidaknya sekali
- Blog: The Illustrated Transformer (Jay Alammar) — visualisasi terbaik yang ada
- Course: Stanford CS224N (YouTube) — untuk yang mau lebih dalam ke riset
- Praktik: HuggingFace Spaces — deploy model dan lihat feedback nyata
- Tools: Google AI Studio / Anthropic Console — eksperimen cepat dengan LLM terbaru

---

*Selamat belajar! Dunia NLP bergerak cepat — tetapi fondasi yang kuat akan membuat siapa pun mampu beradaptasi dan terus mengikuti perkembangan teknologi dengan mudah.*
