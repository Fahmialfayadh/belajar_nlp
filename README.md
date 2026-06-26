# Modul Belajar NLP Modern (2026)

> **Tentang Modul**: Modul ini disusun berdasarkan perkembangan teknologi AI & NLP terbaru hingga Juni 2026. Kurikulum dirancang secara khusus untuk mengajarkan **fundamental NLP** yang kokoh, namun tetap **difilter secara ketat** agar materi yang dipelajari tetap relevan dengan kebutuhan industri modern saat ini (seperti era LLM multi-modal, agentic AI, dan optimasi arsitektur efisien), tanpa membuang waktu pada metode-metode klasik yang sudah sepenuhnya ditinggalkan di industri.
>
> **Level**: Menengah (Python & matematika dasar (terutama aljabar linier dan kalkulus sederhana) sudah dikuasai).
>
> **Filosofi**: Kode hanyalah alat — pemahaman teori dan intuisi di balik arsitektur adalah kunci utama menguasai NLP.

---

## Road Map

```
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
  └── AI Agents & Orkestrasi (LangGraph, MCP, A2A)
  └── Tool Use / Function Calling
  └── Multi-Agent Systems
```

Setiap fase **membangun di atas fase sebelumnya**. Jangan loncat.

---

## 📂 Struktur Modul

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
- [Modul 4.1 — AI Agents](fase-4-agentic/modul-4.1-ai-agents.md)

---

## 🏁 Roadmap Selanjutnya

### Apa yang Sudah Kamu Pelajari

```
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
   ├── AI Agent: Anatomy, Memory, Tools
   ├── ReAct: Loop Think-Act-Observe
   ├── Function Calling, MCP & A2A
   └── Multi-Agent: Kolaborasi agen spesialis
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
