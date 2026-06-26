# MODUL 3.4 — Evaluasi RAG: RAGAS & LLM-as-a-Judge

> **Estimasi waktu**: 2–3 jam
> **Prerequisite**: Modul 3.1 (RAG)

---

### 🎯 Tujuan Belajar

- Memahami mengapa metrik tradisional (BLEU, ROUGE) tidak cocok untuk RAG
- Menjelaskan dimensi evaluasi RAG: Faithfulness, Answer Relevance, Context Precision
- Mengimplementasikan evaluasi sederhana menggunakan LLM-as-a-Judge

---

### 📖 Masalah Evaluasi di RAG

Pertanyaan: "Siapa presiden Indonesia saat ini?"
Jawaban RAG: "Berdasarkan data kami, presiden Indonesia adalah Joko Widodo."
Jawaban Referensi (ground truth): "Prabowo Subianto."

BLEU score: mungkin 0.1 (karena kata-katanya sangat berbeda dari referensi)
Tapi masalah sebenarnya: sistem RAG mengambil dokumen lama yang tidak ter-update (Prabowo Subianto menjabat sejak Oktober 2024). Ini masalah *retrieval*, bukan masalah *generation*.

Evaluasi RAG yang baik harus bisa **mendiagnosis di mana sistem gagal** — apakah di retrieval, di generation, atau di keduanya.

---

### 📖 Dimensi Evaluasi RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment) mendefinisikan 4 metrik:

**1. Faithfulness** — Apakah jawaban hanya berisi klaim yang didukung oleh context yang diambil?
- Nilai 0-1; 1 = semua klaim dalam jawaban bisa ditelusuri ke context
- Ini mengukur apakah model "berhalusinasi" atau tidak
- Contoh gagal: model menjawab "Harga adalah Rp 50.000" padahal context tidak menyebutkan angka

**2. Answer Relevance** — Apakah jawaban menjawab pertanyaan yang ditanyakan?
- Nilai 0-1; 1 = jawaban sangat relevan dengan pertanyaan
- Ini mengukur apakah model menjawab sesuai topik
- Contoh gagal: query "cara reset password" tapi jawaban membahas "cara daftar akun"

**3. Context Precision** — Apakah chunk yang diambil memang relevan dengan pertanyaan?
- Nilai 0-1; 1 = semua chunk yang diambil relevan
- Ini mengukur kualitas retrieval
- Contoh gagal: 3 dari 5 chunk yang diambil tidak relevan sama sekali

**4. Context Recall** — Apakah semua informasi yang diperlukan ada dalam chunk yang diambil?
- Nilai 0-1; 1 = semua informasi yang ada di ground truth answer juga ada di retrieved context
- Ini mengukur kelengkapan retrieval

---

### 📖 LLM-as-a-Judge: Evaluasi Berbasis AI

Menilai "apakah jawaban ini faithful?" secara otomatis adalah tugas yang kompleks — terlalu rumit untuk rule-based metrics. Solusi modern: gunakan LLM yang kuat (Gemini 3.1 Pro, Claude Opus 4.8) sebagai *penilai otomatis*.

**Cara kerjanya (untuk Faithfulness)**:
1. Berikan ke LLM-judge: context yang diambil + jawaban yang dihasilkan
2. Prompt: *"Periksa setiap klaim dalam jawaban. Apakah setiap klaim bisa didukung oleh informasi dalam context? Beri penilaian 1-5 dan jelaskan."*
3. Parse output LLM untuk mendapatkan skor

**Kelemahan yang perlu disadari**:
- LLM-judge bisa bias terhadap gaya bahasa yang formal atau panjang
- LLM yang sama dengan yang digunakan untuk generate jawaban tidak baik sebagai judge (bias positif)
- Masih ada ketidakkonsistenan — eval yang sama bisa memberi skor berbeda jika dijalankan dua kali
- Pendekatan baru seperti **LLM-as-a-Judge with rubrics** dan **pairwise comparison** membantu mengurangi bias ini

---

### 💻 Kode: Evaluasi Faithfulness Sederhana

```python
# Implementasi LLM-as-a-Judge tanpa framework — agar kamu benar-benar paham cara kerjanya

import json

# Simulasi output RAG
context = """
Kebijakan cuti karyawan: Setiap karyawan berhak atas 12 hari cuti per tahun.
Pengajuan cuti harus dilakukan minimal 3 hari kerja sebelumnya melalui sistem HR.
Cuti tidak bisa dikumulasikan ke tahun berikutnya.
"""

jawaban_rag = """
Karyawan berhak mendapat 12 hari cuti per tahun. Pengajuan cuti dilakukan
minimal 3 hari kerja sebelumnya. Karyawan juga bisa mendapat cuti tambahan
jika kinerja sangat baik.
"""
# Perhatikan: kalimat terakhir TIDAK ADA dalam context — ini halusinasi!

def evaluasi_faithfulness(context, jawaban, llm_client):
    """
    Gunakan LLM untuk menilai apakah setiap klaim dalam jawaban
    didukung oleh context.
    
    llm_client: function yang menerima prompt dan return string response
    """
    prompt = f"""
Kamu adalah evaluator yang teliti. Tugasmu adalah menilai apakah jawaban berikut
HANYA berisi informasi yang ada dalam context yang diberikan.

CONTEXT:
{context}

JAWABAN YANG DIEVALUASI:
{jawaban}

Instruksi:
1. Pecah jawaban menjadi klaim-klaim individual
2. Untuk setiap klaim, tentukan apakah didukung oleh context (1) atau tidak (0)
3. Hitung faithfulness = jumlah_klaim_didukung / total_klaim

Respond HANYA dengan JSON format berikut:
{{
  "klaim": [
    {{"teks": "...", "didukung": true/false, "alasan": "..."}},
    ...
  ],
  "faithfulness_score": 0.XX,
  "kesimpulan": "..."
}}
"""
    response = llm_client(prompt)
    try:
        return json.loads(response)
    except:
        return {"error": "Gagal parse JSON", "raw": response}

# Contoh penggunaan (kamu perlu replace dengan API call nyata):
# Gunakan Google Gemini API, Anthropic Claude API, atau OpenAI API
def dummy_llm(prompt):
    # Ganti ini dengan:
    # - google.genai.Client().models.generate_content(...)  # Gemini
    # - anthropic.Anthropic().messages.create(...)          # Claude
    # - openai.OpenAI().chat.completions.create(...)        # OpenAI
    return json.dumps({
        "klaim": [
            {"teks": "Karyawan berhak 12 hari cuti per tahun", "didukung": True, 
             "alasan": "Tercantum eksplisit di context"},
            {"teks": "Pengajuan minimal 3 hari kerja sebelumnya", "didukung": True,
             "alasan": "Tercantum di context"},
            {"teks": "Cuti tambahan jika kinerja sangat baik", "didukung": False,
             "alasan": "Informasi ini TIDAK ADA dalam context — ini halusinasi!"}
        ],
        "faithfulness_score": 0.67,
        "kesimpulan": "2 dari 3 klaim didukung context. Ada 1 halusinasi."
    })

hasil = evaluasi_faithfulness(context, jawaban_rag, dummy_llm)
print(json.dumps(hasil, indent=2, ensure_ascii=False))
```

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Faithfulness | Apakah jawaban hanya berisi info dari context? (deteksi halusinasi) |
| Answer Relevance | Apakah jawaban menjawab pertanyaan? |
| Context Precision/Recall | Kualitas dan kelengkapan retrieval |
| LLM-as-a-Judge | Gunakan LLM kuat untuk menilai output LLM lain secara otomatis |

> **Takeaway utama**: Evaluasi yang baik adalah yang bisa mendiagnosis di mana sistem gagal — bukan hanya memberikan satu angka.

**Selanjutnya → Fase 4: Agentic AI** — Dari sistem yang *menjawab* ke sistem yang *bertindak*.
