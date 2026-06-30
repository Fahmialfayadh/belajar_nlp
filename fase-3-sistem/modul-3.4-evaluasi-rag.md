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

Menilai "apakah jawaban ini faithful?" secara otomatis adalah tugas yang kompleks — terlalu rumit untuk rule-based metrics. Solusi modern: gunakan LLM yang kuat (seperti Gemini 1.5/2 Pro, Claude 3.5 Sonnet) sebagai **LLM-as-a-Judge (Penilai Otomatis)**.

#### Masalah: Bias pada LLM-as-a-Judge
Meskipun andal, LLM-as-a-Judge memiliki bias internal yang harus kamu ketahui sebelum menggunakannya di produksi:
1. **Verbosity Bias**: LLM cenderung memberi nilai lebih tinggi pada jawaban yang *panjang dan bertele-tele*, meskipun tingkat keakuratannya sama dengan jawaban yang padat dan singkat.
2. **Self-Enhancement Bias**: Model cenderung memberi skor lebih tinggi pada jawaban yang ditulis oleh model itu sendiri dibandingkan model kompetitor (misalnya, Gemini menilai lebih ramah respon Gemini).
3. **Position Bias (dalam Pairwise)**: Jika ditanya membandingkan dua jawaban (A vs B), model cenderung lebih sering memilih opsi yang diletakkan *di urutan pertama (A)*.

#### Strategi Mitigasi Bias di Produksi

Untuk mengatasi bias tersebut, sistem evaluasi modern di 2026 menerapkan beberapa teknik berikut:

##### A. Reference Rubrics (Evaluasi Berbasis Rubrik Detil)
Alih-alih menyuruh LLM menilai secara umum (*"Beri nilai 1-5"*), kita memberikan **rubrik penilaian eksplisit** lengkap dengan definisi skor yang ketat:
- **Skor 5**: Jawaban 100% benar secara faktual, menjawab semua bagian pertanyaan, dan didukung penuh oleh dokumen rujukan.
- **Skor 3**: Jawaban benar sebagian, tetapi ada bagian kecil dari pertanyaan yang terlewat atau ada sedikit redundansi informasi.
- **Skor 1**: Jawaban mengandung halusinasi fatal atau tidak relevan dengan query.

##### B. Pairwise Comparison dengan Positional Swap
Meminta LLM membandingkan Respon A vs Respon B secara langsung. Untuk memitigasi position bias, evaluasi dijalankan **dua kali**:
- Run 1: Prompt membandingkan [Respon A, Respon B].
- Run 2: Prompt membandingkan [Respon B, Respon A].
- Jika model tidak konsisten memilih pemenang yang sama di kedua run tersebut, hasil dianggap seri (tie) atau dibuang.

##### C. Chain-of-Thought (CoT) Judgement
Prompt penilai wajib memaksa model menuliskan alasan/justifikasi langkah-demi-langkah *sebelum* mengeluarkan skor akhir JSON. Hal ini memaksa model melakukan penalaran logika yang konsisten dan mengurangi bias instan.

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

### 🧩 Latihan

**Level 1 — Recall:**
Mengapa metrik evaluasi tradisional berbasis teks seperti BLEU atau ROUGE tidak cocok digunakan untuk mengevaluasi sistem RAG pada domain faktual?

**Level 2 — Aplikasi:**
Rancang sebuah prompt untuk metrik **Answer Relevance** menggunakan pendekatan **Chain-of-Thought (CoT)**. Evaluator harus membandingkan query pengguna dengan jawaban RAG, dan memberikan skor kelayakan relevansi dalam format JSON.

**Level 3 — Eksplorasi:**
Jelaskan fenomena **Verbosity Bias** pada LLM-as-a-Judge. Bagaimana kamu merancang prompt mitigasinya agar evaluator tidak memberikan nilai tinggi secara otomatis pada jawaban yang sangat panjang?

---

### 🔬 Eksperimen Google Colab: Implementasi LLM-as-a-Judge

Copy-paste kode ini ke Google Colab (kamu memerlukan API Key Google Gemini atau penyedia LLM lainnya untuk menjalankannya secara nyata, namun kode ini menyediakan visualisasi framework evaluasinya):

```python
# ============================================================
# EKSPERIMEN: LLM-as-a-Judge untuk Faithfulness (Halusinasi)
# ============================================================

import json

# Data Uji RAG
eval_dataset = [
    {
        "query": "Berapa hari jatah cuti karyawan?",
        "context": "Setiap karyawan berhak atas 12 hari cuti per tahun yang diajukan 3 hari sebelumnya.",
        "output": "Karyawan berhak mendapat 12 hari cuti setahun dan ada bonus cuti tambahan jika berprestasi."
    }
]

def generate_evaluation_prompt(context, output):
    return f"""
    Kamu adalah evaluator RAG yang objektif. Tugasmu adalah menguji apakah output sistem mengandung halusinasi (fakta yang tidak didukung context).
    
    CONTEXT:
    {context}
    
    OUTPUT SISTEM:
    {output}
    
    Langkah Evaluasi (Chain-of-Thought):
    1. Identifikasi klaim-klaim individual dari output sistem.
    2. Verifikasi apakah setiap klaim didukung oleh context.
    3. Hitung score = jumlah klaim terverifikasi / total klaim.
    
    Keluarkan JSON dengan format:
    {{
      "reasoning": "tulis penalaran langkah demi langkah di sini",
      "claims": [
        {{"claim": "...", "supported": true/false, "explanation": "..."}}
      ],
      "score": 0.XX
    }}
    """

# Visualisasi prompt evaluasi yang dikirim ke LLM Judge
prompt = generate_evaluation_prompt(eval_dataset[0]["context"], eval_dataset[0]["output"])
print(prompt)
```

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| **Faithfulness** | Mengukur apakah jawaban bebas dari halusinasi (hanya menggunakan informasi dalam retrieved context). |
| **Answer Relevance** | Mengukur seberapa tepat jawaban merespon esensi pertanyaan yang diajukan pengguna. |
| **Context Precision/Recall** | Mengukur ketepatan dan kelengkapan dokumen yang diambil oleh modul retrieval. |
| **LLM-as-a-Judge** | Penggunaan LLM berkemampuan tinggi (seperti Gemini Pro atau Claude Sonnet) sebagai evaluator otomatis. |
| **Judge Bias** | Tantangan bias model (verbosity, position, self-enhancement); dimitigasi dengan reference rubrics dan positional swap. |

> **Takeaway utama**: Evaluasi RAG yang andal harus mengukur aspek retrieval dan generation secara terpisah. Menggunakan LLM-as-a-Judge dengan panduan rubrik terperinci (Reference Rubrics) adalah metode standar industri paling efektif untuk mengukur kualitas sistem di 2026.

---

**Selanjutnya → Fase 4: Agentic AI** — Dari sistem yang *menjawab* ke sistem yang *bertindak*.

