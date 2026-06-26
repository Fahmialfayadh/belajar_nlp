# MODUL 4.1 — AI Agents: Dari Chatbot ke Sistem Otonom

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Seluruh Fase 3

---

### 🎯 Tujuan Belajar

- Memahami perbedaan fundamental antara LLM chatbot dan AI Agent
- Menjelaskan loop ReAct (Reason + Act) yang mendasari semua agent
- Memahami cara LLM "memanggil" alat eksternal (Function Calling)
- Mengerti konsep Multi-Agent Systems dan kapan mereka diperlukan

---

### 🤔 Kenapa Ini Penting?

Chatbot: kamu tanya → model jawab → selesai. Satu putaran.

AI Agent: kamu kasih tujuan → model *merencanakan*, *mengambil alat yang tepat*, *mengeksekusi*, *melihat hasilnya*, *merevisi rencana*, *mengeksekusi lagi*... sampai tujuan tercapai atau batas tertentu.

Ini adalah pergeseran paradigma dari LLM sebagai "oracle yang menjawab" ke LLM sebagai "agen yang bekerja." Dan ini adalah arah yang sedang dan akan terus berkembang pesat — di 2025-2026, agentic AI menjadi fokus utama hampir semua lab AI besar.

---

### 📖 Anatomi AI Agent

Sebuah AI Agent memiliki komponen:

```
┌────────────────────────────────────────────────────────┐
│                     AI AGENT                          │
│                                                        │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐  │
│  │  OTAK    │────▶│ PLANNING │────▶│   MEMORY     │  │
│  │  (LLM)   │◀────│ & REASON │     │ (Short/Long) │  │
│  └──────────┘     └──────────┘     └──────────────┘  │
│       │                                                │
│       ▼                                                │
│  ┌────────────────────────────────────────────────┐   │
│  │               TOOLS / ACTIONS                  │   │
│  │  [Web Search] [Code Exec] [DB Query] [API Call]│   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

**Otak (LLM)**: Mengambil keputusan — apa yang harus dilakukan selanjutnya?
**Planning & Reasoning**: Memecah tujuan besar menjadi langkah-langkah kecil yang bisa dieksekusi
**Memory**: 
- *Short-term* = conversation history (dalam context window)
- *Long-term* = vector database, file, database
**Tools**: Kemampuan untuk berinteraksi dengan dunia luar

---

### 📖 ReAct: Loop Berpikir-Bertindak

**ReAct** (Reason + Act, Yao et al., 2022) adalah paradigma paling fundamental untuk AI agents. Alurnya:

```
[THINK] Apa yang harus saya lakukan untuk mencapai tujuan ini?
[ACT]   Jalankan tool X dengan parameter Y
[OBS]   Lihat hasil tool X
[THINK] Berdasarkan hasil ini, apa langkah berikutnya?
[ACT]   Jalankan tool Z...
[OBS]   ...
[FINAL] Tujuan tercapai → berikan jawaban final
```

**Contoh nyata**: "Cari tahu berapa kurs dollar hari ini, lalu hitung total belanja $150 dalam rupiah"

```
[THINK] Saya perlu dua hal: (1) kurs USD-IDR hari ini, (2) kalkulator.
[ACT]   Tool: web_search("kurs dollar hari ini")
[OBS]   "1 USD = 16.450 IDR (26 Juni 2026)"
[THINK] Dapat kursnya. Sekarang hitung: 150 × 16.450.
[ACT]   Tool: calculator("150 * 16450")
[OBS]   "2.467.500"
[THINK] Sudah punya semua informasi. Susun jawaban.
[FINAL] "Dengan kurs 1 USD = Rp 16.450, belanja senilai $150 setara Rp 2.467.500"
```

Ini terlihat sederhana, tapi sangat powerful. Model bisa menjalankan loop ini ratusan kali untuk task yang kompleks.

---

### 📖 Function Calling / Tool Use: Bagaimana LLM "Memegang" Alat

Secara teknis, LLM tidak bisa langsung "menjalankan kode" atau "searching internet." Yang terjadi adalah:

1. Kamu mendefinisikan **daftar tool yang tersedia** beserta skema input/output-nya (dalam format JSON)
2. Ketika LLM memutuskan perlu menggunakan tool, ia **mengeluarkan output terstruktur** yang berisi: nama tool + parameter yang akan dikirimkan
3. **Runtime (kode Python kamu)** yang benar-benar memanggil tool tersebut
4. Hasil tool dikembalikan ke LLM sebagai observation
5. LLM melanjutkan reasoning

```
Kamu → [definisi tools] → LLM
LLM  → [tool call: {"name": "web_search", "args": {"query": "..."}}] → Kamu
Kamu → [jalankan web_search() di Python] → [hasil]
Kamu → [hasil] → LLM
LLM  → [lanjutkan reasoning] → ...
```

LLM tidak punya agency sejati — ia hanya "menulis instruksi" dan kamu yang mengeksekusinya. Tapi dari sudut pandang hasil, efeknya sama: model bisa "berinteraksi dengan dunia luar."

**Protokol standar 2025-2026**: **Model Context Protocol (MCP)** dari Anthropic menjadi standar de facto untuk menghubungkan LLM dengan tools dan data sources. MCP (sekarang dikelola Linux Foundation) menyediakan interface universal sehingga satu tool bisa digunakan oleh banyak agent tanpa integrasi kustom. Selain itu, **Agent2Agent (A2A)** protocol dari Google memungkinkan agent-agent yang dibangun di framework berbeda untuk berkomunikasi satu sama lain.

---

### 💻 Kode: AI Agent Sederhana dengan Function Calling

```python
import json
import math
from anthropic import Anthropic  # pip install anthropic

client = Anthropic()

# === STEP 1: Definisikan tools yang tersedia ===
tools = [
    {
        "name": "kalkulator",
        "description": "Hitung operasi matematika. Gunakan untuk semua perhitungan numerik.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ekspresi": {
                    "type": "string",
                    "description": "Ekspresi matematika yang valid, contoh: '150 * 16350' atau 'sqrt(144)'"
                }
            },
            "required": ["ekspresi"]
        }
    },
    {
        "name": "konversi_suhu",
        "description": "Konversi suhu antar skala (Celsius, Fahrenheit, Kelvin)",
        "input_schema": {
            "type": "object",
            "properties": {
                "nilai": {"type": "number", "description": "Nilai suhu yang akan dikonversi"},
                "dari": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]},
                "ke": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"]}
            },
            "required": ["nilai", "dari", "ke"]
        }
    }
]

# === STEP 2: Implementasi tools (ini kode Python biasa, bukan LLM!) ===
def jalankan_tool(nama_tool, args):
    if nama_tool == "kalkulator":
        try:
            # Eval ekspresi matematika (hati-hati di produksi — ini unsafe untuk input sembarang!)
            hasil = eval(args["ekspresi"], {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
            return f"Hasil: {hasil}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif nama_tool == "konversi_suhu":
        nilai, dari, ke = args["nilai"], args["dari"], args["ke"]
        # Konversi ke Celsius dulu
        if dari == "fahrenheit": celsius = (nilai - 32) * 5/9
        elif dari == "kelvin": celsius = nilai - 273.15
        else: celsius = nilai
        # Konversi dari Celsius ke target
        if ke == "fahrenheit": hasil = celsius * 9/5 + 32
        elif ke == "kelvin": hasil = celsius + 273.15
        else: hasil = celsius
        return f"{nilai}° {dari.capitalize()} = {hasil:.2f}° {ke.capitalize()}"

# === STEP 3: ReAct Loop ===
def jalankan_agent(pertanyaan_user):
    print(f"\n🎯 Task: {pertanyaan_user}\n")
    messages = [{"role": "user", "content": pertanyaan_user}]
    
    while True:
        # Kirim ke LLM
        response = client.messages.create(
            model="claude-opus-4-20250528",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )
        
        # Cek apakah LLM mau pakai tool atau sudah selesai
        if response.stop_reason == "end_turn":
            # Model sudah selesai, tidak perlu tool lagi
            jawaban = response.content[0].text
            print(f"✅ Jawaban Final: {jawaban}")
            return jawaban
        
        # Model mau pakai tool
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"🔧 Tool: {block.name}({block.input})")
                hasil = jalankan_tool(block.name, block.input)
                print(f"   Hasil: {hasil}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": hasil
                })
        
        # Kembalikan hasil tool ke model
        messages.append({"role": "user", "content": tool_results})

# Test
jalankan_agent("Berapa akar kuadrat dari 2025, dan berapa suhu 37 Celsius dalam Fahrenheit?")
```

> **Catatan**: Kode di atas menggunakan Anthropic Claude API. Kamu juga bisa menggantinya dengan Google Gemini API (`google-genai`) atau OpenAI API — konsep function calling-nya sama, hanya format JSON schema-nya sedikit berbeda.

---

### 📖 Multi-Agent Systems: Kenapa Satu Agent Tidak Cukup

Satu agent bisa menyelesaikan banyak task. Tapi ada skenario di mana beberapa agent yang berkolaborasi lebih baik:

**Skenario 1: Task yang terlalu panjang untuk satu context window**
Penelitian 100 halaman tidak muat dalam satu context window. Pecah: Agent Extractor (ekstrak fakta kunci per bab), Agent Synthesizer (gabungkan temuan), Agent Writer (tulis laporan).

**Skenario 2: Specialized expertise**
Tidak ada satu agent yang jago di segalanya. Agent Researcher (search & summarize), Agent Critic (temukan kelemahan argumen), Agent Coder (implementasi) — masing-masing dengan system prompt yang berbeda.

**Skenario 3: Parallelism**
Task yang bisa dikerjakan secara paralel. Agent A riset tentang topik 1, Agent B riset topik 2, Agent C riset topik 3 — semua bersamaan, lalu hasilnya digabungkan.

**Framework yang sering digunakan (2025-2026)**:
- **LangGraph**: Definisikan agents sebagai nodes dalam graph, edges adalah aliran informasi/kontrol — framework paling mature
- **Google ADK (Agent Development Kit)**: Framework dari Google untuk membangun multi-agent systems dengan integrasi Gemini
- **OpenAI Agents SDK**: Framework baru dari OpenAI untuk agent orchestration
- **CrewAI**: Abstraksi tingkat tinggi untuk "tim" agent dengan peran yang jelas
- **Autogen (Microsoft)**: Lebih dekat ke "agen yang berdialog satu sama lain"

---

### ⚠️ Jebakan Umum

**Jebakan 1: "Lebih banyak agent = lebih baik"**
Agent overhead adalah nyata: setiap agent menambah latency, biaya, dan kompleksitas debugging. Mulai dengan satu agent, tambah hanya jika ada bottleneck yang jelas.

**Jebakan 2: "Agent bisa dipercaya sepenuhnya untuk mengeksekusi aksi permanen"**
Agent masih bisa membuat kesalahan. Untuk aksi yang tidak bisa di-undo (kirim email ke klien, hapus data, transaksi finansial), selalu ada konfirmasi manusia (human-in-the-loop) sebelum eksekusi.

**Jebakan 3: "Prompt injection tidak relevan untuk internal tools"**
Jika agent kamu membaca dokumen dari internet atau dari user lain, dokumen tersebut bisa mengandung instruksi tersembunyi yang "membajak" agent — ini disebut **prompt injection**. Ini adalah ancaman keamanan nyata yang perlu dimitigasi.

---

### 🧩 Latihan

**Level 1 — Recall:**
Gambar diagram ReAct loop untuk skenario ini: "Cari harga saham Apple hari ini, bandingkan dengan harga 1 tahun lalu, dan tentukan persentase perubahannya." Berapa langkah Thought-Action-Observation yang dibutuhkan?

**Level 2 — Aplikasi:**
Tambahkan tool baru ke agent di atas: `konversi_mata_uang(jumlah, dari, ke)` yang menggunakan nilai tukar hardcoded. Buat task yang memaksa agent menggunakan kombinasi tools: kalkulator + konversi mata uang.

**Level 3 — Eksplorasi:**
Cari dan baca tentang **"LLM Agents Benchmark"** seperti SWE-bench (agent yang solve GitHub issues) atau AgentBench. Bagaimana performa model terbaik saat ini? Apa task yang masih sulit bagi agent AI? Apa implikasinya untuk pengembangan agent di masa depan?

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| AI Agent | LLM + Loop + Tools; dari "menjawab" ke "menyelesaikan masalah" |
| ReAct | Think → Act → Observe → Think → ...; loop fundamental semua agent |
| Function Calling | LLM "menulis instruksi" tool, kode Python yang mengeksekusinya |
| MCP | Model Context Protocol — standar universal untuk menghubungkan LLM ke tools |
| Multi-Agent | Beberapa agent spesialis berkolaborasi; gunakan hanya jika satu agent tidak cukup |

> **Takeaway utama**: AI Agent adalah LLM yang diberi kemampuan untuk "bertindak" di dunia nyata — tapi tetap membutuhkan desain yang cermat agar aman dan andal.
