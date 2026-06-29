# MODUL 4.2 — MCP & A2A: Infrastruktur Standar untuk Agent

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Modul 4.1

---

### 🎯 Tujuan Belajar

- Memahami masalah integrasi N×M yang dipecahkan oleh MCP
- Menjelaskan arsitektur client-server MCP dan tiga primitif-nya (Tools, Resources, Prompts)
- Memahami perbedaan dan hubungan komplementer antara MCP dan A2A
- Mampu mengkonfigurasi MCP server dan menghubungkannya ke agent

---

### 🤔 Kenapa Ini Penting?

Di Modul 4.1, kamu sudah belajar bahwa agent perlu **tools** untuk berinteraksi dengan dunia luar. Tapi di sana, tools didefinisikan langsung dalam kode Python — satu agent, satu set tools, semuanya hardcoded.

Di dunia nyata, masalahnya jauh lebih kompleks:
- Kamu punya **puluhan tools** (database, API, file system, web search, email...)
- Kamu punya **beberapa agent** yang semuanya butuh akses ke tools berbeda
- Tools dikembangkan oleh tim yang berbeda, mungkin di bahasa pemrograman yang berbeda
- Setiap model (Claude, Gemini, GPT, Qwen) punya format function calling yang **sedikit berbeda**

Tanpa standar, kamu butuh integrasi kustom untuk setiap kombinasi agent × tool. Kalau punya 5 agent dan 10 tools, itu **50 integrasi kustom**. Ini masalah N×M yang tidak skalabel.

**MCP** mengubah ini menjadi masalah **N + M**: setiap tool cukup mengimplementasikan satu standar, dan setiap agent cukup mendukung satu standar. Selesai.

---

### 📖 Model Context Protocol (MCP): "USB-C untuk AI"

MCP adalah protokol komunikasi terbuka yang menstandardisasi cara AI agent terhubung ke data dan tools eksternal. Awalnya diciptakan oleh Anthropic pada 2024, sekarang dikelola oleh **Agentic AI Foundation (AAIF)** di bawah Linux Foundation — artinya ini bukan milik satu perusahaan, tapi standar industri terbuka.

Analogi paling tepat: **USB-C**. Sebelum USB-C, setiap perangkat punya port charger sendiri. Setelah USB-C, satu kabel untuk semuanya. MCP melakukan hal yang sama untuk koneksi agent-to-tool.

#### Arsitektur: Client-Server

```
┌──────────────────────────────────────┐
│  MCP HOST (aplikasi AI)              │
│  Contoh: Claude Desktop, Cursor,     │
│          VS Code, custom agent       │
│                                      │
│  ┌────────────┐  ┌────────────┐     │
│  │ MCP Client │  │ MCP Client │     │  ← Satu client per server
│  └─────┬──────┘  └─────┬──────┘     │
└────────┼───────────────┼─────────────┘
         │               │
    JSON-RPC 2.0    JSON-RPC 2.0
         │               │
┌────────▼────────┐ ┌────▼────────────┐
│  MCP SERVER A   │ │  MCP SERVER B   │
│  (contoh:       │ │  (contoh:       │
│   PostgreSQL)   │ │   GitHub API)   │
│                 │ │                 │
│  Ekspor:        │ │  Ekspor:        │
│  - Tools        │ │  - Tools        │
│  - Resources    │ │  - Resources    │
│  - Prompts      │ │  - Prompts      │
└─────────────────┘ └─────────────────┘
```

Tiga komponen utama:

- **MCP Host**: Aplikasi AI yang digunakan user — Claude Desktop, Cursor, VS Code, atau agent custom buatanmu
- **MCP Client**: Komponen di dalam host yang mengelola koneksi ke server-server MCP
- **MCP Server**: Service ringan yang mengekspos data atau fungsionalitas spesifik ke host. Komunikasi menggunakan **JSON-RPC 2.0** — standar web yang sudah mapan

---

### 📖 Tiga Primitif MCP

Setiap MCP server mengekspos kemampuannya melalui tiga primitif standar:

#### 1. Tools — "Apa yang bisa saya *lakukan*?"

Tools adalah fungsi yang bisa dipanggil oleh agent untuk melakukan aksi. Ini yang paling mirip dengan function calling di Modul 4.1, tapi sekarang dalam format standar.

Contoh tools yang diekspos oleh MCP server PostgreSQL:
- `query(sql: string)` — jalankan SQL query
- `list_tables()` — tampilkan semua tabel
- `describe_table(name: string)` — tampilkan skema tabel

#### 2. Resources — "Data apa yang bisa saya *baca*?"

Resources adalah data pasif yang bisa dibaca oleh agent — file, log, database row, respons API. Bedanya dengan tools: resources tidak mengubah state, hanya membaca.

Contoh resources dari MCP server file system:
- `file:///path/to/document.pdf` — konten dokumen
- `file:///logs/app.log` — log aplikasi

#### 3. Prompts — "Template apa yang tersedia?"

Prompts adalah template yang bisa dipakai untuk memandu perilaku agent di workflow tertentu. Ini seperti "resep" yang sudah disiapkan oleh server.

Contoh: MCP server untuk code review bisa menyediakan prompt template `review_pull_request` yang sudah berisi instruksi lengkap tentang apa yang harus dicek.

---

### 📖 MCP dalam Praktik: Konfigurasi

Konfigurasi MCP server dilakukan melalui JSON config. Setiap entry mendefinisikan satu server:

```json
{
  "mcpServers": {
    "postgres": {
      "command": "uvx",
      "args": ["mcp-server-postgres", "--connection-string", "postgresql://localhost/mydb"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

Perhatikan:
- Setiap server adalah proses terpisah yang dijalankan oleh host
- Komunikasi via **stdio** (stdin/stdout) — simple dan universal
- Server bisa ditulis dalam bahasa apa pun (Python, TypeScript, Go, Rust...)
- SDK resmi tersedia untuk **Python** dan **TypeScript**

#### Ekosistem MCP Server yang Sudah Ada

Karena MCP sudah menjadi standar de facto, ekosistem server-nya sangat luas:

| Kategori | Contoh Server |
|---|---|
| **Database** | PostgreSQL, Supabase, MongoDB |
| **Developer Tools** | GitHub, GitLab, Jira |
| **Produktivitas** | Slack, Google Drive, Notion |
| **Desain** | Figma, Webflow |
| **File System** | Local filesystem, S3 |
| **Web** | Brave Search, Fetch URL |

Kamu tidak perlu menulis integrasi sendiri untuk tools umum — cukup pasang MCP server yang sudah ada.

---

### 📖 Agent2Agent Protocol (A2A): Ketika Agent Bicara ke Agent

MCP menyelesaikan masalah **agent-to-tool**. Tapi ada masalah lain: bagaimana jika kamu punya **beberapa agent** yang perlu berkolaborasi, dan mereka dibangun di framework atau platform yang berbeda?

Misalnya:
- **Agent Inventory** (dibangun di LangGraph, jalan di Google Cloud)
- **Agent Shipping** (dibangun di CrewAI, jalan di AWS)
- **Agent Payment** (dibangun di custom Python, jalan di on-premise)

Bagaimana mereka berkomunikasi? Tanpa standar, kamu butuh "glue code" kustom untuk setiap pasangan agent.

**A2A (Agent2Agent)** — diperkenalkan oleh Google pada April 2025 dan sekarang di bawah Linux Foundation — menyelesaikan ini.

#### Arsitektur A2A

```
┌──────────────┐                      ┌──────────────┐
│  Agent A     │  ──── A2A ────────▶  │  Agent B     │
│  (Client)    │                      │  (Remote)    │
│              │  1. Discover via     │              │
│  "Saya butuh │     Agent Card      │  "Saya bisa  │
│   cek stok"  │  2. Create Task     │   cek stok!" │
│              │  3. Receive         │              │
│              │     Artifact        │              │
└──────────────┘                      └──────────────┘
```

Tiga konsep utama A2A:

**1. Agent Cards** — "Résumé" agent
Setiap agent mempublikasikan Agent Card dalam format JSON yang mendeskripsikan:
- Apa kemampuannya
- Aksi apa yang tersedia
- Bagaimana cara mengaksesnya

Agent Card adalah mekanisme *discovery* — bagaimana satu agent "menemukan" agent lain yang bisa membantunya.

**2. Tasks** — Unit kerja
Ketika Agent A butuh bantuan Agent B, ia membuat *task* dengan lifecycle yang jelas: `created → running → paused → completed / failed`. Ini memungkinkan operasi yang berjalan lama — agent tidak harus menunggu jawaban dalam hitungan detik.

**3. Artifacts** — Hasil kerja
Setelah task selesai, agent mengembalikan *artifacts* — output atau hasil yang bisa digunakan oleh agent pemanggil.

#### MCP vs A2A: Komplementer, Bukan Kompetitor

Ini pertanyaan yang sering membingungkan. Jawaban singkatnya:

```
┌─────────────────────────────────┐
│           AGENT                 │
│                                 │
│  Otak (LLM) ──── MCP ────→ Tools, Data, APIs
│       │                         │
│       │                         │
│       └──────── A2A ────→ Agent lain (yang punya tools sendiri)
│                                 │
└─────────────────────────────────┘
```

| | MCP | A2A |
|---|---|---|
| **Menghubungkan** | Agent ↔ Tool/Data | Agent ↔ Agent |
| **Analogi** | "Bagaimana saya pakai database ini?" | "Saya butuh Agent Shipping untuk booking pengiriman" |
| **Tujuan** | Standardisasi akses ke *resource* | Standardisasi kolaborasi antar *agent* |
| **Visibilitas** | Agent melihat detail internal tools | Agent *tidak perlu* tahu detail internal agent lain (opacity) |

Keduanya dibangun di atas teknologi web standar (HTTP, JSON-RPC, SSE) dan keduanya di bawah Linux Foundation.

---

### 💻 Kode: Membangun MCP Server Sederhana (Python)

```python
# mcp_cuaca_server.py
# MCP server sederhana yang menyediakan tool "cek cuaca"
# Jalankan: uv run mcp_cuaca_server.py

from mcp.server.fastmcp import FastMCP

# Inisialisasi MCP server
mcp = FastMCP("Cuaca Indonesia")

# --- Definisikan TOOL ---
@mcp.tool()
def cek_cuaca(kota: str) -> str:
    """Cek kondisi cuaca terkini di kota Indonesia tertentu.
    
    Args:
        kota: Nama kota di Indonesia (contoh: "Jakarta", "Bandung", "Surabaya")
    """
    # Dalam produksi, ini akan memanggil API cuaca seperti OpenWeatherMap
    # Untuk demo, kita gunakan data hardcoded
    data_cuaca = {
        "jakarta": "☀️ Cerah, 32°C, kelembaban 65%",
        "bandung": "🌤️ Berawan sebagian, 24°C, kelembaban 72%",
        "surabaya": "🌧️ Hujan ringan, 29°C, kelembaban 80%",
        "yogyakarta": "☀️ Cerah, 30°C, kelembaban 60%",
        "bali": "🌤️ Berawan, 28°C, kelembaban 75%",
    }
    
    kota_lower = kota.lower()
    if kota_lower in data_cuaca:
        return f"Cuaca di {kota}: {data_cuaca[kota_lower]}"
    else:
        return f"Data cuaca untuk '{kota}' tidak tersedia. Kota yang tersedia: {', '.join(k.title() for k in data_cuaca)}"

# --- Definisikan RESOURCE ---
@mcp.resource("cuaca://daftar-kota")
def daftar_kota() -> str:
    """Daftar kota yang tersedia untuk pengecekan cuaca."""
    return "Kota tersedia: Jakarta, Bandung, Surabaya, Yogyakarta, Bali"

# --- Definisikan PROMPT ---
@mcp.prompt()
def laporan_cuaca(kota: str) -> str:
    """Template prompt untuk membuat laporan cuaca lengkap."""
    return f"""Buatkan laporan cuaca untuk kota {kota} dengan format berikut:
    
    1. Kondisi saat ini (gunakan tool cek_cuaca)
    2. Rekomendasi pakaian
    3. Rekomendasi aktivitas outdoor
    
    Gunakan bahasa Indonesia yang natural dan ramah."""

if __name__ == "__main__":
    mcp.run()
```

Untuk menghubungkan server ini ke Claude Desktop, tambahkan ke konfigurasi:

```json
{
  "mcpServers": {
    "cuaca": {
      "command": "uv",
      "args": ["run", "mcp_cuaca_server.py"]
    }
  }
}
```

Sekarang Claude bisa langsung menggunakan tool `cek_cuaca`, membaca resource `daftar_kota`, dan menggunakan prompt template `laporan_cuaca` — tanpa kode integrasi tambahan.

> **Yang perlu kamu perhatikan**: Bandingkan kode di atas dengan kode function calling di Modul 4.1. Di Modul 4.1, kamu harus menulis loop ReAct, mengirim tool definitions ke API, mengeksekusi tool, dan mengembalikan hasilnya secara manual. Dengan MCP, semua itu ditangani oleh protokol. Kamu hanya perlu menulis logika tool-nya.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "MCP server = web server biasa"**
Tidak. MCP server biasanya berkomunikasi via **stdio** (stdin/stdout), bukan HTTP endpoint. Ini membuatnya lebih ringan dan aman (tidak perlu expose port). Untuk deployment produksi, ada mode **Streamable HTTP** yang sedang dikembangkan untuk skenario stateless dan scalable.

**Jebakan 2: "Semua MCP server aman untuk dipakai"**
Hati-hati. Karena ekosistem MCP terdesentralisasi, siapa pun bisa mempublikasikan MCP server. Selalu verifikasi sumber server sebelum menggunakannya — terutama yang membutuhkan akses ke credential atau data sensitif.

**Jebakan 3: "A2A berarti agent harus saling melihat internal state"**
Justru sebaliknya. Salah satu prinsip desain A2A adalah **opacity** — agent bisa berkolaborasi dan mendelegasikan tugas tanpa perlu mengekspos memori internal, konfigurasi tools, atau prompt system mereka ke agent lain.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan perbedaan antara MCP Tools, Resources, dan Prompts menggunakan analogi restoran: pelayan, menu, dan buku resep. Mengapa tiga primitif ini cukup untuk mengcover hampir semua skenario integrasi?

**Level 2 — Aplikasi:**
Modifikasi MCP server cuaca di atas. Tambahkan tool `perbandingan_cuaca(kota1, kota2)` yang membandingkan cuaca dua kota dan memberikan rekomendasi mana yang lebih nyaman untuk dikunjungi hari ini.

**Level 3 — Desain:**
Rancang arsitektur multi-agent e-commerce menggunakan MCP dan A2A. Identifikasi: (a) Agent apa saja yang dibutuhkan? (b) MCP server apa yang diperlukan setiap agent? (c) Bagaimana A2A menghubungkan agent-agent tersebut? Gambar diagram arsitekturnya.

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| Masalah N×M | Tanpa standar, setiap agent butuh integrasi kustom ke setiap tool — tidak skalabel |
| MCP | Protokol standar agent-to-tool; tiga primitif: Tools, Resources, Prompts |
| MCP Server | Service ringan yang mengekspos kemampuan; komunikasi via JSON-RPC 2.0 |
| A2A | Protokol standar agent-to-agent; tiga konsep: Agent Cards, Tasks, Artifacts |
| MCP vs A2A | Komplementer — MCP untuk tools, A2A untuk kolaborasi antar agent |

> **Takeaway utama**: MCP dan A2A adalah "jalan raya" yang menghubungkan agent ke dunia luar dan ke agent lain. Tanpa standar ini, setiap integrasi akan menjadi proyek custom yang mahal dan rapuh.

**Selanjutnya → Modul 4.3: Agent di Produksi** — Membangun agent itu mudah. Menjalankannya secara andal di produksi? Itu tantangan sebenarnya.
